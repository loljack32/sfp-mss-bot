# ============================================================
# FORWARD TEST EVALUATOR
# ============================================================
# Читает журнал живых сигналов (data/forward_journal.jsonl,
# пишется сканером в момент отправки сигнала в Telegram) и
# считает исход каждого сигнала на невиданных живых данных
# OKX по тем же правилам выхода, что backtest.py
# (_simulate_outcome):
#   - горизонт 96 баров 15m (24 часа);
#   - половина объема закрывается на TP1, половина на TP2;
#   - после хода +1R стоп переносится в безубыток (со след. бара);
#   - при одновременном касании SL и TP в одной свече - SL первым;
#   - из результата каждой закрытой сделки вычитается COST_R.
#
# Использование:
#   python forward_eval.py [--journal PATH] [--out-dir DIR]
#                          [--horizon 96]
# Результаты: forward_trades.csv и forward_summary.json в out-dir.
# ============================================================

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

OKX_HISTORY_CANDLES = "https://www.okx.com/api/v5/market/history-candles"
BAR_MS = 15 * 60 * 1000
COST_R = 0.05
BREAKEVEN_TRIGGER_R = 1.0

# Базовая точка отсчета: бэктест v3 на 12 месяцах (коммит bb243f4).
BASELINE_V3 = {"n": 170, "total_r": 8.93, "win_rate": 0.56, "pf": 1.12}


def load_journal(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = (rec.get("ts_utc"), rec.get("symbol"), rec.get("direction"))
        if key in seen:
            continue
        seen.add(key)
        records.append(rec)
    records.sort(key=lambda r: r.get("ts_utc", ""))
    return records


def fetch_window_candles(inst_id: str, start_ms: int, bars_needed: int) -> list[dict]:
    """
    Свечи 15m, начиная с бара, следующего за сигнальным.

    history-candles с параметром after возвращает до 100 свечей
    СТРОГО РАНЬШЕ указанного ts, от новых к старым - запрашиваем
    окно, закрывающееся на (start + bars_needed) баров.
    """
    after_ms = start_ms + (bars_needed + 2) * BAR_MS
    resp = requests.get(
        OKX_HISTORY_CANDLES,
        params={"instId": inst_id, "bar": "15m", "after": str(after_ms), "limit": "100"},
        timeout=20,
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("code") != "0":
        raise RuntimeError(f"OKX error: {payload.get('msg')}")
    candles = []
    for row in payload.get("data", []):
        ts = int(row[0])
        if ts <= start_ms:
            continue
        candles.append(
            {
                "ts": ts,
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "confirmed": row[8] == "1" if len(row) > 8 else True,
            }
        )
    candles.sort(key=lambda c: c["ts"])
    return candles[:bars_needed]


def simulate(rec: dict, candles: list[dict], horizon: int) -> dict:
    """Тот же алгоритм исхода, что backtest._simulate_outcome."""
    entry = float(rec["entry"])
    stop = float(rec["stop_loss"])
    tp1 = float(rec["tp1"])
    tp2 = float(rec["tp2"])
    rr_tp1 = float(rec.get("rr_tp1") or 1.0)
    rr_tp2 = float(rec.get("rr_tp2") or 2.0)
    risk = abs(entry - stop)
    is_long = rec["direction"] == "LONG"
    if risk <= 0:
        return {"status": "ERROR", "exit": "BAD_RISK", "result_r": None, "bars": 0}

    def mtm_r(price: float) -> float:
        return (price - entry) / risk if is_long else (entry - price) / risk

    tp1_done = False
    be_armed = False
    active_stop = stop

    for j, c in enumerate(candles):
        high, low = c["high"], c["low"]
        if is_long:
            hit_stop = low <= active_stop
            hit_tp1 = high >= tp1
            hit_tp2 = high >= tp2
            fav_r = (high - entry) / risk
        else:
            hit_stop = high >= active_stop
            hit_tp1 = low <= tp1
            hit_tp2 = low <= tp2
            fav_r = (entry - low) / risk

        if not tp1_done:
            if hit_stop:
                if be_armed:
                    return {"status": "CLOSED", "exit": "BE", "result_r": 0.0 - COST_R, "bars": j + 1}
                return {"status": "CLOSED", "exit": "SL", "result_r": -1.0 - COST_R, "bars": j + 1}
            if hit_tp1:
                tp1_done = True
                if hit_tp2:
                    blended = 0.5 * rr_tp1 + 0.5 * rr_tp2
                    return {"status": "CLOSED", "exit": "TP2", "result_r": blended - COST_R, "bars": j + 1}
        else:
            if hit_stop:
                if be_armed:
                    blended = 0.5 * rr_tp1 + 0.0
                    return {"status": "CLOSED", "exit": "TP1+BE", "result_r": blended - COST_R, "bars": j + 1}
                blended = 0.5 * rr_tp1 - 0.5
                return {"status": "CLOSED", "exit": "TP1+SL", "result_r": blended - COST_R, "bars": j + 1}
            if hit_tp2:
                blended = 0.5 * rr_tp1 + 0.5 * rr_tp2
                return {"status": "CLOSED", "exit": "TP2", "result_r": blended - COST_R, "bars": j + 1}

        if not be_armed and fav_r >= BREAKEVEN_TRIGGER_R:
            be_armed = True
            active_stop = entry

    if len(candles) >= horizon:
        last_close = candles[horizon - 1]["close"]
        if tp1_done:
            result = 0.5 * rr_tp1 + 0.5 * mtm_r(last_close)
            return {"status": "CLOSED", "exit": "TIMEOUT_TP1", "result_r": result - COST_R, "bars": horizon}
        return {"status": "CLOSED", "exit": "TIMEOUT", "result_r": mtm_r(last_close) - COST_R, "bars": horizon}

    last_close = candles[-1]["close"] if candles else entry
    return {
        "status": "OPEN",
        "exit": "OPEN_TP1" if tp1_done else "OPEN",
        "result_r": None,
        "mtm_r": round(mtm_r(last_close), 3),
        "bars": len(candles),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", default="data/forward_journal.jsonl")
    parser.add_argument("--out-dir", default="data/forward")
    parser.add_argument("--horizon", type=int, default=96)
    args = parser.parse_args()

    records = load_journal(Path(args.journal))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trades = []
    for rec in records:
        symbol = rec["symbol"]
        inst_id = symbol if symbol.endswith("-SWAP") else f"{symbol}-SWAP"
        ts = datetime.fromisoformat(rec["ts_utc"])
        signal_bar_open_ms = int(ts.timestamp() * 1000) // BAR_MS * BAR_MS
        try:
            candles = fetch_window_candles(inst_id, signal_bar_open_ms, args.horizon)
            outcome = simulate(rec, candles, args.horizon)
        except Exception as exc:
            outcome = {"status": "ERROR", "exit": f"FETCH_ERROR: {exc}", "result_r": None, "bars": 0}
        trades.append({**rec, **outcome})
        time.sleep(0.15)

    closed = [t for t in trades if t["status"] == "CLOSED" and t["result_r"] is not None]
    gross_profit = sum(t["result_r"] for t in closed if t["result_r"] > 0)
    gross_loss = -sum(t["result_r"] for t in closed if t["result_r"] < 0)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "journal": str(args.journal),
        "horizon_bars": args.horizon,
        "n_signals": len(trades),
        "n_closed": len(closed),
        "n_open": sum(1 for t in trades if t["status"] == "OPEN"),
        "n_error": sum(1 for t in trades if t["status"] == "ERROR"),
        "total_r": round(sum(t["result_r"] for t in closed), 2) if closed else 0.0,
        "avg_r": round(sum(t["result_r"] for t in closed) / len(closed), 3) if closed else None,
        "win_rate": round(sum(1 for t in closed if t["result_r"] > 0) / len(closed), 3) if closed else None,
        "pf": round(gross_profit / gross_loss, 3) if gross_loss > 0 else (None if gross_profit == 0 else float("inf")),
        "baseline_backtest_v3": BASELINE_V3,
    }

    with open(out_dir / "forward_trades.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["ts_utc", "symbol", "direction", "setup_type", "entry", "stop_loss",
                  "tp1", "tp2", "signal_score", "status", "exit", "result_r", "bars"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(trades)

    with open(out_dir / "forward_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
