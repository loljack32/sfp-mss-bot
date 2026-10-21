# ============================================================
# WALK-FORWARD BACKTESTER
# SFP + MSS BOT
# ============================================================
#
# Прогоняет ту же логику, что и live-сканер, по историческим
# свечам OKX без look-ahead:
#
# - свинг считается существующим только после подтверждения
#   (index + right <= текущий бар);
# - сетап SFP+MSS оценивается только на барах, где он свеж
#   по тем же правилам, что и в live (MAX_SIGNAL_AGE_BARS);
# - вход - закрытие сигнальной свечи, выход моделируется по
#   следующим свечам: половина позиции на TP1, половина на TP2,
#   стоп общий; если SL и TP в одной свече - засчитывается SL;
# - из результата каждой сделки вычитается COST_R на комиссии
#   и проскальзывание.
#
# Использование:
#   python backtest.py --symbols BTC-USDT,ETH-USDT --bars 1500
#   python backtest.py --bars 3000 --horizon 128 --csv trades.csv
#
# ВНИМАНИЕ: бэктест не гарантирует будущую доходность. Это
# инструмент проверки логики и подбора порогов, а не обещание
# "правильных сигналов".
# ============================================================

from __future__ import annotations

import argparse
import csv
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import (
    SYMBOLS,
    HTF_TIMEFRAME,
    ENTRY_TIMEFRAME,
    DEFAULT_RISK_PERCENT,
    DEFAULT_LEVERAGE,
    SWING_RIGHT,
    HTF_SWING_LEFT,
    HTF_SWING_RIGHT,
    MAX_SIGNAL_AGE_BARS,
)

from core.okx import OKXClient, normalize_instrument_id
from core.structure import build_structure, get_structure_state
from core.sfp import find_sfps
from core.mss import find_mss_after_sfp
from core.signals import generate_signal


COST_R = 0.05  # примерные комиссии + проскальзывание в R на сделку


# ============================================================
# HELPERS
# ============================================================

def _bar_duration(df: pd.DataFrame) -> pd.Timedelta:
    diffs = df["timestamp"].diff().dropna()
    return diffs.median()


def _confirmed_swings(structure, upto_index: int, right: int):
    """Свинги, которые к бару upto_index уже подтверждены."""
    return [s for s in structure if s.index + right <= upto_index]


def _htf_state_asof(
    htf_df: pd.DataFrame,
    htf_structure,
    ltf_close_time: pd.Timestamp,
    htf_duration: pd.Timedelta,
):
    """Состояние HTF-структуры только по закрытым HTF-свечам."""
    closed = int(((htf_df["timestamp"] + htf_duration) <= ltf_close_time).sum())
    last_closed_index = closed - 1

    if last_closed_index < 0:
        return None

    confirmed = _confirmed_swings(
        htf_structure, last_closed_index, HTF_SWING_RIGHT
    )
    if not confirmed:
        return None

    return get_structure_state(confirmed)


def _invalidated_asof(
    closes: np.ndarray,
    direction: str,
    extreme: float,
    from_index: int,
    upto_index: int,
) -> bool:
    """Было ли закрытие за экстремумом свипа на отрезке (from, upto]."""
    if upto_index <= from_index:
        return False

    window = closes[from_index + 1: upto_index + 1]
    if window.size == 0:
        return False

    if direction == "LONG":
        return bool((window < extreme).any())
    return bool((window > extreme).any())


def _simulate_outcome(
    df: pd.DataFrame,
    signal,
    start_index: int,
    horizon: int,
) -> dict:
    """
    Моделирует сделку по свечам после сигнальной.

    Половина объёма закрывается на TP1, половина - на TP2.
    Если вторая половина доходит до исходного стопа, она в -1R.
    При одновременном касании SL и TP в одной свече - SL первым.
    """
    entry = signal.entry
    stop = signal.stop_loss
    tp1 = signal.tp1
    tp2 = signal.tp2
    risk_distance = abs(entry - stop)
    is_long = signal.direction == "LONG"

    end = min(len(df) - 1, start_index + horizon)
    tp1_done = False

    def mtm_r(price: float) -> float:
        if is_long:
            return (price - entry) / risk_distance
        return (entry - price) / risk_distance

    for j in range(start_index + 1, end + 1):
        high = float(df.iloc[j]["high"])
        low = float(df.iloc[j]["low"])

        if is_long:
            hit_stop = low <= stop
            hit_tp1 = high >= tp1
            hit_tp2 = high >= tp2
        else:
            hit_stop = high >= stop
            hit_tp1 = low <= tp1
            hit_tp2 = low <= tp2

        if not tp1_done:
            if hit_stop:
                return {"result_r": -1.0 - COST_R, "exit": "SL", "bars": j - start_index}
            if hit_tp1:
                tp1_done = True
                if hit_tp2:
                    blended = 0.5 * signal.rr_tp1 + 0.5 * signal.rr_tp2
                    return {"result_r": blended - COST_R, "exit": "TP2", "bars": j - start_index}
        else:
            if hit_stop:
                blended = 0.5 * signal.rr_tp1 - 0.5
                return {"result_r": blended - COST_R, "exit": "TP1+SL", "bars": j - start_index}
            if hit_tp2:
                blended = 0.5 * signal.rr_tp1 + 0.5 * signal.rr_tp2
                return {"result_r": blended - COST_R, "exit": "TP2", "bars": j - start_index}

    last_close = float(df.iloc[end]["close"])
    if tp1_done:
        result = 0.5 * signal.rr_tp1 + 0.5 * mtm_r(last_close)
        return {"result_r": result - COST_R, "exit": "TIMEOUT_TP1", "bars": end - start_index}

    return {"result_r": mtm_r(last_close) - COST_R, "exit": "TIMEOUT", "bars": end - start_index}


# ============================================================
# SINGLE SYMBOL BACKTEST
# ============================================================

def backtest_symbol(
    client: OKXClient,
    symbol: str,
    ltf_bars: int,
    horizon: int,
    warmup: int,
    cooldown_bars: int,
    balance: float,
    risk_percent: float,
    leverage: int,
) -> List[dict]:
    symbol = normalize_instrument_id(symbol)

    df = client.get_candles_history(symbol, ENTRY_TIMEFRAME, ltf_bars)
    htf_bars_needed = ltf_bars // 16 + 300
    df_htf = client.get_candles_history(symbol, HTF_TIMEFRAME, htf_bars_needed)

    ltf_duration = _bar_duration(df)
    htf_duration = _bar_duration(df_htf)

    structure_full = build_structure(df)
    htf_structure_full = build_structure(
        df_htf, left=HTF_SWING_LEFT, right=HTF_SWING_RIGHT
    )

    # Все сетапы считаются один раз по полному ряду: детекторы
    # причинны (SFP зависит только от свечей до свипа, MSS - до
    # пробоя), а доступность сетапа на баре i проверяется ниже
    # через подтверждения свингов и свежесть.
    sfps = find_sfps(df, structure_full)
    pairs = []
    for sfp in sfps:
        mss = find_mss_after_sfp(df, sfp, structure_full)
        if mss is not None:
            pairs.append((sfp, mss))

    closes = df["close"].to_numpy(dtype=float)

    trades: List[dict] = []
    used_fingerprints = set()
    last_trade_bar = -10**9

    for i in range(warmup, len(df) - 1):
        if i - last_trade_bar < cooldown_bars:
            continue

        # Кандидаты: MSS уже случился и ещё свеж к бару i.
        candidates = []
        for sfp, mss in pairs:
            if mss.break_index > i:
                continue
            if i - mss.break_index > MAX_SIGNAL_AGE_BARS:
                continue
            # Все свинги сетапа должны быть подтверждены к бару i,
            # иначе в live-сканере их бы ещё не существовало.
            if sfp.source_swing_index + SWING_RIGHT > i:
                continue
            if mss.structure_index + SWING_RIGHT > i:
                continue
            if _invalidated_asof(
                closes, sfp.direction, sfp.sweep_extreme, sfp.sweep_index, i
            ):
                continue
            if _invalidated_asof(
                closes, sfp.direction, sfp.sweep_extreme, mss.break_index, i
            ):
                continue
            candidates.append((sfp, mss))

        if not candidates:
            continue

        candidates.sort(key=lambda pair: pair[1].break_index)
        sfp, mss = candidates[-1]

        fingerprint = (
            f"{symbol}_{sfp.direction}"
            f"_{sfp.sweep_timestamp.isoformat()}"
            f"_{mss.break_timestamp.isoformat()}"
        )
        if fingerprint in used_fingerprints:
            continue

        ltf_close_time = df.iloc[i]["timestamp"] + ltf_duration
        htf_state = _htf_state_asof(
            df_htf, htf_structure_full, ltf_close_time, htf_duration
        )
        if htf_state is None or htf_state.trend not in {"BULLISH", "BEARISH"}:
            continue

        df_slice = df.iloc[: i + 1].reset_index(drop=True)
        # Индексы свингов в срезе совпадают с полным рядом (срез - префикс).
        structure_asof = _confirmed_swings(structure_full, i, SWING_RIGHT)

        signal = generate_signal(
            symbol=symbol,
            timeframe=ENTRY_TIMEFRAME,
            df=df_slice,
            sfp=sfp,
            mss=mss,
            structure=structure_asof,
            htf_state=htf_state,
            balance=balance,
            risk_percent=risk_percent,
            leverage=leverage,
        )
        if signal is None:
            continue

        outcome = _simulate_outcome(df, signal, i, horizon)

        used_fingerprints.add(fingerprint)
        last_trade_bar = i

        trades.append(
            {
                "symbol": symbol,
                "time": str(signal.timestamp),
                "direction": signal.direction,
                "setup_type": signal.setup_type,
                "score": round(float(signal.signal_score), 1),
                "entry": signal.entry,
                "stop_loss": signal.stop_loss,
                "tp1": signal.tp1,
                "tp2": signal.tp2,
                "exit": outcome["exit"],
                "bars_held": outcome["bars"],
                "result_r": round(float(outcome["result_r"]), 3),
            }
        )

    return trades


# ============================================================
# REPORT
# ============================================================

def summarize(trades: List[dict], title: str) -> None:
    print(f"\n--- {title} ---")
    if not trades:
        print("Сделок нет.")
        return

    results = np.array([t["result_r"] for t in trades], dtype=float)
    wins = results[results > 0]
    losses = results[results <= 0]

    total_r = float(results.sum())
    winrate = 100.0 * len(wins) / len(results)
    gross_win = float(wins.sum())
    gross_loss = abs(float(losses.sum()))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

    equity = np.cumsum(results)
    peak = np.maximum.accumulate(equity)
    max_dd = float((peak - equity).max()) if len(equity) else 0.0

    print(f"Сделок:        {len(results)}")
    print(f"Winrate:       {winrate:.1f}%")
    print(f"Средний R:     {results.mean():+.3f}R")
    print(f"Итого:         {total_r:+.2f}R")
    print(f"Profit factor: {profit_factor:.2f}")
    print(f"Макс. просадка: {max_dd:.2f}R")


def main() -> int:
    parser = argparse.ArgumentParser(description="Walk-forward backtest SFP+MSS")
    parser.add_argument("--symbols", default=",".join(SYMBOLS))
    parser.add_argument("--bars", type=int, default=1500)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--warmup", type=int, default=150)
    parser.add_argument("--cooldown-bars", type=int, default=16)
    parser.add_argument("--balance", type=float, default=10000.0)
    parser.add_argument("--risk", type=float, default=DEFAULT_RISK_PERCENT)
    parser.add_argument("--leverage", type=int, default=DEFAULT_LEVERAGE)
    parser.add_argument("--csv", default="")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    all_trades: List[dict] = []

    with OKXClient() as client:
        for symbol in symbols:
            try:
                trades = backtest_symbol(
                    client=client,
                    symbol=symbol,
                    ltf_bars=args.bars,
                    horizon=args.horizon,
                    warmup=args.warmup,
                    cooldown_bars=args.cooldown_bars,
                    balance=args.balance,
                    risk_percent=args.risk,
                    leverage=args.leverage,
                )
            except Exception as exc:
                print(f"[{symbol}] ERROR: {type(exc).__name__}: {exc}")
                continue

            summarize(trades, symbol)
            all_trades.extend(trades)

    summarize(all_trades, "ИТОГО")

    if args.csv and all_trades:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_trades[0].keys()))
            writer.writeheader()
            writer.writerows(all_trades)
        print(f"\nСделки записаны в {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
