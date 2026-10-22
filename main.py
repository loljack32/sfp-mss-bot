# ============================================================
# MAIN SCANNER PIPELINE (FAST & CLEAN EXECUTION)
# SFP + MSS BOT
# ============================================================

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import (
    SYMBOLS,
    DYNAMIC_TOP_SYMBOLS,
    TOP_SYMBOLS_COUNT,
    HTF_TIMEFRAME,
    ENTRY_TIMEFRAME,
    CANDLE_LIMIT,
    DATA_DIR,
    SIGNAL_STATE_FILE,
    DEFAULT_RISK_PERCENT,
    DEFAULT_LEVERAGE,
    SIGNAL_COOLDOWN_MINUTES,
    SIGNAL_DEDUPE_RETENTION_DAYS,
    ENABLE_COUNTER_TREND_SIGNALS,
    ENABLE_REGIME_FILTER,
    REGIME_SYMBOL,
    REGIME_TIMEFRAME,
    REGIME_EMA_PERIOD,
)

from core.okx import OKXClient, OKXError, normalize_instrument_id
from core.structure import analyze_structure
from core.sfp import find_sfps, is_sfp_invalidated
from core.mss import find_latest_mss, is_mss_invalidated
from core.signals import generate_signal, get_signal_failure_reasons
from core.filters import (
    evaluate_setup,
    compute_htf_ema_dist_atr,
    compute_regime_uptrend,
)
from core.telegram import TelegramNotifier


# ============================================================
# SETTINGS & STATE MANAGEMENT
# ============================================================

USER_SETTINGS_FILE = "data/user_settings.json"
FORWARD_JOURNAL_FILE = "data/forward_journal.jsonl"


def load_user_settings(filepath: str = USER_SETTINGS_FILE) -> Tuple[float, float, int]:
    """
    Загружает: (баланс, процент риска, кредитное плечо).
    """
    path = Path(filepath)
    if not path.exists():
        return 10000.0, float(DEFAULT_RISK_PERCENT), int(DEFAULT_LEVERAGE)

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            balance = float(data.get("balance", 10000.0))
            risk = float(data.get("risk_percent", DEFAULT_RISK_PERCENT))
            leverage = int(data.get("leverage", DEFAULT_LEVERAGE))
            return balance, risk, leverage
    except Exception:
        return 10000.0, float(DEFAULT_RISK_PERCENT), int(DEFAULT_LEVERAGE)


def load_signal_state(filepath: str = SIGNAL_STATE_FILE) -> Dict[str, str]:
    path = Path(filepath)
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_signal_state(state: Dict[str, str], filepath: str = SIGNAL_STATE_FILE) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        print(f"[ERROR] Failed to save state file: {exc}")


def append_forward_journal(signal, filepath: str = FORWARD_JOURNAL_FILE) -> None:
    """
    Дописывает отправленный сигнал в журнал форвард-теста.

    Журнал - единственный источник точных цен живых сигналов:
    по нему forward_eval.py считает исходы на невиданных данных
    по тем же правилам выхода, что и backtest.py. Ошибка записи
    никогда не должна ронять сканер, поэтому все исключения глушатся.
    """
    try:
        record = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "symbol": signal.symbol,
            "direction": signal.direction,
            "timeframe": signal.timeframe,
            "setup_type": getattr(signal, "setup_type", "TREND"),
            "entry": float(signal.entry),
            "stop_loss": float(signal.stop_loss),
            "tp1": float(signal.tp1),
            "tp2": float(signal.tp2),
            "rr_tp1": float(signal.rr_tp1),
            "rr_tp2": float(signal.rr_tp2),
            "signal_score": float(signal.signal_score),
        }
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"[ERROR] Failed to append forward journal: {exc}")


def commit_forward_journal() -> None:
    """
    В GitHub Actions коммитит журнал форвард-теста в репозиторий.

    Шаг workflow коммитит только data/signals_state.json, а файл
    workflow через интеграцию не обновить (нет права workflows),
    поэтому сканер пушит журнал сам токеном раннера. Вне Actions
    (локальные запуски) ничего не делает. Ошибки только логируются:
    сбой коммита журнала не должен ронять скан.
    """
    if os.getenv("GITHUB_ACTIONS") != "true":
        return
    try:
        def git(*args: str) -> subprocess.CompletedProcess:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, timeout=60
            )

        git("config", "user.name", "github-actions[bot]")
        git("config", "user.email", "github-actions[bot]@users.noreply.github.com")
        git("add", FORWARD_JOURNAL_FILE)
        staged = git("diff", "--staged", "--quiet")
        if staged.returncode == 0:
            return
        commit = git("commit", "-m", "chore: forward journal [skip ci]")
        if commit.returncode != 0:
            print(f"[ERROR] Forward journal commit failed: {commit.stderr.strip()}")
            return
        push = git("push")
        if push.returncode != 0:
            # Не страшно: последующий шаг workflow все равно запушит
            # все локальные коммиты вместе со стейтом.
            print(f"[WARN] Forward journal push deferred: {push.stderr.strip()}")
    except Exception as exc:
        print(f"[ERROR] Forward journal commit error: {exc}")


def setup_fingerprint(symbol: str, sfp, mss) -> str:
    """
    Уникальный отпечаток конкретного сетапа SFP+MSS.

    Один и тот же сетап нельзя отправлять дважды, даже если
    кулдаун по символу давно истёк.
    """
    return (
        f"{symbol}_{sfp.direction}"
        f"_SFP:{sfp.sweep_timestamp.isoformat()}@{sfp.level:.10g}"
        f"_MSS:{mss.break_timestamp.isoformat()}@{mss.structure_level:.10g}"
    )


def is_on_cooldown(
    symbol: str,
    direction: str,
    state: Dict[str, str],
    cooldown_minutes: int = SIGNAL_COOLDOWN_MINUTES,
    sfp=None,
    mss=None,
) -> bool:
    # Конкретный сетап уже отправлялся - повторно не шлём никогда.
    if sfp is not None and mss is not None:
        if setup_fingerprint(symbol, sfp, mss) in state:
            return True

    key = f"{symbol}_{direction}"
    last_sent_str = state.get(key)
    if not last_sent_str:
        return False

    try:
        last_sent = datetime.fromisoformat(last_sent_str)
        now = datetime.now(timezone.utc)
        elapsed_minutes = (now - last_sent).total_seconds() / 60.0
        return elapsed_minutes < cooldown_minutes
    except Exception:
        return False


def record_signal_sent(
    symbol: str,
    direction: str,
    state: Dict[str, str],
    sfp=None,
    mss=None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    state[f"{symbol}_{direction}"] = now

    if sfp is not None and mss is not None:
        state[setup_fingerprint(symbol, sfp, mss)] = now


def prune_signal_state(
    state: Dict[str, str],
    retention_days: int = SIGNAL_DEDUPE_RETENTION_DAYS,
) -> Dict[str, str]:
    """Удаляет записи старше retention_days, чтобы стейт не рос вечно."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    pruned: Dict[str, str] = {}

    for key, value in state.items():
        try:
            ts = datetime.fromisoformat(value)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= cutoff:
                pruned[key] = value
        except Exception:
            continue

    return pruned


# ============================================================
# MAIN SCANNER LOOP
# ============================================================

def run_scanner() -> None:
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    state = load_signal_state()
    account_balance, risk_percent, leverage = load_user_settings()
    telegram = TelegramNotifier()

    print("=" * 65)
    print(f"Starting SFP + MSS Scanner at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"Account: ${account_balance:,.2f} | Risk: {risk_percent:.2f}% | Leverage: {leverage}x")
    print(f"HTF: {HTF_TIMEFRAME} | LTF: {ENTRY_TIMEFRAME} | Counter-Trend: {ENABLE_COUNTER_TREND_SIGNALS}")
    print("=" * 65)

    stats = {
        "total_scanned": 0,
        "skipped_htf": 0,
        "no_sfp": 0,
        "no_mss": 0,
        "invalidated": 0,
        "cooldown": 0,
        "filter_rejected": 0,
        "filter_reasons": {},
        "signals_trend": 0,
        "signals_counter_trend": 0,
        "signals_sent": 0,
    }

    with OKXClient() as client:
        if DYNAMIC_TOP_SYMBOLS:
            try:
                symbols_to_scan = client.get_top_volume_symbols(top_n=TOP_SYMBOLS_COUNT)
            except Exception:
                symbols_to_scan = [
                    normalize_instrument_id(s) for s in SYMBOLS
                ]
        else:
            symbols_to_scan = [
                normalize_instrument_id(s) for s in SYMBOLS
            ]

        total = len(symbols_to_scan)
        stats["total_scanned"] = total
        print(f"Scanning {total} liquid symbols from OKX...")

        # Режим широкого рынка - один раз на весь скан: бенчмарк
        # выше/ниже своей EMA. None - режим неизвестен, фильтр
        # в evaluate_setup пропускается.
        regime_uptrend = None
        if ENABLE_REGIME_FILTER:
            try:
                df_regime = client.get_candles(
                    symbol=normalize_instrument_id(REGIME_SYMBOL),
                    timeframe=REGIME_TIMEFRAME,
                    limit=max(REGIME_EMA_PERIOD + 50, CANDLE_LIMIT),
                    closed_only=True,
                )
                regime_uptrend = compute_regime_uptrend(
                    df_regime, REGIME_EMA_PERIOD
                )
            except (OKXError, Exception):
                regime_uptrend = None

        for symbol in symbols_to_scan:
            try:
                time.sleep(0.04)  # Минимальная пауза для защиты от rate-limit OKX

                df_htf = client.get_candles(
                    symbol=symbol,
                    timeframe=HTF_TIMEFRAME,
                    limit=CANDLE_LIMIT,
                    closed_only=True,
                )

                df_ltf = client.get_candles(
                    symbol=symbol,
                    timeframe=ENTRY_TIMEFRAME,
                    limit=CANDLE_LIMIT,
                    closed_only=True,
                )
            except (OKXError, Exception):
                continue

            # Анализ структуры
            _, htf_state = analyze_structure(df_htf, htf=True)
            structure_ltf, _ = analyze_structure(df_ltf, htf=False)
            htf_dist = compute_htf_ema_dist_atr(df_htf)

            if htf_state.trend not in {"BULLISH", "BEARISH"}:
                stats["skipped_htf"] += 1
                continue

            # Поиск SFP
            sfps = find_sfps(df_ltf, structure_ltf)
            if not sfps:
                stats["no_sfp"] += 1
                continue

            # Поиск MSS
            result = find_latest_mss(df_ltf, sfps, structure_ltf)
            if result is None:
                stats["no_mss"] += 1
                continue

            sfp, mss = result

            # Проверка инвалидации
            if is_sfp_invalidated(df_ltf, sfp) or is_mss_invalidated(df_ltf, sfp, mss):
                stats["invalidated"] += 1
                continue

            # Проверка кулдауна (включая дедупликацию конкретного сетапа)
            if is_on_cooldown(symbol, sfp.direction, state, sfp=sfp, mss=mss):
                stats["cooldown"] += 1
                continue

            # Фильтрация и генерация сигнала
            signal = generate_signal(
                symbol=symbol,
                timeframe=ENTRY_TIMEFRAME,
                df=df_ltf,
                sfp=sfp,
                mss=mss,
                structure=structure_ltf,
                htf_state=htf_state,
                balance=account_balance,
                risk_percent=risk_percent,
                leverage=leverage,
                htf_dist_atr=htf_dist,
                regime_uptrend=regime_uptrend,
            )

            if signal is None:
                stats["filter_rejected"] += 1

                # Получаем конкретные причины, почему кандидат был отклонён.
                try:
                    failure_reasons = get_signal_failure_reasons(
                        df=df_ltf,
                        sfp=sfp,
                        mss=mss,
                        structure=structure_ltf,
                        htf_state=htf_state,
                        allow_counter_trend=ENABLE_COUNTER_TREND_SIGNALS,
                    )
                except Exception as exc:
                    failure_reasons = [f"DIAGNOSTIC_ERROR: {exc}"]

                if not failure_reasons:
                    failure_reasons = ["Unknown filter rejection"]

                for reason in failure_reasons:
                    stats["filter_reasons"][reason] = (
                        stats["filter_reasons"].get(reason, 0) + 1
                    )

                # ========================================================
                # DETAILED CANDIDATE DIAGNOSTICS
                # ========================================================
                try:
                    debug_result = evaluate_setup(
                        df=df_ltf,
                        sfp=sfp,
                        mss=mss,
                        structure=structure_ltf,
                        htf_state=htf_state,
                        entry_price=float(df_ltf["close"].iloc[-1]),
                        allow_counter_trend=ENABLE_COUNTER_TREND_SIGNALS,
                    )

                    metrics = getattr(debug_result, "metrics", {}) or {}

                    entry = metrics.get("entry")
                    stop = metrics.get("stop_loss")
                    liquidity_target = metrics.get("liquidity_target")
                    liquidity_label = metrics.get("liquidity_target_label")

                    print(f"\\n[REJECTED CANDIDATE] {symbol} | {sfp.direction}")
                    print(
                        f"  Entry:             "
                        f"{entry if entry is not None else 'N/A'}"
                    )
                    print(
                        f"  Stop Loss:         "
                        f"{stop if stop is not None else 'N/A'}"
                    )
                    print(
                        f"  Liquidity Target:  "
                        f"{liquidity_target if liquidity_target is not None else 'N/A'}"
                        f" ({liquidity_label or 'N/A'})"
                    )

                    if entry is not None and stop is not None:
                        risk_distance = abs(float(entry) - float(stop))

                        print(
                            f"  Risk distance:     "
                            f"{risk_distance:.8g}"
                        )

                        if liquidity_target is not None and risk_distance > 0:
                            liquidity_rr = (
                                abs(float(liquidity_target) - float(entry))
                                / risk_distance
                            )
                            print(
                                f"  Liquidity RR:      "
                                f"{liquidity_rr:.2f}R"
                            )
                            required_liquidity_rr = max(1.60, 2.00)
                            print(
                                f"  Minimum Liquidity RR: "
                                f"{required_liquidity_rr:.2f}R"
                            )
                            print(
                                f"  RR Margin:         "
                                f"{liquidity_rr - required_liquidity_rr:+.2f}R"
                            )

                            tp1 = (
                                float(entry) + 2.0 * risk_distance
                                if sfp.direction.upper() == "LONG"
                                else float(entry) - 2.0 * risk_distance
                            )
                            tp2 = (
                                float(entry) + 3.0 * risk_distance
                                if sfp.direction.upper() == "LONG"
                                else float(entry) - 3.0 * risk_distance
                            )

                            print(f"  TP1 (2R):          {tp1:.8g}")
                            print(f"  TP2 (3R):          {tp2:.8g}")
                            print(f"  TP1 RR:            2.00R")
                            print(f"  TP2 RR:            3.00R")

                    print(
                        f"  SFP:               "
                        f"{getattr(sfp, 'direction', 'N/A')}"
                    )
                    print(
                        f"  MSS:               "
                        f"{getattr(mss, 'direction', 'N/A')}"
                    )
                    print(
                        f"  Score:             "
                        f"{getattr(debug_result, 'score', 'N/A')}"
                    )
                    print(
                        f"  Passed:            "
                        f"{getattr(debug_result, 'passed', 'N/A')}"
                    )
                    print(
                        f"  Reasons:           "
                        f"{' | '.join(failure_reasons)}"
                    )

                except Exception as debug_exc:
                    print(
                        f"[DEBUG ERROR] {symbol}: "
                        f"{type(debug_exc).__name__}: {debug_exc}"
                    )

                continue

            # Вывод найденного сигнала
            setup_type = getattr(signal, "setup_type", "TREND")
            if setup_type == "TREND":
                stats["signals_trend"] += 1
                badge = "🎯 [TREND]"
            else:
                stats["signals_counter_trend"] += 1
                badge = "⚡️ [PULLBACK]"

            print(f"\n{'-'*65}")
            print(
                f"{badge} {signal.symbol} {signal.direction} | "
                f"Score: {signal.signal_score:.1f} | "
                f"Entry: {signal.entry} | "
                f"SL: {signal.stop_loss} | "
                f"TP1: {signal.tp1} ({signal.rr_tp1:.2f}R) | "
                f"TP2: {signal.tp2} ({signal.rr_tp2:.2f}R) | "
                f"Margin: ${signal.margin_required:.2f}"
            )
            print(f"{'-'*65}\n")

            if telegram.is_configured:
                sent = telegram.send_signal(signal)
                if sent:
                    print(f"[{signal.symbol}] -> Successfully sent to Telegram.")
                    record_signal_sent(
                        symbol,
                        signal.direction,
                        state,
                        sfp=signal.sfp,
                        mss=signal.mss,
                    )
                    append_forward_journal(signal)
                    stats["signals_sent"] += 1
                else:
                    print(f"[{signal.symbol}] -> Failed to send Telegram message.")
            else:
                record_signal_sent(
                    symbol,
                    signal.direction,
                    state,
                    sfp=signal.sfp,
                    mss=signal.mss,
                )
                append_forward_journal(signal)

    save_signal_state(prune_signal_state(state))
    commit_forward_journal()

    print("\n" + "=" * 65)
    print("SCAN DIAGNOSTICS")
    print("=" * 65)

    total_signals = stats["signals_trend"] + stats["signals_counter_trend"]

    print(f"Total symbols scanned:      {stats['total_scanned']}")
    print(f"Rejected: HTF:              {stats['skipped_htf']}")
    print(f"Rejected: No SFP:            {stats['no_sfp']}")
    print(f"Rejected: No MSS:            {stats['no_mss']}")
    print(f"Rejected: Invalidated:       {stats['invalidated']}")
    print(f"Rejected: Cooldown:          {stats['cooldown']}")
    print(f"Rejected: Filters:           {stats['filter_rejected']}")
    print(f"Trend signals:               {stats['signals_trend']}")
    print(f"Counter-trend signals:       {stats['signals_counter_trend']}")
    print(f"Signals found:               {total_signals}")
    print(f"Signals sent:                {stats['signals_sent']}")

    if stats["filter_reasons"]:
        print("\nFILTER REJECTION REASONS:")
        for reason, count in sorted(
            stats["filter_reasons"].items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            print(f"  {count:4d}x  {reason}")

    print("=" * 65)


if __name__ == "__main__":
    try:
        run_scanner()
    except KeyboardInterrupt:
        print("\nScanner stopped by user.")
        sys.exit(0)
    except Exception as exc:
        print(f"\n[FATAL] Scanner crashed: {exc}")
        sys.exit(1)
