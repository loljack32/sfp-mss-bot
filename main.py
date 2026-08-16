# ============================================================
# MAIN SCANNER PIPELINE (FULL FUTURES SUPPORT WITH LEVERAGE)
# SFP + MSS BOT
# ============================================================

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

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
    ENABLE_COUNTER_TREND_SIGNALS,
)

from core.okx import OKXClient, OKXError
from core.structure import analyze_structure
from core.sfp import find_sfps, is_sfp_invalidated
from core.mss import find_latest_mss, is_mss_invalidated
from core.signals import generate_signal, get_signal_failure_reasons
from core.telegram import TelegramNotifier


# ============================================================
# SETTINGS & STATE MANAGEMENT
# ============================================================

USER_SETTINGS_FILE = "data/user_settings.json"


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
    except Exception as exc:
        print(f"[WARN] Failed to read user settings: {exc}")
        return 10000.0, float(DEFAULT_RISK_PERCENT), int(DEFAULT_LEVERAGE)


def load_signal_state(filepath: str = SIGNAL_STATE_FILE) -> Dict[str, str]:
    path = Path(filepath)
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"[WARN] Failed to read state file: {exc}")
        return {}


def save_signal_state(state: Dict[str, str], filepath: str = SIGNAL_STATE_FILE) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        print(f"[ERROR] Failed to save state file: {exc}")


def is_on_cooldown(
    symbol: str,
    direction: str,
    state: Dict[str, str],
    cooldown_minutes: int = SIGNAL_COOLDOWN_MINUTES,
) -> bool:
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


def record_signal_sent(symbol: str, direction: str, state: Dict[str, str]) -> None:
    key = f"{symbol}_{direction}"
    state[key] = datetime.now(timezone.utc).isoformat()


# ============================================================
# MAIN SCANNER LOOP
# ============================================================

def run_scanner() -> None:
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    state = load_signal_state()
    account_balance, risk_percent, leverage = load_user_settings()
    telegram = TelegramNotifier()

    print("=" * 70)
    print(f"Starting SFP + MSS Scanner at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"Balance: ${account_balance:,.2f} | Risk: {risk_percent:.2f}% | Leverage: {leverage}x")
    print(f"HTF: {HTF_TIMEFRAME} | LTF: {ENTRY_TIMEFRAME} | Counter-Trend: {ENABLE_COUNTER_TREND_SIGNALS}")
    print("=" * 70)

    stats = {
        "total_scanned": 0,
        "skipped_htf": 0,
        "no_sfp": 0,
        "no_mss": 0,
        "invalidated": 0,
        "cooldown": 0,
        "filter_rejected": 0,
        "signals_trend": 0,
        "signals_counter_trend": 0,
        "signals_sent": 0,
    }

    with OKXClient() as client:
        if DYNAMIC_TOP_SYMBOLS:
            print(f"\n[INFO] Fetching Top {TOP_SYMBOLS_COUNT} spot symbols by 24h volume from OKX...")
            try:
                symbols_to_scan = client.get_top_volume_symbols(top_n=TOP_SYMBOLS_COUNT)
                print(f"[INFO] Loaded {len(symbols_to_scan)} liquid symbols from OKX.\n")
            except Exception as exc:
                print(f"[WARN] Failed to fetch top symbols, using fallback list: {exc}\n")
                symbols_to_scan = SYMBOLS
        else:
            symbols_to_scan = SYMBOLS

        total = len(symbols_to_scan)
        stats["total_scanned"] = total

        for idx, symbol in enumerate(symbols_to_scan, 1):
            tag = f"[{idx:03d}/{total:03d}] {symbol}"

            try:
                time.sleep(0.05)

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
            except OKXError as exc:
                print(f"{tag} -> ❌ OKX Error: {exc}")
                continue
            except Exception as exc:
                print(f"{tag} -> ❌ Data fetch error: {exc}")
                continue

            # Анализ структуры
            _, htf_state = analyze_structure(df_htf, htf=True)
            structure_ltf, _ = analyze_structure(df_ltf, htf=False)

            if htf_state.trend not in {"BULLISH", "BEARISH"}:
                stats["skipped_htf"] += 1
                print(f"{tag} -> ⏭ Skipped: HTF trend is {htf_state.trend} (no clear directional bias)")
                continue

            # SFP
            sfps = find_sfps(df_ltf, structure_ltf)
            if not sfps:
                stats["no_sfp"] += 1
                print(f"{tag} -> ⏭ HTF: {htf_state.trend} | No valid SFP found on LTF")
                continue

            # MSS
            result = find_latest_mss(df_ltf, sfps, structure_ltf)
            if result is None:
                stats["no_mss"] += 1
                latest_sfp = sfps[-1]
                print(f"{tag} -> ⏭ HTF: {htf_state.trend} | SFP {latest_sfp.direction} found, but no confirmed MSS shift yet")
                continue

            sfp, mss = result

            # Инвалидация
            if is_sfp_invalidated(df_ltf, sfp):
                stats["invalidated"] += 1
                print(f"{tag} -> ⏭ Setup invalidated: SFP extreme breached")
                continue

            if is_mss_invalidated(df_ltf, sfp, mss):
                stats["invalidated"] += 1
                print(f"{tag} -> ⏭ Setup invalidated: MSS level breached")
                continue

            # Кулдаун
            if is_on_cooldown(symbol, sfp.direction, state):
                stats["cooldown"] += 1
                print(f"{tag} -> ⏭ Signal {sfp.direction} is on cooldown")
                continue

            # Фильтрация и генерация сигнала с учётом плеча
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
            )

            if signal is None:
                stats["filter_rejected"] += 1
                reasons = get_signal_failure_reasons(
                    df=df_ltf,
                    sfp=sfp,
                    mss=mss,
                    structure=structure_ltf,
                    htf_state=htf_state,
                )
                reasons_str = "; ".join(reasons) if reasons else "Filters not met"
                print(f"{tag} -> ⚠️ SFP+MSS ({sfp.direction}), REJECTED by filters: [{reasons_str}]")
                continue

            # Сигнал сформирован
            setup_type = getattr(signal, "setup_type", "TREND")
            if setup_type == "TREND":
                stats["signals_trend"] += 1
                icon = "🎯 [TREND]"
            else:
                stats["signals_counter_trend"] += 1
                icon = "⚡️ [PULLBACK/COUNTER-TREND]"

            print(f"\n{'='*70}")
            print(f"{icon} {tag} -> VALID SIGNAL: {signal.direction} | Score: {signal.signal_score:.1f} | Entry: {signal.entry} | Margin: ${signal.margin_required:.2f}")
            print(f"{'='*70}\n")

            if telegram.is_configured:
                sent = telegram.send_signal(signal)
                if sent:
                    print(f"{tag} -> 📤 Sent to Telegram successfully.")
                    record_signal_sent(symbol, signal.direction, state)
                    stats["signals_sent"] += 1
                else:
                    print(f"{tag} -> ❌ Failed to send Telegram message.")
            else:
                print(f"{tag} -> ℹ️ Telegram not configured, recorded locally.")
                record_signal_sent(symbol, signal.direction, state)

    save_signal_state(state)

    print("\n" + "=" * 70)
    print("📊 SCAN SUMMARY REPORT:")
    print(f"• Total coins scanned:           {stats['total_scanned']}")
    print(f"• Skipped (HTF trend mixed):      {stats['skipped_htf']}")
    print(f"• Skipped (No SFP pattern):      {stats['no_sfp']}")
    print(f"• Skipped (SFP found, no MSS):   {stats['no_mss']}")
    print(f"• Skipped (Invalidated):         {stats['invalidated']}")
    print(f"• Skipped (Cooldown active):     {stats['cooldown']}")
    print(f"• Rejected by Quality Filters:   {stats['filter_rejected']}")
    print(f"• Trend Signals (4H aligned):    {stats['signals_trend']}")
    print(f"• Pullback Signals (Counter-4H): {stats['signals_counter_trend']}")
    print(f"• Signals Sent to Telegram:      {stats['signals_sent']}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        run_scanner()
    except KeyboardInterrupt:
        print("\nScanner stopped by user.")
        sys.exit(0)
    except Exception as exc:
        print(f"\n[FATAL] Scanner crashed with exception: {exc}")
        sys.exit(1)
