# ============================================================
# MAIN SCANNER PIPELINE
# SFP + MSS BOT
# ============================================================

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from config import (
    SYMBOLS,
    HTF_TIMEFRAME,
    ENTRY_TIMEFRAME,
    CANDLE_LIMIT,
    DATA_DIR,
    SIGNAL_STATE_FILE,
    DEFAULT_RISK_PERCENT,
    SIGNAL_COOLDOWN_MINUTES,
)

from core.okx import OKXClient, OKXError
from core.structure import analyze_structure
from core.sfp import find_sfps, is_sfp_invalidated
from core.mss import find_latest_mss, is_mss_invalidated
from core.signals import generate_signal, get_signal_failure_reasons
from core.telegram import TelegramNotifier


# ============================================================
# STATE / COOLDOWN MANAGEMENT
# ============================================================

def load_signal_state(filepath: str = SIGNAL_STATE_FILE) -> Dict[str, str]:
    """
    Загружает историю отправленных сигналов для предотвращения дублирования.
    Формат: { "SYMBOL_DIRECTION": "ISO_TIMESTAMP" }
    """
    path = Path(filepath)
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"[WARN] Failed to read state file {filepath}: {exc}")
        return {}


def save_signal_state(state: Dict[str, str], filepath: str = SIGNAL_STATE_FILE) -> None:
    """
    Сохраняет состояние сигналов на диск.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        print(f"[ERROR] Failed to save state file {filepath}: {exc}")


def is_on_cooldown(
    symbol: str,
    direction: str,
    state: Dict[str, str],
    cooldown_minutes: int = SIGNAL_COOLDOWN_MINUTES,
) -> bool:
    """
    Проверяет, прошло ли достаточно времени с момента прошлого сигнала.
    """
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
    """
    Записывает время отправки сигнала.
    """
    key = f"{symbol}_{direction}"
    state[key] = datetime.now(timezone.utc).isoformat()


# ============================================================
# MAIN EXECUTION
# ============================================================

def run_scanner() -> None:
    """
    Основной рабочий цикл сканирования инструментов по OKX API.
    """
    print("=" * 60)
    print(f"Starting SFP + MSS Scanner at {datetime.now(timezone.utc).isoformat()} UTC")
    print(f"Symbols ({len(SYMBOLS)}): {', '.join(SYMBOLS)}")
    print(f"HTF: {HTF_TIMEFRAME} | LTF Entry: {ENTRY_TIMEFRAME}")
    print("=" * 60)

    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    state = load_signal_state()

    telegram = TelegramNotifier()
    signals_found = 0
    signals_sent = 0

    account_balance = 10000.0

    with OKXClient() as client:
        for symbol in SYMBOLS:
            print(f"\n[{symbol}] Fetching market data...")

            try:
                # 1. Получаем HTF свечи (4H) для общего тренда
                df_htf = client.get_candles(
                    symbol=symbol,
                    timeframe=HTF_TIMEFRAME,
                    limit=CANDLE_LIMIT,
                    closed_only=True,
                )

                # 2. Получаем LTF свечи (15m) для поиска сетапов
                df_ltf = client.get_candles(
                    symbol=symbol,
                    timeframe=ENTRY_TIMEFRAME,
                    limit=CANDLE_LIMIT,
                    closed_only=True,
                )
            except OKXError as exc:
                print(f"[{symbol}] OKX API Error: {exc}")
                continue
            except Exception as exc:
                print(f"[{symbol}] Unexpected error: {exc}")
                continue

            # 3. Анализируем структуру на HTF и LTF
            _, htf_state = analyze_structure(df_htf, htf=True)
            structure_ltf, _ = analyze_structure(df_ltf, htf=False)

            print(f"[{symbol}] HTF Trend: {htf_state.trend} | LTF Swings: {len(structure_ltf)}")

            # 4. Ищем сформированные SFP
            sfps = find_sfps(df_ltf, structure_ltf)
            if not sfps:
                print(f"[{symbol}] No valid SFP found.")
                continue

            # 5. Ищем подтверждённый MSS после SFP
            result = find_latest_mss(df_ltf, sfps, structure_ltf)
            if result is None:
                print(f"[{symbol}] SFP found, but no confirmed MSS yet.")
                continue

            sfp, mss = result

            # 6. Проверяем инвалидацию
            if is_sfp_invalidated(df_ltf, sfp):
                print(f"[{symbol}] Setup invalidated: SFP extreme violated.")
                continue

            if is_mss_invalidated(df_ltf, sfp, mss):
                print(f"[{symbol}] Setup invalidated: MSS violated.")
                continue

            # 7. Проверяем cooldown
            if is_on_cooldown(symbol, sfp.direction, state):
                print(f"[{symbol}] Signal {sfp.direction} is on cooldown. Skipping.")
                continue

            # 8. Генерируем торговый сигнал со всеми фильтрами и риск-менеджментом
            signal = generate_signal(
                symbol=symbol,
                timeframe=ENTRY_TIMEFRAME,
                df=df_ltf,
                sfp=sfp,
                mss=mss,
                structure=structure_ltf,
                htf_state=htf_state,
                balance=account_balance,
                risk_percent=DEFAULT_RISK_PERCENT,
            )

            if signal is None:
                reasons = get_signal_failure_reasons(
                    df=df_ltf,
                    sfp=sfp,
                    mss=mss,
                    structure=structure_ltf,
                    htf_state=htf_state,
                )
                print(f"[{symbol}] Setup did not pass filters: {'; '.join(reasons)}")
                continue

            # 9. Успешный сигнал
            signals_found += 1
            print(f"[{symbol}] 🎯 VALID SIGNAL GENERATED! Direction: {signal.direction} | Score: {signal.signal_score:.1f}")

            # Отправка в Telegram
            if telegram.is_configured:
                sent = telegram.send_signal(signal)
                if sent:
                    print(f"[{symbol}] Signal successfully sent to Telegram.")
                    record_signal_sent(symbol, signal.direction, state)
                    signals_sent += 1
                else:
                    print(f"[{symbol}] Failed to send signal to Telegram.")
            else:
                print(f"[{symbol}] Telegram not configured, logging locally:")
                print(f"Entry: {signal.entry} | SL: {signal.stop_loss} | TP1: {signal.tp1} | TP2: {signal.tp2}")
                record_signal_sent(symbol, signal.direction, state)

    # Сохраняем состояние cooldown
    save_signal_state(state)

    print("\n" + "=" * 60)
    print(f"Scan finished. Signals Found: {signals_found} | Signals Sent: {signals_sent}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        run_scanner()
    except KeyboardInterrupt:
        print("\nScanner stopped by user.")
        sys.exit(0)
    except Exception as exc:
        print(f"\n[FATAL] Scanner crashed with exception: {exc}")
        sys.exit(1)
