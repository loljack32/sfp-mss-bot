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
    ENABLE_BREAKEVEN,
    BREAKEVEN_TRIGGER_R,
    HTF_EMA_PERIOD,
    ATR_PERIOD,
    REGIME_SYMBOL,
    REGIME_EMA_PERIOD,
)

from core.okx import OKXClient, normalize_instrument_id
from core.structure import build_structure, get_structure_state
from core.sfp import find_sfps, calculate_atr
from core.mss import find_mss_after_sfp
from core.signals import generate_signal


COST_R = 0.05  # примерные комиссии + проскальзывание в R на сделку


# ============================================================
# HELPERS
# ============================================================

def _bar_duration(df: pd.DataFrame) -> pd.Timedelta:
    diffs = df["timestamp"].diff().dropna()
    return diffs.median()


_REGIME_CACHE: Dict[tuple, tuple] = {}


def _regime_series(
    client, mode, symbol, total_ltf_bars, ema_period, slope_bars: int = 0
):
    """
    Ряд режима рынка: (close_times_ns, uptrend по барам режима).

    mode: btc4h - BTC на 4H, btc1d - BTC на дневке, self1d -
    дневка самого символа. uptrend = close выше EMA(ema_period),
    до прогрева EMA - False. Ряд кэшируется: бенчмарк один на
    все символы прогона.
    """
    if mode == "btc4h":
        r_symbol, r_tf, per_bar = REGIME_SYMBOL, "4H", 16
    elif mode == "btc1d":
        r_symbol, r_tf, per_bar = REGIME_SYMBOL, "1D", 96
    elif mode == "self1d":
        r_symbol, r_tf, per_bar = symbol, "1D", 96
    else:
        return None

    key = (r_symbol, r_tf, ema_period, total_ltf_bars, slope_bars)
    if key in _REGIME_CACHE:
        return _REGIME_CACHE[key]

    bars_needed = total_ltf_bars // per_bar + ema_period + 100
    df_r = client.get_candles_history(
        normalize_instrument_id(r_symbol), r_tf, bars_needed
    )
    duration = _bar_duration(df_r)
    close_times_ns = (
        (df_r["timestamp"] + duration).astype("int64").to_numpy()
    )
    closes = df_r["close"].to_numpy(dtype=float)
    ema = (
        df_r["close"]
        .ewm(span=ema_period, adjust=False)
        .mean()
        .to_numpy(dtype=float)
    )
    uptrend = closes > ema
    if slope_bars > 0:
        # Подтверждённый рост: EMA не просто под ценой, а растёт
        # относительно slope_bars баров назад. Отсекает отскоки
        # внутри нисходящего тренда (бычьи ловушки).
        ema_prev = np.roll(ema, slope_bars)
        uptrend = uptrend & (ema > ema_prev)
    if ema_period > 1:
        uptrend[: ema_period - 1 + max(slope_bars, 0)] = False

    result = (close_times_ns, uptrend)
    _REGIME_CACHE[key] = result
    return result


def _confirmed_swings(structure, upto_index: int, right: int):
    """Свинги, которые к бару upto_index уже подтверждены."""
    return [s for s in structure if s.index + right <= upto_index]


def _htf_state_asof(
    htf_df: pd.DataFrame,
    htf_structure,
    ltf_close_time: pd.Timestamp,
    htf_duration: pd.Timedelta,
    close_times_ns: Optional[np.ndarray] = None,
    cache: Optional[dict] = None,
):
    """Состояние HTF-структуры только по закрытым HTF-свечам."""
    if close_times_ns is None:
        close_times_ns = (
            (htf_df["timestamp"] + htf_duration).astype("int64").to_numpy()
        )

    # Сравнение в int64 наносекундах: без проблем с tz-aware/naive.
    closed = int(
        np.searchsorted(close_times_ns, int(ltf_close_time.value), side="right")
    )
    last_closed_index = closed - 1

    if last_closed_index < 0:
        return None

    if cache is not None and last_closed_index in cache:
        return cache[last_closed_index]

    confirmed = _confirmed_swings(
        htf_structure, last_closed_index, HTF_SWING_RIGHT
    )
    state = get_structure_state(confirmed) if confirmed else None

    if cache is not None:
        cache[last_closed_index] = state

    return state


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
    Безубыток: после хода в плюс на BREAKEVEN_TRIGGER_R стоп
    переносится в точку входа (действует со следующего бара).
    """
    entry = signal.entry
    stop = signal.stop_loss
    tp1 = signal.tp1
    tp2 = signal.tp2
    risk_distance = abs(entry - stop)
    is_long = signal.direction == "LONG"

    end = min(len(df) - 1, start_index + horizon)
    tp1_done = False
    be_armed = False
    active_stop = stop

    def mtm_r(price: float) -> float:
        if is_long:
            return (price - entry) / risk_distance
        return (entry - price) / risk_distance

    for j in range(start_index + 1, end + 1):
        high = float(df.iloc[j]["high"])
        low = float(df.iloc[j]["low"])

        if is_long:
            hit_stop = low <= active_stop
            hit_tp1 = high >= tp1
            hit_tp2 = high >= tp2
            fav_r = (high - entry) / risk_distance
        else:
            hit_stop = high >= active_stop
            hit_tp1 = low <= tp1
            hit_tp2 = low <= tp2
            fav_r = (entry - low) / risk_distance

        if not tp1_done:
            if hit_stop:
                if be_armed:
                    return {"result_r": 0.0 - COST_R, "exit": "BE", "bars": j - start_index}
                return {"result_r": -1.0 - COST_R, "exit": "SL", "bars": j - start_index}
            if hit_tp1:
                tp1_done = True
                if hit_tp2:
                    blended = 0.5 * signal.rr_tp1 + 0.5 * signal.rr_tp2
                    return {"result_r": blended - COST_R, "exit": "TP2", "bars": j - start_index}
        else:
            if hit_stop:
                if be_armed:
                    blended = 0.5 * signal.rr_tp1 + 0.0
                    return {"result_r": blended - COST_R, "exit": "TP1+BE", "bars": j - start_index}
                blended = 0.5 * signal.rr_tp1 - 0.5
                return {"result_r": blended - COST_R, "exit": "TP1+SL", "bars": j - start_index}
            if hit_tp2:
                blended = 0.5 * signal.rr_tp1 + 0.5 * signal.rr_tp2
                return {"result_r": blended - COST_R, "exit": "TP2", "bars": j - start_index}

        if ENABLE_BREAKEVEN and not be_armed and fav_r >= BREAKEVEN_TRIGGER_R:
            be_armed = True
            active_stop = entry

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
    offset_bars: int = 0,
    regime_mode: str = "btc4h",
    regime_ema: int = 0,
    regime_slope: int = 0,
) -> List[dict]:
    symbol = normalize_instrument_id(symbol)

    df = client.get_candles_history(symbol, ENTRY_TIMEFRAME, ltf_bars + offset_bars)
    htf_bars_needed = ltf_bars // 16 + 300
    df_htf = client.get_candles_history(
        symbol, HTF_TIMEFRAME, htf_bars_needed + offset_bars // 16 + 50
    )
    if offset_bars > 0:
        # Контрольное окно: отрезаем последние offset_bars свечей, чтобы
        # прогнать более старый период, которого настройка не видела.
        df = df.iloc[: len(df) - offset_bars]
        df_htf = df_htf[df_htf.index <= df.index[-1]]

    regime = None
    if regime_mode != "off":
        regime = _regime_series(
            client,
            regime_mode,
            symbol,
            ltf_bars + offset_bars,
            regime_ema if regime_ema > 0 else REGIME_EMA_PERIOD,
            slope_bars=regime_slope,
        )

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

    # Индекс пар по бару пробоя: на баре i свежи только пары с
    # break_index в окне [i - MAX_SIGNAL_AGE_BARS, i].
    pairs_by_break: Dict[int, list] = {}
    for pair in pairs:
        pairs_by_break.setdefault(pair[1].break_index, []).append(pair)

    htf_close_times_ns = (
        (df_htf["timestamp"] + htf_duration).astype("int64").to_numpy()
    )
    htf_state_cache: dict = {}

    # EMA/ATR старшего ТФ один раз на весь ряд: дистанция цены от
    # EMA50 в ATR считается по последнему закрытому бару HTF.
    htf_closes_arr = df_htf["close"].to_numpy(dtype=float)
    htf_ema_arr = (
        df_htf["close"]
        .ewm(span=HTF_EMA_PERIOD, adjust=False)
        .mean()
        .to_numpy(dtype=float)
    )
    htf_atr_arr = calculate_atr(df_htf, period=ATR_PERIOD).to_numpy(dtype=float)

    closes = df["close"].to_numpy(dtype=float)

    trades: List[dict] = []
    used_fingerprints = set()
    last_trade_bar = -10**9

    for i in range(warmup, len(df) - 1):
        if i - last_trade_bar < cooldown_bars:
            continue

        # Кандидаты: MSS уже случился и ещё свеж к бару i.
        candidates = []
        for b in range(max(0, i - MAX_SIGNAL_AGE_BARS), i + 1):
            for sfp, mss in pairs_by_break.get(b, []):
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
            df_htf,
            htf_structure_full,
            ltf_close_time,
            htf_duration,
            close_times_ns=htf_close_times_ns,
            cache=htf_state_cache,
        )
        if htf_state is None or htf_state.trend not in {"BULLISH", "BEARISH"}:
            continue

        htf_dist = None
        closed_cnt = int(
            np.searchsorted(
                htf_close_times_ns, int(ltf_close_time.value), side="right"
            )
        )
        hi = closed_cnt - 1
        if hi >= HTF_EMA_PERIOD and htf_atr_arr[hi] > 0:
            htf_dist = float(
                (htf_closes_arr[hi] - htf_ema_arr[hi]) / htf_atr_arr[hi]
            )

        regime_uptrend = None
        if regime is not None:
            r_close_ns, r_uptrend = regime
            ri = (
                int(
                    np.searchsorted(
                        r_close_ns, int(ltf_close_time.value), side="right"
                    )
                )
                - 1
            )
            if ri >= 0:
                regime_uptrend = bool(r_uptrend[ri])

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
            htf_dist_atr=htf_dist,
            regime_uptrend=regime_uptrend,
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

    # Разбивка по направлениям и типам выхода - для настройки.
    from collections import Counter

    for direction in ("LONG", "SHORT"):
        subset = [t for t in trades if t["direction"] == direction]
        if subset:
            r = np.array([t["result_r"] for t in subset], dtype=float)
            print(
                f"  {direction}: {len(subset)} сделок, "
                f"winrate {100.0 * (r > 0).mean():.0f}%, итого {r.sum():+.2f}R"
            )

    exits = Counter(t["exit"] for t in trades)
    print("  Выходы: " + ", ".join(f"{k}={v}" for k, v in exits.most_common()))


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
    parser.add_argument("--offset-bars", type=int, default=0)
    parser.add_argument(
        "--regime-mode",
        choices=["off", "btc4h", "btc1d", "self1d"],
        default="btc1d",
    )
    parser.add_argument("--regime-ema", type=int, default=0)
    parser.add_argument("--enable-shorts", action="store_true")
    parser.add_argument("--regime-slope", type=int, default=0)
    args = parser.parse_args()

    if args.enable_shorts:
        # Экспериментальный прогон: разрешить шорты в обход конфига
        # (гейт направления живёт в core.filters).
        import core.filters as _filters

        _filters.ENABLE_SHORT_SIGNALS = True

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
                    offset_bars=args.offset_bars,
                    regime_mode=args.regime_mode,
                    regime_ema=args.regime_ema,
                    regime_slope=args.regime_slope,
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
