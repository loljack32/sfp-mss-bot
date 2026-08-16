# ============================================================
# SFP — SWING FAILURE PATTERN
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from config import (
    ATR_PERIOD,
    MIN_SFP_SWEEP_ATR,
    MAX_SFP_SWEEP_ATR,
    MIN_SFP_RECLAIM_RATIO,
)

from core.structure import (
    SwingPoint,
)


# ============================================================
# DATA CLASS
# ============================================================

@dataclass(frozen=True)
class SFPSetup:
    """
    Подтверждённый Swing Failure Pattern.

    direction:
        LONG  = bullish SFP
        SHORT = bearish SFP

    level:
        уровень swing/liquidity

    sweep_extreme:
        экстремум свечи, которая забрала ликвидность

    sweep_index:
        индекс свечи SFP

    sweep_timestamp:
        время свечи SFP

    entry_reference:
        цена закрытия SFP свечи.

    atr:
        ATR на момент SFP.

    sweep_distance:
        абсолютная глубина прокола.

    sweep_atr_ratio:
        глубина прокола / ATR.

    reclaim_ratio:
        насколько хорошо свеча вернулась обратно.

    score:
        базовая оценка качества SFP.
    """

    direction: str

    level: float
    sweep_extreme: float

    sweep_index: int
    sweep_timestamp: pd.Timestamp

    entry_reference: float

    atr: float

    sweep_distance: float
    sweep_atr_ratio: float

    reclaim_ratio: float

    candle_body_ratio: float

    score: float


# ============================================================
# ATR
# ============================================================

def calculate_atr(
    df: pd.DataFrame,
    period: int = ATR_PERIOD,
) -> pd.Series:
    """
    Рассчитывает классический Wilder-style ATR.

    TR:
        max(
            high-low,
            abs(high-prev_close),
            abs(low-prev_close)
        )
    """

    if period < 1:
        raise ValueError(
            "ATR period must be >= 1"
        )

    high = df["high"]
    low = df["low"]
    close = df["close"]

    previous_close = close.shift(1)

    tr1 = high - low

    tr2 = (
        high - previous_close
    ).abs()

    tr3 = (
        low - previous_close
    ).abs()

    true_range = pd.concat(
        [
            tr1,
            tr2,
            tr3,
        ],
        axis=1,
    ).max(axis=1)

    # Wilder RMA через ewm(alpha=1/period)
    atr = true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    return atr


# ============================================================
# CANDLE METRICS
# ============================================================

def candle_range(
    row: pd.Series,
) -> float:

    return float(
        row["high"] - row["low"]
    )


def candle_body(
    row: pd.Series,
) -> float:

    return abs(
        float(row["close"] - row["open"])
    )


def candle_body_ratio(
    row: pd.Series,
) -> float:

    full_range = candle_range(row)

    if full_range <= 0:
        return 0.0

    return (
        candle_body(row)
        / full_range
    )


# ============================================================
# BULLISH SFP
# ============================================================

def detect_bullish_sfp(
    df: pd.DataFrame,
    swing: SwingPoint,
    sweep_index: int,
    atr: float,
) -> Optional[SFPSetup]:
    """
    Проверяет bullish SFP.

    Условия:

    1. Swing должен быть LOW.
    2. Low текущей свечи должен уйти ниже swing level.
    3. Sweep должен быть достаточно глубоким.
    4. Sweep не должен быть чрезмерным.
    5. Close должен вернуться выше swing level.
    6. Свеча должна иметь осмысленный диапазон.
    """

    if swing.kind != "LOW":
        return None

    if sweep_index <= swing.index:
        return None

    if sweep_index >= len(df):
        return None

    if atr <= 0:
        return None

    row = df.iloc[sweep_index]

    level = float(swing.price)

    low = float(row["low"])
    high = float(row["high"])
    open_price = float(row["open"])
    close = float(row["close"])

    full_range = high - low

    if full_range <= 0:
        return None

    # --------------------------------------------------------
    # SWEEP
    # --------------------------------------------------------

    if low >= level:
        return None

    sweep_distance = (
        level - low
    )

    sweep_atr_ratio = (
        sweep_distance / atr
    )

    if (
        sweep_atr_ratio
        < MIN_SFP_SWEEP_ATR
    ):
        return None

    if (
        sweep_atr_ratio
        > MAX_SFP_SWEEP_ATR
    ):
        return None

    # --------------------------------------------------------
    # RECLAIM
    # --------------------------------------------------------

    if close <= level:
        return None

    # Расстояние от low до close.
    # Чем ближе close к high, тем сильнее возврат.
    reclaim_ratio = (
        close - low
    ) / full_range

    if (
        reclaim_ratio
        < MIN_SFP_RECLAIM_RATIO
    ):
        return None

    # --------------------------------------------------------
    # BODY
    # --------------------------------------------------------

    body_ratio = (
        abs(close - open_price)
        / full_range
    )

    # Для bullish SFP желательно bullish close.
    bullish_close = (
        close > open_price
    )

    # Не запрещаем полностью doji,
    # но снижаем score.
    # --------------------------------------------------------

    score = calculate_sfp_score(
        direction="LONG",
        sweep_atr_ratio=sweep_atr_ratio,
        reclaim_ratio=reclaim_ratio,
        body_ratio=body_ratio,
        directional_close=bullish_close,
    )

    return SFPSetup(
        direction="LONG",
        level=level,
        sweep_extreme=low,
        sweep_index=sweep_index,
        sweep_timestamp=row["timestamp"],
        entry_reference=close,
        atr=atr,
        sweep_distance=sweep_distance,
        sweep_atr_ratio=sweep_atr_ratio,
        reclaim_ratio=reclaim_ratio,
        candle_body_ratio=body_ratio,
        score=score,
    )


# ============================================================
# BEARISH SFP
# ============================================================

def detect_bearish_sfp(
    df: pd.DataFrame,
    swing: SwingPoint,
    sweep_index: int,
    atr: float,
) -> Optional[SFPSetup]:
    """
    Проверяет bearish SFP.

    Зеркальная логика bullish SFP.
    """

    if swing.kind != "HIGH":
        return None

    if sweep_index <= swing.index:
        return None

    if sweep_index >= len(df):
        return None

    if atr <= 0:
        return None

    row = df.iloc[sweep_index]

    level = float(swing.price)

    low = float(row["low"])
    high = float(row["high"])
    open_price = float(row["open"])
    close = float(row["close"])

    full_range = high - low

    if full_range <= 0:
        return None

    # --------------------------------------------------------
    # SWEEP
    # --------------------------------------------------------

    if high <= level:
        return None

    sweep_distance = (
        high - level
    )

    sweep_atr_ratio = (
        sweep_distance / atr
    )

    if (
        sweep_atr_ratio
        < MIN_SFP_SWEEP_ATR
    ):
        return None

    if (
        sweep_atr_ratio
        > MAX_SFP_SWEEP_ATR
    ):
        return None

    # --------------------------------------------------------
    # RECLAIM
    # --------------------------------------------------------

    if close >= level:
        return None

    # Для bearish SFP:
    #
    # high → sweep
    # close → возврат вниз
    #
    # Чем ближе close к low,
    # тем сильнее reclaim.

    reclaim_ratio = (
        high - close
    ) / full_range

    if (
        reclaim_ratio
        < MIN_SFP_RECLAIM_RATIO
    ):
        return None

    # --------------------------------------------------------
    # BODY
    # --------------------------------------------------------

    body_ratio = (
        abs(close - open_price)
        / full_range
    )

    bearish_close = (
        close < open_price
    )

    score = calculate_sfp_score(
        direction="SHORT",
        sweep_atr_ratio=sweep_atr_ratio,
        reclaim_ratio=reclaim_ratio,
        body_ratio=body_ratio,
        directional_close=bearish_close,
    )

    return SFPSetup(
        direction="SHORT",
        level=level,
        sweep_extreme=high,
        sweep_index=sweep_index,
        sweep_timestamp=row["timestamp"],
        entry_reference=close,
        atr=atr,
        sweep_distance=sweep_distance,
        sweep_atr_ratio=sweep_atr_ratio,
        reclaim_ratio=reclaim_ratio,
        candle_body_ratio=body_ratio,
        score=score,
    )


# ============================================================
# SFP SCORE
# ============================================================

def calculate_sfp_score(
    direction: str,
    sweep_atr_ratio: float,
    reclaim_ratio: float,
    body_ratio: float,
    directional_close: bool,
) -> float:
    """
    Базовый score SFP.

    Это НЕ финальный signal score.

    Здесь оценивается только качество самого SFP.
    """

    score = 0.0

    # --------------------------------------------------------
    # SWEEP QUALITY
    # --------------------------------------------------------

    # Оптимальная зона примерно 0.10–0.50 ATR.
    if (
        0.10
        <= sweep_atr_ratio
        <= 0.50
    ):
        score += 35

    elif (
        0.05
        <= sweep_atr_ratio
        <= 0.70
    ):
        score += 25

    else:
        score += 15

    # --------------------------------------------------------
    # RECLAIM
    # --------------------------------------------------------

    if reclaim_ratio >= 0.80:
        score += 35

    elif reclaim_ratio >= 0.65:
        score += 28

    elif reclaim_ratio >= 0.50:
        score += 20

    else:
        score += 5

    # --------------------------------------------------------
    # BODY
    # --------------------------------------------------------

    if body_ratio >= 0.70:
        score += 20

    elif body_ratio >= 0.50:
        score += 15

    elif body_ratio >= 0.30:
        score += 10

    else:
        score += 5

    # --------------------------------------------------------
    # DIRECTIONAL CLOSE
    # --------------------------------------------------------

    if directional_close:
        score += 10

    return min(
        100.0,
        score,
    )


# ============================================================
# FIND SFPs
# ============================================================

def find_sfps(
    df: pd.DataFrame,
    structure: List[SwingPoint],
    max_bars_after_swing: int = 100,
) -> List[SFPSetup]:
    """
    Ищет все потенциальные SFP в DataFrame.

    Важно:

    Мы анализируем только уже сформировавшиеся swing points.

    Это означает, что swing должен быть подтверждён
    достаточным количеством свечей справа.
    """

    if df.empty:
        return []

    atr_series = calculate_atr(
        df
    )

    results: List[SFPSetup] = []

    for swing in structure:

        start_index = (
            swing.index + 1
        )

        end_index = min(
            len(df),
            swing.index
            + max_bars_after_swing
            + 1,
        )

        for i in range(
            start_index,
            end_index,
        ):

            atr_value = atr_series.iloc[i]

            if pd.isna(atr_value):
                continue

            atr = float(
                atr_value
            )

            if atr <= 0:
                continue

            if swing.kind == "LOW":

                setup = detect_bullish_sfp(
                    df=df,
                    swing=swing,
                    sweep_index=i,
                    atr=atr,
                )

            else:

                setup = detect_bearish_sfp(
                    df=df,
                    swing=swing,
                    sweep_index=i,
                    atr=atr,
                )

            if setup is not None:

                results.append(
                    setup
                )

                # Один swing не должен создавать
                # бесконечное количество SFP.
                #
                # После первого валидного sweep
                # этот swing больше не используем.
                break

    results.sort(
        key=lambda x: x.sweep_index
    )

    return results


# ============================================================
# FIND LATEST SFP
# ============================================================

def find_latest_sfp(
    df: pd.DataFrame,
    structure: List[SwingPoint],
    max_bars_after_swing: int = 100,
) -> Optional[SFPSetup]:
    """
    Возвращает последний подтверждённый SFP.
    """

    sfps = find_sfps(
        df=df,
        structure=structure,
        max_bars_after_swing=max_bars_after_swing,
    )

    if not sfps:
        return None

    return sfps[-1]


# ============================================================
# SFP INVALIDATION
# ============================================================

def is_sfp_invalidated(
    df: pd.DataFrame,
    sfp: SFPSetup,
) -> bool:
    """
    Проверяет, был ли SFP полностью сломан.

    Bullish:
        последующее закрытие ниже sweep extreme.

    Bearish:
        последующее закрытие выше sweep extreme.
    """

    future = df.iloc[
        sfp.sweep_index + 1:
    ]

    if future.empty:
        return False

    if sfp.direction == "LONG":

        return bool(
            (
                future["close"]
                < sfp.sweep_extreme
            ).any()
        )

    return bool(
        (
            future["close"]
            > sfp.sweep_extreme
        ).any()
    )


# ============================================================
# SFP DESCRIPTION
# ============================================================

def sfp_to_dict(
    sfp: SFPSetup,
) -> dict:
    """
    Преобразует SFP в обычный dict.

    Удобно для логирования, JSON и Telegram.
    """

    return {
        "direction": sfp.direction,
        "level": sfp.level,
        "sweep_extreme": sfp.sweep_extreme,
        "sweep_index": sfp.sweep_index,
        "sweep_timestamp": str(
            sfp.sweep_timestamp
        ),
        "entry_reference": sfp.entry_reference,
        "atr": sfp.atr,
        "sweep_distance": sfp.sweep_distance,
        "sweep_atr_ratio": sfp.sweep_atr_ratio,
        "reclaim_ratio": sfp.reclaim_ratio,
        "candle_body_ratio": sfp.candle_body_ratio,
        "score": sfp.score,
    }
