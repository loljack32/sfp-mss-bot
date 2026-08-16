# ============================================================
# MSS — MARKET STRUCTURE SHIFT
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from config import (
    ATR_PERIOD,
    MIN_MSS_DISPLACEMENT_ATR,
    MIN_MSS_BODY_RATIO,
    MSS_REQUIRE_CLOSE,
)

from core.sfp import (
    SFPSetup,
    calculate_atr,
)

from core.structure import (
    SwingPoint,
)


# ============================================================
# DATA CLASS
# ============================================================

@dataclass(frozen=True)
class MSSSetup:
    """
    Подтверждённый Market Structure Shift.

    direction:
        LONG
        SHORT

    structure_level:
        уровень структуры, который был сломан.

    break_index:
        индекс свечи, которая подтвердила MSS.

    break_price:
        цена закрытия MSS-свечи.

    displacement:
        абсолютный размер движения от открытия до закрытия.

    displacement_atr_ratio:
        displacement / ATR.

    body_ratio:
        размер тела относительно полного диапазона свечи.

    score:
        качество MSS.
    """

    direction: str

    structure_level: float
    structure_label: str

    structure_index: int

    break_index: int
    break_timestamp: pd.Timestamp

    break_price: float

    displacement: float
    displacement_atr_ratio: float

    body_ratio: float

    candle_range: float

    score: float


# ============================================================
# CANDLE METRICS
# ============================================================

def _get_candle_metrics(
    row: pd.Series,
) -> tuple[float, float, float]:
    """
    Возвращает:

        candle_range
        body
        body_ratio
    """

    open_price = float(
        row["open"]
    )

    high = float(
        row["high"]
    )

    low = float(
        row["low"]
    )

    close = float(
        row["close"]
    )

    candle_range = high - low

    if candle_range <= 0:
        return 0.0, 0.0, 0.0

    body = abs(
        close - open_price
    )

    body_ratio = (
        body / candle_range
    )

    return (
        candle_range,
        body,
        body_ratio,
    )


# ============================================================
# BULLISH MSS
# ============================================================

def detect_bullish_mss(
    df: pd.DataFrame,
    sfp: SFPSetup,
    structure: List[SwingPoint],
    max_bars_after_sfp: int = 20,
) -> Optional[MSSSetup]:
    """
    Ищет bullish MSS после bullish SFP.

    Логика:

        Bullish SFP
             ↓
        найти последний значимый HIGH
             ↓
        цена закрывается выше HIGH
             ↓
        displacement достаточный
             ↓
        MSS confirmed

    Для bullish MSS нас интересует HIGH, который находится
    перед SFP и представляет собой локальную bearish structure.
    """

    if sfp.direction != "LONG":
        return None

    if sfp.sweep_index >= len(df):
        return None

    # --------------------------------------------------------
    # Ищем структурный HIGH перед SFP.
    # --------------------------------------------------------

    candidate_highs = [
        swing
        for swing in structure
        if (
            swing.kind == "HIGH"
            and swing.index < sfp.sweep_index
        )
    ]

    if not candidate_highs:
        return None

    # Последний подтверждённый high перед SFP.
    structure_high = candidate_highs[-1]

    start_index = (
        sfp.sweep_index + 1
    )

    end_index = min(
        len(df),
        start_index + max_bars_after_sfp,
    )

    atr_series = calculate_atr(
        df,
        period=ATR_PERIOD,
    )

    for i in range(
        start_index,
        end_index,
    ):

        row = df.iloc[i]

        atr_value = atr_series.iloc[i]

        if pd.isna(atr_value):
            continue

        atr = float(
            atr_value
        )

        if atr <= 0:
            continue

        close = float(
            row["close"]
        )

        # ----------------------------------------------------
        # STRUCTURE BREAK
        # ----------------------------------------------------

        if MSS_REQUIRE_CLOSE:

            if close <= structure_high.price:
                continue

        else:

            if float(row["high"]) <= structure_high.price:
                continue

        # ----------------------------------------------------
        # CANDLE METRICS
        # ----------------------------------------------------

        candle_range, body, body_ratio = (
            _get_candle_metrics(row)
        )

        if candle_range <= 0:
            continue

        # ----------------------------------------------------
        # DISPLACEMENT
        # ----------------------------------------------------

        open_price = float(
            row["open"]
        )

        displacement = (
            close - open_price
        )

        # Bullish MSS должен иметь bullish displacement.
        if displacement <= 0:
            continue

        displacement_atr_ratio = (
            displacement / atr
        )

        if (
            displacement_atr_ratio
            < MIN_MSS_DISPLACEMENT_ATR
        ):
            continue

        # ----------------------------------------------------
        # BODY STRENGTH
        # ----------------------------------------------------

        if (
            body_ratio
            < MIN_MSS_BODY_RATIO
        ):
            continue

        # ----------------------------------------------------
        # SCORE
        # ----------------------------------------------------

        score = calculate_mss_score(
            direction="LONG",
            displacement_atr_ratio=(
                displacement_atr_ratio
            ),
            body_ratio=body_ratio,
        )

        return MSSSetup(
            direction="LONG",
            structure_level=structure_high.price,
            structure_label=structure_high.label,
            structure_index=structure_high.index,
            break_index=i,
            break_timestamp=row["timestamp"],
            break_price=close,
            displacement=displacement,
            displacement_atr_ratio=(
                displacement_atr_ratio
            ),
            body_ratio=body_ratio,
            candle_range=candle_range,
            score=score,
        )

    return None


# ============================================================
# BEARISH MSS
# ============================================================

def detect_bearish_mss(
    df: pd.DataFrame,
    sfp: SFPSetup,
    structure: List[SwingPoint],
    max_bars_after_sfp: int = 20,
) -> Optional[MSSSetup]:
    """
    Ищет bearish MSS после bearish SFP.

    Логика:

        Bearish SFP
             ↓
        найти последний значимый LOW
             ↓
        цена закрывается ниже LOW
             ↓
        displacement
             ↓
        MSS confirmed
    """

    if sfp.direction != "SHORT":
        return None

    if sfp.sweep_index >= len(df):
        return None

    # --------------------------------------------------------
    # Ищем LOW перед SFP.
    # --------------------------------------------------------

    candidate_lows = [
        swing
        for swing in structure
        if (
            swing.kind == "LOW"
            and swing.index < sfp.sweep_index
        )
    ]

    if not candidate_lows:
        return None

    structure_low = candidate_lows[-1]

    start_index = (
        sfp.sweep_index + 1
    )

    end_index = min(
        len(df),
        start_index + max_bars_after_sfp,
    )

    atr_series = calculate_atr(
        df,
        period=ATR_PERIOD,
    )

    for i in range(
        start_index,
        end_index,
    ):

        row = df.iloc[i]

        atr_value = atr_series.iloc[i]

        if pd.isna(atr_value):
            continue

        atr = float(
            atr_value
        )

        if atr <= 0:
            continue

        close = float(
            row["close"]
        )

        # ----------------------------------------------------
        # STRUCTURE BREAK
        # ----------------------------------------------------

        if MSS_REQUIRE_CLOSE:

            if close >= structure_low.price:
                continue

        else:

            if float(row["low"]) >= structure_low.price:
                continue

        # ----------------------------------------------------
        # CANDLE METRICS
        # ----------------------------------------------------

        candle_range, body, body_ratio = (
            _get_candle_metrics(row)
        )

        if candle_range <= 0:
            continue

        # ----------------------------------------------------
        # DISPLACEMENT
        # ----------------------------------------------------

        open_price = float(
            row["open"]
        )

        displacement = (
            open_price - close
        )

        # Bearish MSS должен иметь bearish displacement.
        if displacement <= 0:
            continue

        displacement_atr_ratio = (
            displacement / atr
        )

        if (
            displacement_atr_ratio
            < MIN_MSS_DISPLACEMENT_ATR
        ):
            continue

        # ----------------------------------------------------
        # BODY
        # ----------------------------------------------------

        if (
            body_ratio
            < MIN_MSS_BODY_RATIO
        ):
            continue

        # ----------------------------------------------------
        # SCORE
        # ----------------------------------------------------

        score = calculate_mss_score(
            direction="SHORT",
            displacement_atr_ratio=(
                displacement_atr_ratio
            ),
            body_ratio=body_ratio,
        )

        return MSSSetup(
            direction="SHORT",
            structure_level=structure_low.price,
            structure_label=structure_low.label,
            structure_index=structure_low.index,
            break_index=i,
            break_timestamp=row["timestamp"],
            break_price=close,
            displacement=displacement,
            displacement_atr_ratio=(
                displacement_atr_ratio
            ),
            body_ratio=body_ratio,
            candle_range=candle_range,
            score=score,
        )

    return None


# ============================================================
# MSS SCORE
# ============================================================

def calculate_mss_score(
    direction: str,
    displacement_atr_ratio: float,
    body_ratio: float,
) -> float:
    """
    Оценивает только качество MSS.

    Финальный signal score рассчитывается позже.
    """

    score = 0.0

    # --------------------------------------------------------
    # DISPLACEMENT
    # --------------------------------------------------------

    if displacement_atr_ratio >= 0.80:
        score += 50

    elif displacement_atr_ratio >= 0.50:
        score += 40

    elif displacement_atr_ratio >= 0.30:
        score += 30

    elif displacement_atr_ratio >= 0.20:
        score += 20

    else:
        score += 5

    # --------------------------------------------------------
    # BODY
    # --------------------------------------------------------

    if body_ratio >= 0.80:
        score += 40

    elif body_ratio >= 0.70:
        score += 35

    elif body_ratio >= 0.60:
        score += 30

    elif body_ratio >= 0.50:
        score += 20

    else:
        score += 5

    # --------------------------------------------------------
    # DIRECTION
    # --------------------------------------------------------

    if direction in {
        "LONG",
        "SHORT",
    }:
        score += 10

    return min(
        100.0,
        score,
    )


# ============================================================
# FIND MSS AFTER SFP
# ============================================================

def find_mss_after_sfp(
    df: pd.DataFrame,
    sfp: SFPSetup,
    structure: List[SwingPoint],
    max_bars_after_sfp: int = 20,
) -> Optional[MSSSetup]:
    """
    Главная функция.

    Для LONG ищет bullish MSS.

    Для SHORT ищет bearish MSS.
    """

    if sfp.direction == "LONG":

        return detect_bullish_mss(
            df=df,
            sfp=sfp,
            structure=structure,
            max_bars_after_sfp=(
                max_bars_after_sfp
            ),
        )

    if sfp.direction == "SHORT":

        return detect_bearish_mss(
            df=df,
            sfp=sfp,
            structure=structure,
            max_bars_after_sfp=(
                max_bars_after_sfp
            ),
        )

    return None


# ============================================================
# FIND LATEST CONFIRMED MSS
# ============================================================

def find_latest_mss(
    df: pd.DataFrame,
    sfps: List[SFPSetup],
    structure: List[SwingPoint],
    max_bars_after_sfp: int = 20,
) -> Optional[tuple[SFPSetup, MSSSetup]]:
    """
    Перебирает SFP от самого нового к старому
    и возвращает первый SFP, после которого
    действительно появился MSS.
    """

    if not sfps:
        return None

    ordered = sorted(
        sfps,
        key=lambda x: x.sweep_index,
        reverse=True,
    )

    for sfp in ordered:

        mss = find_mss_after_sfp(
            df=df,
            sfp=sfp,
            structure=structure,
            max_bars_after_sfp=(
                max_bars_after_sfp
            ),
        )

        if mss is not None:
            return sfp, mss

    return None


# ============================================================
# MSS INVALIDATION
# ============================================================

def is_mss_invalidated(
    df: pd.DataFrame,
    sfp: SFPSetup,
    mss: MSSSetup,
) -> bool:
    """
    Проверяет, была ли идея MSS сломана.

    LONG:
        после MSS close ниже SFP extreme.

    SHORT:
        после MSS close выше SFP extreme.
    """

    future = df.iloc[
        mss.break_index + 1:
    ]

    if future.empty:
        return False

    if mss.direction == "LONG":

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
# SERIALIZATION
# ============================================================

def mss_to_dict(
    mss: MSSSetup,
) -> dict:
    """
    Преобразует MSS в обычный dict.
    """

    return {
        "direction": mss.direction,
        "structure_level": mss.structure_level,
        "structure_label": mss.structure_label,
        "structure_index": mss.structure_index,
        "break_index": mss.break_index,
        "break_timestamp": str(
            mss.break_timestamp
        ),
        "break_price": mss.break_price,
        "displacement": mss.displacement,
        "displacement_atr_ratio": (
            mss.displacement_atr_ratio
        ),
        "body_ratio": mss.body_ratio,
        "candle_range": mss.candle_range,
        "score": mss.score,
    }
