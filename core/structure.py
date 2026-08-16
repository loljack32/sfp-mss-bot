# ============================================================
# MARKET STRUCTURE
# HH / HL / LH / LL
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from config import (
    SWING_LEFT,
    SWING_RIGHT,
    HTF_SWING_LEFT,
    HTF_SWING_RIGHT,
)


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass(frozen=True)
class SwingPoint:
    """
    Подтверждённая swing-точка.

    kind:
        HIGH или LOW

    label:
        HH / LH / HL / LL / UNKNOWN

    index:
        индекс свечи в исходном DataFrame

    price:
        цена swing high / swing low

    timestamp:
        время свечи
    """

    index: int
    timestamp: pd.Timestamp
    price: float
    kind: str
    label: str


@dataclass(frozen=True)
class StructureState:
    """
    Текущее состояние рыночной структуры.
    """

    trend: str

    last_high: Optional[SwingPoint]
    previous_high: Optional[SwingPoint]

    last_low: Optional[SwingPoint]
    previous_low: Optional[SwingPoint]

    last_swing: Optional[SwingPoint]


# ============================================================
# VALIDATION
# ============================================================

def _validate_ohlcv(df: pd.DataFrame) -> None:
    """
    Проверяет наличие необходимых OHLCV колонок.
    """

    required = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            "DataFrame is missing required columns: "
            f"{sorted(missing)}"
        )

    if len(df) == 0:
        raise ValueError("DataFrame is empty.")


# ============================================================
# SWING HIGH
# ============================================================

def _is_swing_high(
    highs: np.ndarray,
    index: int,
    left: int,
    right: int,
) -> bool:

    if index - left < 0:
        return False

    if index + right >= len(highs):
        return False

    current = highs[index]

    left_values = highs[
        index - left:index
    ]

    right_values = highs[
        index + 1:index + right + 1
    ]

    # Текущий high должен быть не ниже всех соседей.
    if current < np.max(left_values):
        return False

    if current < np.max(right_values):
        return False

    # Чтобы не считать плоскую серию одинаковых high
    # несколькими swing high, требуем хотя бы одного
    # строго меньшего значения с каждой стороны.
    if not np.any(current > left_values):
        return False

    if not np.any(current > right_values):
        return False

    return True


# ============================================================
# SWING LOW
# ============================================================

def _is_swing_low(
    lows: np.ndarray,
    index: int,
    left: int,
    right: int,
) -> bool:

    if index - left < 0:
        return False

    if index + right >= len(lows):
        return False

    current = lows[index]

    left_values = lows[
        index - left:index
    ]

    right_values = lows[
        index + 1:index + right + 1
    ]

    if current > np.min(left_values):
        return False

    if current > np.min(right_values):
        return False

    if not np.any(current < left_values):
        return False

    if not np.any(current < right_values):
        return False

    return True


# ============================================================
# RAW SWINGS
# ============================================================

def detect_raw_swings(
    df: pd.DataFrame,
    left: int = SWING_LEFT,
    right: int = SWING_RIGHT,
) -> List[dict]:
    """
    Определяет swing high / swing low.

    ВАЖНО:

    Swing на индексе i считается подтверждённым только после
    появления right свечей справа.

    Это принципиально важно для предотвращения look-ahead bias.

    Возвращает список словарей без HH/HL/LH/LL классификации.
    """

    _validate_ohlcv(df)

    if left < 1:
        raise ValueError("left must be >= 1")

    if right < 1:
        raise ValueError("right must be >= 1")

    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)

    swings: List[dict] = []

    start = left
    end = len(df) - right

    for i in range(start, end):

        is_high = _is_swing_high(
            highs,
            i,
            left,
            right,
        )

        is_low = _is_swing_low(
            lows,
            i,
            left,
            right,
        )

        if is_high:
            swings.append(
                {
                    "index": i,
                    "timestamp": df.iloc[i]["timestamp"],
                    "price": float(df.iloc[i]["high"]),
                    "kind": "HIGH",
                }
            )

        if is_low:
            swings.append(
                {
                    "index": i,
                    "timestamp": df.iloc[i]["timestamp"],
                    "price": float(df.iloc[i]["low"]),
                    "kind": "LOW",
                }
            )

    swings.sort(
        key=lambda x: (
            x["index"],
            0 if x["kind"] == "HIGH" else 1,
        )
    )

    return swings


# ============================================================
# SWING CLASSIFICATION
# ============================================================

def _classify_high(
    current_price: float,
    previous_price: Optional[float],
) -> str:

    if previous_price is None:
        return "UNKNOWN"

    if current_price > previous_price:
        return "HH"

    if current_price < previous_price:
        return "LH"

    return "EQH"


def _classify_low(
    current_price: float,
    previous_price: Optional[float],
) -> str:

    if previous_price is None:
        return "UNKNOWN"

    if current_price > previous_price:
        return "HL"

    if current_price < previous_price:
        return "LL"

    return "EQL"


# ============================================================
# STRUCTURE SWINGS
# ============================================================

def build_structure(
    df: pd.DataFrame,
    left: int = SWING_LEFT,
    right: int = SWING_RIGHT,
) -> List[SwingPoint]:
    """
    Определяет swing points и классифицирует их:

        HIGH:
            HH
            LH
            EQH

        LOW:
            HL
            LL
            EQL

    UNKNOWN используется только для первого swing данного типа.
    """

    raw_swings = detect_raw_swings(
        df,
        left=left,
        right=right,
    )

    structure: List[SwingPoint] = []

    previous_high_price: Optional[float] = None
    previous_low_price: Optional[float] = None

    for swing in raw_swings:

        if swing["kind"] == "HIGH":

            label = _classify_high(
                swing["price"],
                previous_high_price,
            )

            previous_high_price = swing["price"]

        else:

            label = _classify_low(
                swing["price"],
                previous_low_price,
            )

            previous_low_price = swing["price"]

        structure.append(
            SwingPoint(
                index=swing["index"],
                timestamp=swing["timestamp"],
                price=swing["price"],
                kind=swing["kind"],
                label=label,
            )
        )

    return structure


# ============================================================
# STRUCTURE STATE
# ============================================================

def get_structure_state(
    structure: List[SwingPoint],
) -> StructureState:

    highs = [
        swing
        for swing in structure
        if swing.kind == "HIGH"
    ]

    lows = [
        swing
        for swing in structure
        if swing.kind == "LOW"
    ]

    last_high = highs[-1] if highs else None

    previous_high = (
        highs[-2]
        if len(highs) >= 2
        else None
    )

    last_low = lows[-1] if lows else None

    previous_low = (
        lows[-2]
        if len(lows) >= 2
        else None
    )

    last_swing = (
        structure[-1]
        if structure
        else None
    )

    trend = determine_structure_trend(
        last_high=last_high,
        previous_high=previous_high,
        last_low=last_low,
        previous_low=previous_low,
    )

    return StructureState(
        trend=trend,
        last_high=last_high,
        previous_high=previous_high,
        last_low=last_low,
        previous_low=previous_low,
        last_swing=last_swing,
    )


# ============================================================
# TREND
# ============================================================

def determine_structure_trend(
    last_high: Optional[SwingPoint],
    previous_high: Optional[SwingPoint],
    last_low: Optional[SwingPoint],
    previous_low: Optional[SwingPoint],
) -> str:
    """
    Определяет базовое состояние структуры.

    BULLISH:
        HH + HL

    BEARISH:
        LH + LL

    MIXED:
        структура противоречива или недостаточна.

    UNKNOWN:
        недостаточно данных.
    """

    if (
        last_high is None
        or previous_high is None
        or last_low is None
        or previous_low is None
    ):
        return "UNKNOWN"

    higher_high = (
        last_high.price > previous_high.price
    )

    higher_low = (
        last_low.price > previous_low.price
    )

    lower_high = (
        last_high.price < previous_high.price
    )

    lower_low = (
        last_low.price < previous_low.price
    )

    if higher_high and higher_low:
        return "BULLISH"

    if lower_high and lower_low:
        return "BEARISH"

    return "MIXED"


# ============================================================
# LAST CONFIRMED HIGH / LOW
# ============================================================

def get_last_confirmed_high(
    structure: List[SwingPoint],
    before_index: Optional[int] = None,
) -> Optional[SwingPoint]:

    candidates = [
        swing
        for swing in structure
        if swing.kind == "HIGH"
    ]

    if before_index is not None:

        candidates = [
            swing
            for swing in candidates
            if swing.index < before_index
        ]

    if not candidates:
        return None

    return candidates[-1]


def get_last_confirmed_low(
    structure: List[SwingPoint],
    before_index: Optional[int] = None,
) -> Optional[SwingPoint]:

    candidates = [
        swing
        for swing in structure
        if swing.kind == "LOW"
    ]

    if before_index is not None:

        candidates = [
            swing
            for swing in candidates
            if swing.index < before_index
        ]

    if not candidates:
        return None

    return candidates[-1]


# ============================================================
# STRUCTURE BREAK
# ============================================================

def bullish_structure_break(
    df: pd.DataFrame,
    level: SwingPoint,
    start_index: int,
) -> Optional[int]:
    """
    Ищет первое закрытие выше указанного swing high.

    Возвращает индекс свечи, которая закрылась выше уровня.

    Используется позже для MSS.
    """

    if level.kind != "HIGH":
        raise ValueError(
            "bullish_structure_break requires HIGH level"
        )

    closes = df["close"].to_numpy(dtype=float)

    start = max(
        start_index,
        level.index + 1,
    )

    for i in range(start, len(df)):

        if closes[i] > level.price:
            return i

    return None


def bearish_structure_break(
    df: pd.DataFrame,
    level: SwingPoint,
    start_index: int,
) -> Optional[int]:
    """
    Ищет первое закрытие ниже указанного swing low.
    """

    if level.kind != "LOW":
        raise ValueError(
            "bearish_structure_break requires LOW level"
        )

    closes = df["close"].to_numpy(dtype=float)

    start = max(
        start_index,
        level.index + 1,
    )

    for i in range(start, len(df)):

        if closes[i] < level.price:
            return i

    return None


# ============================================================
# STRUCTURE SUMMARY
# ============================================================

def structure_summary(
    structure: List[SwingPoint],
) -> dict:
    """
    Удобный сериализуемый summary для логов.
    """

    state = get_structure_state(
        structure
    )

    return {
        "trend": state.trend,

        "last_high": (
            state.last_high.price
            if state.last_high
            else None
        ),

        "last_high_label": (
            state.last_high.label
            if state.last_high
            else None
        ),

        "previous_high": (
            state.previous_high.price
            if state.previous_high
            else None
        ),

        "previous_high_label": (
            state.previous_high.label
            if state.previous_high
            else None
        ),

        "last_low": (
            state.last_low.price
            if state.last_low
            else None
        ),

        "last_low_label": (
            state.last_low.label
            if state.last_low
            else None
        ),

        "previous_low": (
            state.previous_low.price
            if state.previous_low
            else None
        ),

        "previous_low_label": (
            state.previous_low.label
            if state.previous_low
            else None
        ),

        "last_swing": (
            {
                "price": state.last_swing.price,
                "kind": state.last_swing.kind,
                "label": state.last_swing.label,
                "index": state.last_swing.index,
            }
            if state.last_swing
            else None
        ),
    }


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def analyze_structure(
    df: pd.DataFrame,
    htf: bool = False,
) -> tuple[List[SwingPoint], StructureState]:

    if htf:
        left = HTF_SWING_LEFT
        right = HTF_SWING_RIGHT

    else:
        left = SWING_LEFT
        right = SWING_RIGHT

    structure = build_structure(
        df,
        left=left,
        right=right,
    )

    state = get_structure_state(
        structure
    )

    return structure, state
