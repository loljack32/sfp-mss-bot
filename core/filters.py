# ============================================================
# SFP + MSS SIGNAL FILTERS
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from config import (
    ATR_PERIOD,
    MIN_ATR_PERCENT,
    MIN_VOLUME_RATIO,
    VOLUME_LOOKBACK,
    MIN_SIGNAL_SCORE,
    MIN_RR,
    SIGNAL_SCORE_WEIGHTS,
    ENABLE_COUNTER_TREND_SIGNALS,
)

from core.sfp import (
    SFPSetup,
    calculate_atr,
)

from core.mss import (
    MSSSetup,
)

from core.structure import (
    SwingPoint,
    StructureState,
)


# ============================================================
# DATA CLASS
# ============================================================

@dataclass(frozen=True)
class FilterResult:
    passed: bool
    score: float
    reasons: List[str]
    warnings: List[str]
    metrics: dict
    setup_type: str


# ============================================================
# VOLUME
# ============================================================

def calculate_volume_ratio(
    df: pd.DataFrame,
    index: int,
    lookback: int = VOLUME_LOOKBACK,
) -> Optional[float]:
    if index < 1:
        return None

    start = max(0, index - lookback)
    previous = df.iloc[start:index]

    if previous.empty:
        return None

    average_volume = float(previous["volume"].mean())
    if average_volume <= 0:
        return None

    current_volume = float(df.iloc[index]["volume"])
    return current_volume / average_volume


# ============================================================
# ATR PERCENT
# ============================================================

def calculate_atr_percent(
    df: pd.DataFrame,
    index: int,
) -> Optional[float]:
    if index < ATR_PERIOD:
        return None

    atr_series = calculate_atr(df, period=ATR_PERIOD)
    atr_value = atr_series.iloc[index]

    if pd.isna(atr_value):
        return None

    close = float(df.iloc[index]["close"])
    if close <= 0:
        return None

    return float(atr_value) / close


# ============================================================
# HTF BIAS
# ============================================================

def check_htf_alignment(
    direction: str,
    htf_state: StructureState,
    allow_counter_trend: bool = ENABLE_COUNTER_TREND_SIGNALS,
    *args,
    **kwargs,
) -> tuple[bool, str, str]:
    if htf_state.trend not in {"BULLISH", "BEARISH"}:
        return False, f"HTF structure is {htf_state.trend} (no clear trend)", "UNKNOWN"

    if direction == "LONG":
        if htf_state.trend == "BULLISH":
            return True, "HTF bullish alignment", "TREND"

        if htf_state.trend == "BEARISH":
            if allow_counter_trend:
                return True, "Counter-trend LONG pullback against 4H Bearish", "COUNTER_TREND"
            return False, "HTF bearish against LONG", "COUNTER_TREND"

    if direction == "SHORT":
        if htf_state.trend == "BEARISH":
            return True, "HTF bearish alignment", "TREND"

        if htf_state.trend == "BULLISH":
            if allow_counter_trend:
                return True, "Counter-trend SHORT pullback against 4H Bullish", "COUNTER_TREND"
            return False, "HTF bullish against SHORT", "COUNTER_TREND"

    return False, "Unknown signal direction", "UNKNOWN"


# ============================================================
# STRUCTURE QUALITY
# ============================================================

def check_local_structure(
    direction: str,
    sfp: SFPSetup,
    mss: MSSSetup,
    structure: List[SwingPoint],
) -> tuple[bool, str]:
    if direction == "LONG":
        if mss.structure_label == "LH":
            return True, "Bullish MSS broke LH"
        if mss.structure_label == "HH":
            return True, "Bullish MSS broke HH"
        return False, "Bullish MSS did not break a valid LH/HH structure level"

    if direction == "SHORT":
        if mss.structure_label == "HL":
            return True, "Bearish MSS broke HL"
        if mss.structure_label == "LL":
            return True, "Bearish MSS broke LL"
        return False, "Bearish MSS did not break a valid HL/LL structure level"

    return False, "Unknown direction"


# ============================================================
# LIQUIDITY LEVELS & TP
# ============================================================

def get_liquidity_targets(
    direction: str,
    current_index: int,
    structure: List[SwingPoint],
    current_price: float,
) -> List[SwingPoint]:
    if direction == "LONG":
        targets = [
            swing
            for swing in structure
            if (
                swing.kind == "HIGH"
                and swing.index < current_index
                and swing.price > current_price
            )
        ]
        return sorted(targets, key=lambda x: x.price)

    if direction == "SHORT":
        targets = [
            swing
            for swing in structure
            if (
                swing.kind == "LOW"
                and swing.index < current_index
                and swing.price < current_price
            )
        ]
        return sorted(targets, key=lambda x: x.price, reverse=True)

    return []


def find_nearest_liquidity_target(
    direction: str,
    current_index: int,
    current_price: float,
    structure: List[SwingPoint],
) -> Optional[SwingPoint]:
    targets = get_liquidity_targets(
        direction=direction,
        current_index=current_index,
        structure=structure,
        current_price=current_price,
    )
    if not targets:
        return None
    return targets[0]


# ============================================================
# STOP LOSS & RR
# ============================================================

def calculate_stop_loss(
    direction: str,
    sfp: SFPSetup,
    atr: float,
    buffer_atr: float = 0.05,
) -> Optional[float]:
    if atr <= 0:
        return None

    buffer = atr * buffer_atr

    if direction == "LONG":
        return sfp.sweep_extreme - buffer

    if direction == "SHORT":
        return sfp.sweep_extreme + buffer

    return None


def calculate_rr(
    direction: str,
    entry: float,
    stop_loss: float,
    target: float,
) -> Optional[float]:
    if direction == "LONG":
        risk = entry - stop_loss
        reward = target - entry
    elif direction == "SHORT":
        risk = stop_loss - entry
        reward = entry - target
    else:
        return None

    if risk <= 0 or reward <= 0:
        return None

    return reward / risk


def check_entry_distance_from_mss(
    direction: str,
    entry: float,
    mss: MSSSetup,
    atr: float,
) -> tuple[bool, str]:
    if atr <= 0:
        return False, "Invalid ATR"

    distance = abs(entry - mss.break_price)
    distance_atr = distance / atr

    if distance_atr <= 1.0:
        return True, f"Entry distance {distance_atr:.2f} ATR"

    return False, f"Entry too far from MSS: {distance_atr:.2f} ATR"


# ============================================================
# SCORE HELPERS
# ============================================================

def score_htf(setup_type: str) -> float:
    if setup_type == "TREND":
        return 100.0
    if setup_type == "COUNTER_TREND":
        return 75.0
    return 0.0


def score_liquidity(target: Optional[SwingPoint]) -> float:
    return 100.0 if target is not None else 0.0


def score_sfp(sfp: SFPSetup) -> float:
    return float(sfp.score)


def score_mss(mss: MSSSetup) -> float:
    return float(mss.score)


def score_displacement(mss: MSSSetup) -> float:
    ratio = mss.displacement_atr_ratio
    if ratio >= 1.0:
        return 100.0
    if ratio >= 0.80:
        return 90.0
    if ratio >= 0.50:
        return 75.0
    if ratio >= 0.30:
        return 60.0
    if ratio >= 0.20:
        return 45.0
    return 0.0


def score_volume(volume_ratio: Optional[float]) -> float:
    if volume_ratio is None:
        return 0.0
    if volume_ratio >= 2.0:
        return 100.0
    if volume_ratio >= 1.5:
        return 90.0
    if volume_ratio >= 1.2:
        return 75.0
    if volume_ratio >= 1.0:
        return 50.0
    return 20.0


def score_rr(rr: Optional[float]) -> float:
    if rr is None:
        return 0.0
    if rr >= 4.0:
        return 100.0
    if rr >= 3.0:
        return 95.0
    if rr >= 2.5:
        return 90.0
    if rr >= 2.0:
        return 80.0
    if rr >= 1.6:
        return 70.0
    if rr >= 1.4:
        return 50.0
    return 0.0


def calculate_signal_score(
    htf_score: float,
    liquidity_score: float,
    sfp_score: float,
    mss_score: float,
    displacement_score: float,
    volume_score: float,
    rr_score: float,
) -> float:
    weights = SIGNAL_SCORE_WEIGHTS
    total_weight = sum(weights.values())

    if total_weight <= 0:
        return 0.0

    weighted = (
        htf_score * weights["htf_structure"]
        + liquidity_score * weights["liquidity"]
        + sfp_score * weights["sfp"]
        + mss_score * weights["mss"]
        + displacement_score * weights["displacement"]
        + volume_score * weights["volume"]
        + rr_score * weights["risk_reward"]
    )

    return weighted / total_weight


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate_setup(
    df: pd.DataFrame,
    sfp: SFPSetup,
    mss: MSSSetup,
    structure: List[SwingPoint],
    htf_state: StructureState,
    entry_price: Optional[float] = None,
    allow_counter_trend: bool = ENABLE_COUNTER_TREND_SIGNALS,
    *args,
    **kwargs,
) -> FilterResult:
    r
