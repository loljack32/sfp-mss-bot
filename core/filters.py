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
    """
    Результат проверки качества сетапа.

    passed:
        прошёл ли сетап все обязательные проверки

    score:
        итоговый score 0-100

    reasons:
        причины отказа

    warnings:
        предупреждения

    metrics:
        числовые метрики для логов.
    """

    passed: bool

    score: float

    reasons: List[str]

    warnings: List[str]

    metrics: dict


# ============================================================
# VOLUME
# ============================================================

def calculate_volume_ratio(
    df: pd.DataFrame,
    index: int,
    lookback: int = VOLUME_LOOKBACK,
) -> Optional[float]:
    """
    Объём сигнальной свечи / средний объём предыдущих свечей.

    ВАЖНО:

    В среднее не включается текущая свеча.

    Иначе сильная сигнальная свеча сама повышает среднее
    и искусственно уменьшает volume ratio.
    """

    if index < 1:
        return None

    start = max(
        0,
        index - lookback,
    )

    previous = df.iloc[
        start:index
    ]

    if previous.empty:
        return None

    average_volume = float(
        previous["volume"].mean()
    )

    if average_volume <= 0:
        return None

    current_volume = float(
        df.iloc[index]["volume"]
    )

    return (
        current_volume
        / average_volume
    )


# ============================================================
# ATR PERCENT
# ============================================================

def calculate_atr_percent(
    df: pd.DataFrame,
    index: int,
) -> Optional[float]:
    """
    ATR / close.

    Например:

        ATR = 100
        BTC = 50 000

        ATR % = 0.002 = 0.2%
    """

    if index < ATR_PERIOD:
        return None

    atr_series = calculate_atr(
        df,
        period=ATR_PERIOD,
    )

    atr_value = atr_series.iloc[index]

    if pd.isna(atr_value):
        return None

    close = float(
        df.iloc[index]["close"]
    )

    if close <= 0:
        return None

    return (
        float(atr_value)
        / close
    )


# ============================================================
# HTF BIAS
# ============================================================

def check_htf_alignment(
    direction: str,
    htf_state: StructureState,
) -> tuple[bool, str]:
    """
    Проверяет соответствие направления сигнала
    направлению старшего таймфрейма.

    LONG:
        HTF должен быть BULLISH.

    SHORT:
        HTF должен быть BEARISH.

    Исключение:

        UNKNOWN / MIXED

    не разрешаем как полноценное направление.
    """

    if direction == "LONG":

        if htf_state.trend == "BULLISH":
            return True, "HTF bullish alignment"

        if htf_state.trend == "BEARISH":
            return False, "HTF bearish against LONG"

        return False, (
            "HTF structure is not clearly bullish"
        )

    if direction == "SHORT":

        if htf_state.trend == "BEARISH":
            return True, "HTF bearish alignment"

        if htf_state.trend == "BULLISH":
            return False, "HTF bullish against SHORT"

        return False, (
            "HTF structure is not clearly bearish"
        )

    return False, "Unknown signal direction"


# ============================================================
# STRUCTURE QUALITY
# ============================================================

def check_local_structure(
    direction: str,
    sfp: SFPSetup,
    mss: MSSSetup,
    structure: List[SwingPoint],
) -> tuple[bool, str]:
    """
    Проверяет, что MSS действительно связан
    с локальной структурой.

    LONG:

        предпочтительно ломается LH.

    SHORT:

        предпочтительно ломается HL.

    Если MSS ломает HH/LL, это не обязательно ошибка,
    но качество сетапа будет ниже.
    """

    if direction == "LONG":

        if mss.structure_label == "LH":
            return True, "Bullish MSS broke LH"

        if mss.structure_label == "HH":
            return True, (
                "Bullish MSS broke HH; "
                "continuation-style structure"
            )

        return False, (
            "Bullish MSS did not break a valid "
            "LH/HH structure level"
        )

    if direction == "SHORT":

        if mss.structure_label == "HL":
            return True, "Bearish MSS broke HL"

        if mss.structure_label == "LL":
            return True, (
                "Bearish MSS broke LL; "
                "continuation-style structure"
            )

        return False, (
            "Bearish MSS did not break a valid "
            "HL/LL structure level"
        )

    return False, "Unknown direction"


# ============================================================
# LIQUIDITY LEVELS
# ============================================================

def get_liquidity_targets(
    direction: str,
    current_index: int,
    structure: List[SwingPoint],
    current_price: float,
) -> List[SwingPoint]:
    """
    Ищет потенциальные противоположные liquidity targets.

    LONG:
        ищем HIGH выше текущей цены.

    SHORT:
        ищем LOW ниже текущей цены.
    """

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

        return sorted(
            targets,
            key=lambda x: x.price,
        )

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

        return sorted(
            targets,
            key=lambda x: x.price,
            reverse=True,
        )

    return []


# ============================================================
# TP CANDIDATES
# ============================================================

def find_nearest_liquidity_target(
    direction: str,
    current_index: int,
    current_price: float,
    structure: List[SwingPoint],
) -> Optional[SwingPoint]:
    """
    Возвращает ближайшую противоположную liquidity target.
    """

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
# STOP LOSS
# ============================================================

def calculate_stop_loss(
    direction: str,
    sfp: SFPSetup,
    atr: float,
    buffer_atr: float = 0.05,
) -> Optional[float]:
    """
    Рассчитывает технический SL.

    LONG:
        ниже экстремума SFP.

    SHORT:
        выше экстремума SFP.

    Добавляем небольшой ATR buffer,
    чтобы стоп не стоял точно на экстремуме.
    """

    if atr <= 0:
        return None

    buffer = (
        atr * buffer_atr
    )

    if direction == "LONG":

        return (
            sfp.sweep_extreme
            - buffer
        )

    if direction == "SHORT":

        return (
            sfp.sweep_extreme
            + buffer
        )

    return None


# ============================================================
# RISK / REWARD
# ============================================================

def calculate_rr(
    direction: str,
    entry: float,
    stop_loss: float,
    target: float,
) -> Optional[float]:
    """
    Рассчитывает RR.
    """

    if direction == "LONG":

        risk = (
            entry - stop_loss
        )

        reward = (
            target - entry
        )

    elif direction == "SHORT":

        risk = (
            stop_loss - entry
        )

        reward = (
            entry - target
        )

    else:
        return None

    if risk <= 0:
        return None

    if reward <= 0:
        return None

    return (
        reward / risk
    )


# ============================================================
# RANGE / EXTENSION FILTER
# ============================================================

def check_entry_distance_from_mss(
    direction: str,
    entry: float,
    mss: MSSSetup,
    atr: float,
) -> tuple[bool, str]:
    """
    Проверяет, не слишком ли далеко цена находится
    от MSS после импульса.

    Это защита от FOMO-входов.

    Если цена уже улетела далеко от MSS,
    не хотим покупать/продавать после движения.
    """

    if atr <= 0:
        return False, "Invalid ATR"

    distance = abs(
        entry - mss.break_price
    )

    distance_atr = (
        distance / atr
    )

    # Если текущая цена не дальше 1 ATR от MSS,
    # сетап ещё может быть актуальным.
    if distance_atr <= 1.0:

        return True, (
            f"Entry distance {distance_atr:.2f} ATR"
        )

    return False, (
        f"Entry too far from MSS: "
        f"{distance_atr:.2f} ATR"
    )


# ============================================================
# SCORE HELPERS
# ============================================================

def score_htf(
    aligned: bool,
) -> float:

    return (
        100.0
        if aligned
        else 0.0
    )


def score_liquidity(
    target: Optional[SwingPoint],
) -> float:

    if target is None:
        return 0.0

    return 100.0


def score_sfp(
    sfp: SFPSetup,
) -> float:

    return float(
        sfp.score
    )


def score_mss(
    mss: MSSSetup,
) -> float:

    return float(
        mss.score
    )


def score_displacement(
    mss: MSSSetup,
) -> float:

    ratio = (
        mss.displacement_atr_ratio
    )

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


def score_volume(
    volume_ratio: Optional[float],
) -> float:

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


def score_rr(
    rr: Optional[float],
) -> float:

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

    if rr >= 1.5:
        return 50.0

    return 0.0


# ============================================================
# FINAL SCORE
# ============================================================

def calculate_signal_score(
    htf_score: float,
    liquidity_score: float,
    sfp_score: float,
    mss_score: float,
    displacement_score: float,
    volume_score: float,
    rr_score: float,
) -> float:
    """
    Рассчитывает финальный score.

    Вес задаётся config.py.
    """

    weights = SIGNAL_SCORE_WEIGHTS

    total_weight = sum(
        weights.values()
    )

    if total_weight <= 0:
        return 0.0

    weighted = (
        htf_score
        * weights["htf_structure"]
        +
        liquidity_score
        * weights["liquidity"]
        +
        sfp_score
        * weights["sfp"]
        +
        mss_score
        * weights["mss"]
        +
        displacement_score
        * weights["displacement"]
        +
        volume_score
        * weights["volume"]
        +
        rr_score
        * weights["risk_reward"]
    )

    return (
        weighted
        / total_weight
    )


# ============================================================
# MAIN FILTER
# ============================================================

def evaluate_setup(
    df: pd.DataFrame,
    sfp: SFPSetup,
    mss: MSSSetup,
    structure: List[SwingPoint],
    htf_state: StructureState,
    entry_price: Optional[float] = None,
) -> FilterResult:
    """
    Главная функция оценки SFP + MSS.

    Она НЕ создаёт окончательный торговый сигнал.

    Она определяет:

        можно ли вообще рассматривать
        данный сетап для торговли.
    """

    reasons: List[str] = []

    warnings: List[str] = []

    metrics = {}

    direction = sfp.direction

    # --------------------------------------------------------
    # ENTRY
    # --------------------------------------------------------

    if entry_price is None:
        entry_price = float(
            df.iloc[-1]["close"]
        )

    entry = float(
        entry_price
    )

    metrics["entry"] = entry

    # --------------------------------------------------------
    # HTF
    # --------------------------------------------------------

    htf_aligned, htf_reason = (
        check_htf_alignment(
            direction=direction,
            htf_state=htf_state,
        )
    )

    metrics["htf_trend"] = (
        htf_state.trend
    )

    metrics["htf_aligned"] = (
        htf_aligned
    )

    if not htf_aligned:

        reasons.append(
            htf_reason
        )

    # --------------------------------------------------------
    # LOCAL STRUCTURE
    # --------------------------------------------------------

    local_structure_ok, structure_reason = (
        check_local_structure(
            direction=direction,
            sfp=sfp,
            mss=mss,
            structure=structure,
        )
    )

    metrics["local_structure_ok"] = (
        local_structure_ok
    )

    if not local_structure_ok:

        reasons.append(
            structure_reason
        )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    atr = float(
        sfp.atr
    )

    atr_percent = (
        atr / entry
        if entry > 0
        else 0
    )

    metrics["atr"] = atr

    metrics["atr_percent"] = (
        atr_percent
    )

    if atr_percent < MIN_ATR_PERCENT:

        reasons.append(
            "ATR volatility is too low"
        )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    volume_ratio = (
        calculate_volume_ratio(
            df=df,
            index=mss.break_index,
        )
    )

    metrics["volume_ratio"] = (
        volume_ratio
    )

    if (
        volume_ratio is None
        or volume_ratio < MIN_VOLUME_RATIO
    ):

        warnings.append(
            "Volume confirmation is weak"
        )

    # --------------------------------------------------------
    # ENTRY DISTANCE
    # --------------------------------------------------------

    entry_distance_ok, entry_distance_reason = (
        check_entry_distance_from_mss(
            direction=direction,
            entry=entry,
            mss=mss,
            atr=atr,
        )
    )

    metrics["entry_distance_ok"] = (
        entry_distance_ok
    )

    if not entry_distance_ok:

        reasons.append(
            entry_distance_reason
        )

    # --------------------------------------------------------
    # STOP LOSS
    # --------------------------------------------------------

    stop_loss = calculate_stop_loss(
        direction=direction,
        sfp=sfp,
        atr=atr,
    )

    metrics["stop_loss"] = (
        stop_loss
    )

    if stop_loss is None:

        reasons.append(
            "Unable to calculate stop loss"
        )

    # --------------------------------------------------------
    # STOP VALIDATION
    # --------------------------------------------------------

    if stop_loss is not None:

        if direction == "LONG":

            if stop_loss >= entry:

                reasons.append(
                    "LONG stop loss is not below entry"
                )

        elif direction == "SHORT":

            if stop_loss <= entry:

                reasons.append(
                    "SHORT stop loss is not above entry"
                )

    # --------------------------------------------------------
    # LIQUIDITY TARGET
    # --------------------------------------------------------

    target = find_nearest_liquidity_target(
        direction=direction,
        current_index=mss.break_index,
        current_price=entry,
        structure=structure,
    )

    metrics["liquidity_target"] = (
        target.price
        if target is not None
        else None
    )

    metrics["liquidity_target_label"] = (
        target.label
        if target is not None
        else None
    )

    if target is None:

        reasons.append(
            "No valid liquidity target above/below entry"
        )

    # --------------------------------------------------------
    # RR
    # --------------------------------------------------------

    rr = None

    if (
        stop_loss is not None
        and target is not None
    ):

        rr = calculate_rr(
            direction=direction,
            entry=entry,
            stop_loss=stop_loss,
            target=target.price,
        )

    metrics["rr_to_liquidity"] = rr

    if rr is None:

        reasons.append(
            "Unable to calculate RR"
        )

    elif rr < MIN_RR:

        reasons.append(
            f"RR is below minimum "
            f"{MIN_RR:.2f}"
        )

    # --------------------------------------------------------
    # INDIVIDUAL SCORES
    # --------------------------------------------------------

    htf_component = score_htf(
        htf_aligned
    )

    liquidity_component = score_liquidity(
        target
    )

    sfp_component = score_sfp(
        sfp
    )

    mss_component = score_mss(
        mss
    )

    displacement_component = score_displacement(
        mss
    )

    volume_component = score_volume(
        volume_ratio
    )

    rr_component = score_rr(
        rr
    )

    metrics["scores"] = {
        "htf": htf_component,
        "liquidity": liquidity_component,
        "sfp": sfp_component,
        "mss": mss_component,
        "displacement": displacement_component,
        "volume": volume_component,
        "rr": rr_component,
    }

    # --------------------------------------------------------
    # FINAL SCORE
    # --------------------------------------------------------

    final_score = calculate_signal_score(
        htf_score=htf_component,
        liquidity_score=liquidity_component,
        sfp_score=sfp_component,
        mss_score=mss_component,
        displacement_score=(
            displacement_component
        ),
        volume_score=volume_component,
        rr_score=rr_component,
    )

    metrics["final_score"] = (
        final_score
    )

    # --------------------------------------------------------
    # SCORE FILTER
    # --------------------------------------------------------

    if final_score < MIN_SIGNAL_SCORE:

        reasons.append(
            f"Signal score "
            f"{final_score:.1f} "
            f"is below minimum "
            f"{MIN_SIGNAL_SCORE:.1f}"
        )

    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    passed = (
        len(reasons) == 0
    )

    return FilterResult(
        passed=passed,
        score=final_score,
        reasons=reasons,
        warnings=warnings,
        metrics=metrics,
    )


# ============================================================
# DEBUG FORMAT
# ============================================================

def filter_result_to_dict(
    result: FilterResult,
) -> dict:
    """
    Преобразует результат фильтрации
    в обычный dict.
    """

    return {
        "passed": result.passed,
        "score": result.score,
        "reasons": result.reasons,
        "warnings": result.warnings,
        "metrics": result.metrics,
    }
