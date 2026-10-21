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
    REQUIRE_VOLUME_CONFIRMATION,
    MIN_SIGNAL_SCORE,
    MIN_SFP_SCORE,
    MIN_MSS_SCORE,
    COUNTER_TREND_MIN_SIGNAL_SCORE,
    MAX_SIGNAL_AGE_BARS,
    MAX_ENTRY_DISTANCE_ATR,
    MAX_STOP_DISTANCE_ATR,
    MAX_STOP_DISTANCE_PERCENT,
    LIQUIDITY_SWING_LEFT,
    LIQUIDITY_SWING_RIGHT,
    MIN_RR,
    TP1_R_MULTIPLE,
    TP2_R_MULTIPLE,
    MIN_TP2_R_MULTIPLE,
    SIGNAL_SCORE_WEIGHTS,
    ENABLE_COUNTER_TREND_SIGNALS,
    ENABLE_SHORT_SIGNALS,
    EXTENSION_LOOKBACK_BARS,
    MAX_ENTRY_EXTENSION_ATR,
    ENABLE_HTF_DIST_FILTER,
    ENABLE_REGIME_FILTER,
    REGIME_EMA_PERIOD,
    HTF_EMA_PERIOD,
    HTF_DIST_MIN_ATR,
    HTF_DIST_MAX_ATR,
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
    build_structure,
)


# ============================================================
# DATA CLASS
# ============================================================

@dataclass(frozen=True)
class FilterResult:
    """
    Результат проверки качества сетапа.
    """

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


def compute_htf_ema_dist_atr(htf_df: pd.DataFrame) -> Optional[float]:
    """
    Дистанция цены старшего ТФ от её EMA: (close - EMA) / ATR.

    htf_df должен содержать только закрытые свечи; берётся
    последняя. None, если данных не хватает или ATR невалиден.
    """
    if htf_df is None or len(htf_df) < HTF_EMA_PERIOD + ATR_PERIOD:
        return None

    closes = htf_df["close"]
    ema_value = closes.ewm(span=HTF_EMA_PERIOD, adjust=False).mean().iloc[-1]
    atr_value = calculate_atr(htf_df, period=ATR_PERIOD).iloc[-1]

    if pd.isna(ema_value) or pd.isna(atr_value) or float(atr_value) <= 0:
        return None

    return float((float(closes.iloc[-1]) - float(ema_value)) / float(atr_value))


def compute_regime_uptrend(
    regime_df: pd.DataFrame, period: int = REGIME_EMA_PERIOD
) -> Optional[bool]:
    """
    Режим широкого рынка по бенчмарку: True, если последняя
    закрытая свеча бенчмарка выше своей EMA(period). None, если
    данных не хватает.
    """
    if regime_df is None or len(regime_df) < period:
        return None

    closes = regime_df["close"]
    ema_value = closes.ewm(span=period, adjust=False).mean().iloc[-1]
    if pd.isna(ema_value):
        return None

    return bool(float(closes.iloc[-1]) > float(ema_value))


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
    # Market structure shift - это слом структуры, которая вела
    # цену к свипу: для лонга пробой LH, для шорта пробой HL.
    # Пробой HH/LL - продолжение уже идущего тренда, а не сдвиг,
    # такие сетапы статистически слабее и отсекаются.
    if direction == "LONG":
        if mss.structure_label == "LH":
            return True, "Bullish MSS broke LH"
        return False, "Bullish MSS did not break an LH structure level"

    if direction == "SHORT":
        if mss.structure_label == "HL":
            return True, "Bearish MSS broke HL"
        return False, "Bearish MSS did not break an HL structure level"

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
    stop_loss: Optional[float] = None,
    min_rr: float = MIN_RR,
) -> Optional[SwingPoint]:
    targets = get_liquidity_targets(
        direction=direction,
        current_index=current_index,
        structure=structure,
        current_price=current_price,
    )

    if not targets:
        return None

    # Without a valid stop we cannot determine whether a liquidity
    # level provides an acceptable risk/reward ratio.
    if stop_loss is None:
        return targets[0]

    # The liquidity target must not sit before TP1.
    # TP1 is currently 2R, while MIN_RR is 1.60R.
    # Therefore the effective minimum target RR is the larger
    # of MIN_RR and TP1_R_MULTIPLE.
    required_rr = max(min_rr, TP1_R_MULTIPLE)

    # Выбор цели: первый уровень с RR не ниже требуемого.
    # Функция вызывается по КРУПНЫМ свингам (см. evaluate_setup):
    # уровни ближе required_rr для геометрии этого сетапа (стоп за
    # экстремумом свипа, вход по закрытию MSS) целью быть не могут -
    # до них нет сделки с приемлемым RR. Но и перескакивать через
    # крупный уровень, который уже даёт нужный RR, нельзя: именно
    # он является целью, а не следующий за ним.
    for target in targets:
        rr = calculate_rr(
            direction=direction,
            entry=current_price,
            stop_loss=stop_loss,
            target=target.price,
        )

        if rr is not None and rr >= required_rr:
            return target

    return None


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

    if distance_atr <= MAX_ENTRY_DISTANCE_ATR:
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
    htf_dist_atr: Optional[float] = None,
    regime_uptrend: Optional[bool] = None,
    *args,
    **kwargs,
) -> FilterResult:
    reasons: List[str] = []
    warnings: List[str] = []
    metrics = {}

    direction = sfp.direction

    if direction == "SHORT" and not ENABLE_SHORT_SIGNALS:
        reasons.append("Short signals disabled by config")

    # --------------------------------------------------------
    # ANTI-CHASE
    # --------------------------------------------------------
    # Если цена уже прошла далеко в сторону сделки за последние
    # EXTENSION_LOOKBACK_BARS баров, свип + MSS происходят на
    # истощении движения, а не в его начале.
    if len(df) > EXTENSION_LOOKBACK_BARS and sfp.atr and float(sfp.atr) > 0:
        closes = df["close"]
        extension = (
            float(closes.iloc[-1])
            - float(closes.iloc[-1 - EXTENSION_LOOKBACK_BARS])
        ) / float(sfp.atr)
        if direction == "SHORT":
            extension = -extension
        metrics["entry_extension_atr"] = round(extension, 3)
        if extension > MAX_ENTRY_EXTENSION_ATR:
            reasons.append(
                f"Entry chase: price already moved {extension:.2f} ATR "
                f"in signal direction over {EXTENSION_LOOKBACK_BARS} bars "
                f"(max {MAX_ENTRY_EXTENSION_ATR})"
            )

    # --------------------------------------------------------
    # HTF DISTANCE WINDOW
    # --------------------------------------------------------
    # Вход только когда цена старшего ТФ рядом со своей EMA:
    # движение началось, но не выдохлось. htf_dist_atr считает
    # вызывающий код по закрытым свечам HTF; None - фильтр
    # пропускается (диагностические вызовы).
    if ENABLE_HTF_DIST_FILTER and htf_dist_atr is not None:
        aligned_dist = htf_dist_atr if direction == "LONG" else -htf_dist_atr
        metrics["htf_dist_atr"] = round(float(aligned_dist), 3)
        if aligned_dist < HTF_DIST_MIN_ATR or aligned_dist > HTF_DIST_MAX_ATR:
            reasons.append(
                f"HTF distance {aligned_dist:.2f} ATR outside entry window "
                f"[{HTF_DIST_MIN_ATR}, {HTF_DIST_MAX_ATR}]"
            )

    # --------------------------------------------------------
    # MARKET REGIME
    # --------------------------------------------------------
    # Лонг - только когда бенчмарк в восходящем режиме (выше
    # своей EMA), шорт - только в нисходящем. regime_uptrend
    # считает вызывающий код по закрытым свечам бенчмарка;
    # None - фильтр пропускается.
    if ENABLE_REGIME_FILTER and regime_uptrend is not None:
        metrics["regime_uptrend"] = bool(regime_uptrend)
        if direction == "LONG" and not regime_uptrend:
            reasons.append(
                "Market regime filter: benchmark below its EMA "
                "(downtrend), long blocked"
            )
        if direction == "SHORT" and regime_uptrend:
            reasons.append(
                "Market regime filter: benchmark above its EMA "
                "(uptrend), short blocked"
            )

    if entry_price is None:
        entry_price = float(df.iloc[-1]["close"])
    entry = float(entry_price)
    metrics["entry"] = entry

    # --------------------------------------------------------
    # FRESHNESS
    # --------------------------------------------------------
    # Сетап живёт только сразу после подтверждения MSS. Если с
    # пробоя прошло больше MAX_SIGNAL_AGE_BARS свечей, цена уже
    # отыграла движение и "сигнал" превращается в погоню за ценой.
    last_index = len(df) - 1
    bars_since_mss = last_index - mss.break_index
    metrics["bars_since_mss"] = bars_since_mss

    if bars_since_mss < 0:
        reasons.append("MSS break is in the future relative to data")

    elif bars_since_mss > MAX_SIGNAL_AGE_BARS:
        reasons.append(
            f"Stale setup: MSS confirmed {bars_since_mss} bars ago "
            f"(max {MAX_SIGNAL_AGE_BARS})"
        )

    # --------------------------------------------------------
    # PRICE STILL VALID
    # --------------------------------------------------------
    # К моменту сигнала цена должна оставаться по нужную сторону
    # и уровня MSS, и снятого уровня ликвидности. Возврат обратно
    # означает, что пробой не удержался.
    if direction == "LONG":
        if entry <= mss.structure_level:
            reasons.append("Price fell back below the MSS level")
        if entry <= sfp.level:
            reasons.append("Price fell back below the swept level")

    elif direction == "SHORT":
        if entry >= mss.structure_level:
            reasons.append("Price rose back above the MSS level")
        if entry >= sfp.level:
            reasons.append("Price rose back above the swept level")

    htf_aligned, htf_reason, setup_type = check_htf_alignment(
        direction=direction,
        htf_state=htf_state,
        allow_counter_trend=allow_counter_trend,
    )

    metrics["htf_trend"] = htf_state.trend
    metrics["htf_aligned"] = htf_aligned
    metrics["setup_type"] = setup_type

    if not htf_aligned:
        reasons.append(htf_reason)

    local_structure_ok, structure_reason = check_local_structure(
        direction=direction,
        sfp=sfp,
        mss=mss,
        structure=structure,
    )
    metrics["local_structure_ok"] = local_structure_ok
    if not local_structure_ok:
        reasons.append(structure_reason)

    atr = float(sfp.atr)
    atr_percent = atr / entry if entry > 0 else 0
    metrics["atr"] = atr
    metrics["atr_percent"] = atr_percent

    if atr_percent < MIN_ATR_PERCENT:
        reasons.append("ATR volatility is too low")

    volume_ratio = calculate_volume_ratio(df=df, index=mss.break_index)
    metrics["volume_ratio"] = volume_ratio
    if REQUIRE_VOLUME_CONFIRMATION:
        # Объём - обязательное подтверждение пробоя, а не warning.
        # Пробой на объёме ниже среднего чаще всего фейковый.
        if volume_ratio is None:
            reasons.append("Volume confirmation is unavailable")
        elif volume_ratio < MIN_VOLUME_RATIO:
            reasons.append(
                f"Volume confirmation too weak: {volume_ratio:.2f} "
                f"< {MIN_VOLUME_RATIO:.2f}"
            )
    elif volume_ratio is None or volume_ratio < MIN_VOLUME_RATIO:
        warnings.append("Volume confirmation is weak")

    # --------------------------------------------------------
    # COMPONENT QUALITY GATES
    # --------------------------------------------------------
    # Слабый SFP или слабый MSS нельзя компенсировать другими
    # компонентами итогового score.
    metrics["sfp_score"] = float(sfp.score)
    metrics["mss_score"] = float(mss.score)

    if sfp.score < MIN_SFP_SCORE:
        reasons.append(
            f"SFP quality too low: {sfp.score:.1f} < {MIN_SFP_SCORE:.1f}"
        )

    if mss.score < MIN_MSS_SCORE:
        reasons.append(
            f"MSS quality too low: {mss.score:.1f} < {MIN_MSS_SCORE:.1f}"
        )

    entry_distance_ok, entry_distance_reason = check_entry_distance_from_mss(
        direction=direction,
        entry=entry,
        mss=mss,
        atr=atr,
    )
    metrics["entry_distance_ok"] = entry_distance_ok
    if not entry_distance_ok:
        reasons.append(entry_distance_reason)

    stop_loss = calculate_stop_loss(direction=direction, sfp=sfp, atr=atr)
    metrics["stop_loss"] = stop_loss

    if stop_loss is None:
        reasons.append("Unable to calculate stop loss")
    else:
        if direction == "LONG" and stop_loss >= entry:
            reasons.append("LONG stop loss is not below entry")
        elif direction == "SHORT" and stop_loss <= entry:
            reasons.append("SHORT stop loss is not above entry")

        # ----------------------------------------------------
        # STOP DISTANCE SANITY
        # ----------------------------------------------------
        # Слишком широкий стоп означает, что вход уже далеко от
        # точки инвалидации: сетап растянут, а расчёт позиции и
        # ликвидации по такому стопу недостоверен.
        risk_distance = abs(entry - stop_loss)
        metrics["stop_distance"] = risk_distance

        if atr > 0:
            stop_distance_atr = risk_distance / atr
            metrics["stop_distance_atr"] = stop_distance_atr
            if stop_distance_atr > MAX_STOP_DISTANCE_ATR:
                reasons.append(
                    f"Stop distance too wide: {stop_distance_atr:.2f} ATR "
                    f"> {MAX_STOP_DISTANCE_ATR:.2f} ATR"
                )

        if entry > 0:
            stop_distance_percent = risk_distance / entry * 100.0
            metrics["stop_distance_percent"] = stop_distance_percent
            if stop_distance_percent > MAX_STOP_DISTANCE_PERCENT:
                reasons.append(
                    f"Stop distance too wide: {stop_distance_percent:.2f}% "
                    f"> {MAX_STOP_DISTANCE_PERCENT:.2f}%"
                )

        # ----------------------------------------------------
        # ALREADY STOPPED AFTER MSS
        # ----------------------------------------------------
        # Если после пробоя цена уже касалась стопа фитилём,
        # сделка по этому сетапу уже была бы закрыта в минус.
        post_mss = df.iloc[mss.break_index + 1:]
        if not post_mss.empty:
            if direction == "LONG" and bool(
                (post_mss["low"] <= stop_loss).any()
            ):
                reasons.append("Price already touched stop loss after MSS")
            elif direction == "SHORT" and bool(
                (post_mss["high"] >= stop_loss).any()
            ):
                reasons.append("Price already touched stop loss after MSS")

    # Цели ищем по крупным свингам (LIQUIDITY_SWING_*), а не по
    # рабочему фракталу структуры: мелкие свинги 3/3 стоят каждые
    # несколько свечей, ликвидность скапливается не за ними.
    liquidity_structure = build_structure(
        df,
        left=LIQUIDITY_SWING_LEFT,
        right=LIQUIDITY_SWING_RIGHT,
    )

    target = find_nearest_liquidity_target(
        direction=direction,
        current_index=last_index,
        current_price=entry,
        structure=liquidity_structure,
        stop_loss=stop_loss,
        min_rr=MIN_RR,
    )

    metrics["liquidity_target"] = target.price if target is not None else None
    metrics["liquidity_target_label"] = target.label if target is not None else None

    if target is None:
        reasons.append("No valid liquidity target above/below entry")

    rr = None
    if stop_loss is not None and target is not None:
        rr = calculate_rr(
            direction=direction,
            entry=entry,
            stop_loss=stop_loss,
            target=target.price,
        )

    metrics["rr_to_liquidity"] = rr

    if rr is None:
        reasons.append("Unable to calculate RR")
    elif rr < MIN_RR:
        reasons.append(f"RR is below minimum {MIN_RR:.2f}")

    # TP1 ставится на 0.5R перед целью-пулом (фронтраннинг),
    # TP2 - на сам пул, но не дальше полных TP2_R_MULTIPLE.
    if rr is not None and rr >= MIN_TP2_R_MULTIPLE:
        metrics["tp1_r_multiple"] = float(min(TP1_R_MULTIPLE, rr - 0.5))
        metrics["tp2_r_multiple"] = float(min(TP2_R_MULTIPLE, rr))
    else:
        metrics["tp1_r_multiple"] = float(TP1_R_MULTIPLE)
        metrics["tp2_r_multiple"] = float(TP2_R_MULTIPLE)

    htf_component = score_htf(setup_type)
    liquidity_component = score_liquidity(target)
    sfp_component = score_sfp(sfp)
    mss_component = score_mss(mss)
    displacement_component = score_displacement(mss)
    volume_component = score_volume(volume_ratio)
    rr_component = score_rr(rr)

    final_score = calculate_signal_score(
        htf_score=htf_component,
        liquidity_score=liquidity_component,
        sfp_score=sfp_component,
        mss_score=mss_component,
        displacement_score=displacement_component,
        volume_score=volume_component,
        rr_score=rr_component,
    )

    metrics["final_score"] = final_score

    # Для контртренда (если включён) порог выше: откат против 4H
    # тренда должен быть заметно сильнее обычного сетапа.
    required_score = (
        COUNTER_TREND_MIN_SIGNAL_SCORE
        if setup_type == "COUNTER_TREND"
        else MIN_SIGNAL_SCORE
    )
    metrics["required_score"] = required_score

    if final_score < required_score:
        reasons.append(f"Signal score {final_score:.1f} is below minimum {required_score:.1f}")

    passed = len(reasons) == 0

    return FilterResult(
        passed=passed,
        score=final_score,
        reasons=reasons,
        warnings=warnings,
        metrics=metrics,
        setup_type=setup_type,
    )


# ============================================================
# SERIALIZATION
# ============================================================

def filter_result_to_dict(
    result: FilterResult,
) -> dict:
    """
    Преобразует результат фильтрации в dict.
    """
    return {
        "passed": result.passed,
        "score": result.score,
        "reasons": result.reasons,
        "warnings": result.warnings,
        "metrics": result.metrics,
        "setup_type": getattr(result, "setup_type", "TREND"),
    }
