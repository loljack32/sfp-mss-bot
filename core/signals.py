# ============================================================
# SIGNAL ENGINE
# SFP + MSS TRADING SIGNAL GENERATOR
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from config import (
    DEFAULT_RISK_PERCENT,
    MAX_POSITION_BALANCE_MULTIPLE,
    MIN_RISK_PERCENT,
    MAX_RISK_PERCENT,
    TP1_R_MULTIPLE,
    TP2_R_MULTIPLE,
)

from core.structure import (
    SwingPoint,
    StructureState,
)

from core.sfp import (
    SFPSetup,
)

from core.mss import (
    MSSSetup,
)

from core.filters import (
    FilterResult,
    evaluate_setup,
)

from core.risk import (
    RiskCalculation,
    calculate_risk,
)


# ============================================================
# DATA CLASS
# ============================================================

@dataclass(frozen=True)
class TradingSignal:
    """
    Полностью сформированный торговый сигнал.

    Signal создаётся только после прохождения
    SFP + MSS + всех обязательных filters + risk validation.
    """

    symbol: str

    direction: str

    timeframe: str

    timestamp: pd.Timestamp

    entry: float

    stop_loss: float

    tp1: float

    tp2: float

    rr_tp1: float

    rr_tp2: float

    risk_percent: float

    risk_amount: float

    position_size: float

    position_notional: float

    stop_distance: float

    stop_distance_percent: float

    signal_score: float

    sfp_score: float

    mss_score: float

    htf_trend: str

    liquidity_target: Optional[float]

    liquidity_target_label: Optional[str]

    sfp: SFPSetup

    mss: MSSSetup

    filters: FilterResult


# ============================================================
# RISK VALIDATION
# ============================================================

def validate_risk_percent_for_signal(
    risk_percent: float,
) -> None:
    """
    Проверяет риск относительно настроек бота.

    В отличие от validate_risk_percent() из risk.py,
    здесь используется диапазон, разрешённый конфигурацией бота.
    """

    if risk_percent < MIN_RISK_PERCENT:

        raise ValueError(
            f"Risk percent {risk_percent:.2f}% "
            f"is below configured minimum "
            f"{MIN_RISK_PERCENT:.2f}%."
        )

    if risk_percent > MAX_RISK_PERCENT:

        raise ValueError(
            f"Risk percent {risk_percent:.2f}% "
            f"exceeds configured maximum "
            f"{MAX_RISK_PERCENT:.2f}%."
        )


# ============================================================
# POSITION SIZE VALIDATION
# ============================================================

def validate_position_notional(
    balance: float,
    position_notional: float,
) -> None:
    """
    Защита от аномально большой позиции.

    Например:

        balance = 1000
        MAX_POSITION_BALANCE_MULTIPLE = 10

    Максимальный notional:

        10 000
    """

    if balance <= 0:

        raise ValueError(
            "Balance must be greater than zero."
        )

    if position_notional <= 0:

        raise ValueError(
            "Position notional must be greater than zero."
        )

    maximum_notional = (
        balance
        * MAX_POSITION_BALANCE_MULTIPLE
    )

    if position_notional > maximum_notional:

        raise ValueError(
            "Position notional exceeds configured "
            f"maximum of {MAX_POSITION_BALANCE_MULTIPLE:.2f}x "
            "account balance."
        )


# ============================================================
# SIGNAL DIRECTION VALIDATION
# ============================================================

def validate_signal_components(
    sfp: SFPSetup,
    mss: MSSSetup,
) -> None:
    """
    Проверяет согласованность SFP и MSS.

    LONG:

        SFP LONG
        MSS LONG

    SHORT:

        SFP SHORT
        MSS SHORT
    """

    if sfp.direction not in {
        "LONG",
        "SHORT",
    }:

        raise ValueError(
            f"Invalid SFP direction: {sfp.direction}"
        )

    if mss.direction not in {
        "LONG",
        "SHORT",
    }:

        raise ValueError(
            f"Invalid MSS direction: {mss.direction}"
        )

    if sfp.direction != mss.direction:

        raise ValueError(
            "SFP and MSS directions do not match."
        )

    if mss.break_index <= sfp.sweep_index:

        raise ValueError(
            "MSS must occur after SFP."
        )


# ============================================================
# BUILD SIGNAL
# ============================================================

def build_signal(
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    sfp: SFPSetup,
    mss: MSSSetup,
    structure: List[SwingPoint],
    htf_state: StructureState,
    balance: float,
    risk_percent: float = DEFAULT_RISK_PERCENT,
    entry_price: Optional[float] = None,
) -> TradingSignal:
    """
    Создаёт готовый торговый сигнал.

    Pipeline:

        SFP
          ↓
        MSS
          ↓
        Filters
          ↓
        SL
          ↓
        TP
          ↓
        Position size
          ↓
        TradingSignal

    Если любой обязательный этап не проходит,
    выбрасывается ValueError.
    """

    # --------------------------------------------------------
    # BASIC VALIDATION
    # --------------------------------------------------------

    if df.empty:

        raise ValueError(
            "Cannot build signal from empty DataFrame."
        )

    if not symbol:

        raise ValueError(
            "Symbol must not be empty."
        )

    if not timeframe:

        raise ValueError(
            "Timeframe must not be empty."
        )

    validate_signal_components(
        sfp=sfp,
        mss=mss,
    )

    validate_risk_percent_for_signal(
        risk_percent=risk_percent,
    )

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    filter_result = evaluate_setup(
        df=df,
        sfp=sfp,
        mss=mss,
        structure=structure,
        htf_state=htf_state,
        entry_price=entry_price,
    )

    if not filter_result.passed:

        reasons = "; ".join(
            filter_result.reasons
        )

        raise ValueError(
            "Signal failed filters: "
            f"{reasons}"
        )

    # --------------------------------------------------------
    # ENTRY
    # --------------------------------------------------------

    if entry_price is None:

        entry = float(
            df.iloc[-1]["close"]
        )

    else:

        entry = float(
            entry_price
        )

    if entry <= 0:

        raise ValueError(
            "Signal entry must be greater than zero."
        )

    # --------------------------------------------------------
    # STOP LOSS
    # --------------------------------------------------------

    stop_loss = filter_result.metrics.get(
        "stop_loss"
    )

    if stop_loss is None:

        raise ValueError(
            "Filter result does not contain a valid stop loss."
        )

    stop_loss = float(
        stop_loss
    )

    # --------------------------------------------------------
    # RISK CALCULATION
    # --------------------------------------------------------

    risk = calculate_risk(
        direction=sfp.direction,
        balance=balance,
        risk_percent=risk_percent,
        entry=entry,
        stop_loss=stop_loss,
        tp1_rr=TP1_R_MULTIPLE,
        tp2_rr=TP2_R_MULTIPLE,
    )

    # --------------------------------------------------------
    # POSITION LIMIT
    # --------------------------------------------------------

    validate_position_notional(
        balance=balance,
        position_notional=risk.position_notional,
    )

    # --------------------------------------------------------
    # LIQUIDITY TARGET
    # --------------------------------------------------------

    liquidity_target = (
        filter_result.metrics.get(
            "liquidity_target"
        )
    )

    if liquidity_target is not None:

        liquidity_target = float(
            liquidity_target
        )

    liquidity_target_label = (
        filter_result.metrics.get(
            "liquidity_target_label"
        )
    )

    # --------------------------------------------------------
    # SIGNAL TIMESTAMP
    # --------------------------------------------------------

    timestamp = df.iloc[-1]["timestamp"]

    if not isinstance(
        timestamp,
        pd.Timestamp,
    ):

        timestamp = pd.Timestamp(
            timestamp
        )

    # --------------------------------------------------------
    # FINAL SIGNAL
    # --------------------------------------------------------

    return TradingSignal(
        symbol=symbol,
        direction=sfp.direction,
        timeframe=timeframe,
        timestamp=timestamp,

        entry=entry,

        stop_loss=risk.stop_loss,

        tp1=risk.tp1,
        tp2=risk.tp2,

        rr_tp1=risk.rr_tp1,
        rr_tp2=risk.rr_tp2,

        risk_percent=risk.risk_percent,
        risk_amount=risk.risk_amount,

        position_size=risk.position_size,
        position_notional=risk.position_notional,

        stop_distance=risk.stop_distance,
        stop_distance_percent=(
            risk.stop_distance_percent
        ),

        signal_score=filter_result.score,

        sfp_score=sfp.score,
        mss_score=mss.score,

        htf_trend=htf_state.trend,

        liquidity_target=liquidity_target,
        liquidity_target_label=(
            liquidity_target_label
        ),

        sfp=sfp,
        mss=mss,
        filters=filter_result,
    )


# ============================================================
# SIGNAL ENGINE
# ============================================================

def generate_signal(
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    sfp: Optional[SFPSetup],
    mss: Optional[MSSSetup],
    structure: List[SwingPoint],
    htf_state: StructureState,
    balance: float,
    risk_percent: float = DEFAULT_RISK_PERCENT,
    entry_price: Optional[float] = None,
) -> Optional[TradingSignal]:
    """
    Безопасная обёртка над build_signal().

    В отличие от build_signal():

        отсутствие SFP/MSS
        или провал фильтров

    не является исключением для основного scanner loop.

    Возвращается:

        TradingSignal
            если сигнал полностью прошёл проверки.

        None
            если торгового сигнала нет.
    """

    if sfp is None:

        return None

    if mss is None:

        return None

    try:

        return build_signal(
            symbol=symbol,
            timeframe=timeframe,
            df=df,
            sfp=sfp,
            mss=mss,
            structure=structure,
            htf_state=htf_state,
            balance=balance,
            risk_percent=risk_percent,
            entry_price=entry_price,
        )

    except ValueError:

        return None


# ============================================================
# SIGNAL SUMMARY
# ============================================================

def signal_to_dict(
    signal: TradingSignal,
) -> dict:
    """
    Преобразует TradingSignal в обычный dict.

    Используется Telegram, JSON и логированием.
    """

    return {
        "symbol": signal.symbol,
        "direction": signal.direction,
        "timeframe": signal.timeframe,
        "timestamp": str(
            signal.timestamp
        ),

        "entry": signal.entry,
        "stop_loss": signal.stop_loss,

        "tp1": signal.tp1,
        "tp2": signal.tp2,

        "rr_tp1": signal.rr_tp1,
        "rr_tp2": signal.rr_tp2,

        "risk_percent": signal.risk_percent,
        "risk_amount": signal.risk_amount,

        "position_size": signal.position_size,
        "position_notional": signal.position_notional,

        "stop_distance": signal.stop_distance,
        "stop_distance_percent": (
            signal.stop_distance_percent
        ),

        "signal_score": signal.signal_score,

        "sfp_score": signal.sfp_score,
        "mss_score": signal.mss_score,

        "htf_trend": signal.htf_trend,

        "liquidity_target": (
            signal.liquidity_target
        ),

        "liquidity_target_label": (
            signal.liquidity_target_label
        ),
    }


# ============================================================
# SIGNAL DESCRIPTION
# ============================================================

def signal_to_text(
    signal: TradingSignal,
) -> str:
    """
    Формирует компактное текстовое представление сигнала.

    Telegram-форматирование будет находиться
    в core/telegram.py.
    """

    direction_icon = (
        "🟢"
        if signal.direction == "LONG"
        else "🔴"
    )

    lines = [
        f"{direction_icon} {signal.symbol} "
        f"{signal.direction}",

        f"TF: {signal.timeframe}",

        f"Score: {signal.signal_score:.1f}",

        f"Entry: {signal.entry:.8f}",
        f"SL: {signal.stop_loss:.8f}",

        f"TP1: {signal.tp1:.8f} "
        f"(1R / RR {signal.rr_tp1:.2f})",

        f"TP2: {signal.tp2:.8f} "
        f"(2R / RR {signal.rr_tp2:.2f})",

        f"Risk: {signal.risk_percent:.2f}% "
        f"({signal.risk_amount:.2f})",

        f"Position: {signal.position_size:.8f}",
        f"Notional: {signal.position_notional:.2f}",

        f"SFP score: {signal.sfp_score:.1f}",
        f"MSS score: {signal.mss_score:.1f}",

        f"HTF: {signal.htf_trend}",
    ]

    if signal.liquidity_target is not None:

        target_text = (
            f"Liquidity: "
            f"{signal.liquidity_target:.8f}"
        )

        if signal.liquidity_target_label:

            target_text += (
                f" ({signal.liquidity_target_label})"
            )

        lines.append(
            target_text
        )

    return "\n".join(
        lines
    )


# ============================================================
# FILTER DIAGNOSTICS
# ============================================================

def get_signal_failure_reasons(
    df: pd.DataFrame,
    sfp: Optional[SFPSetup],
    mss: Optional[MSSSetup],
    structure: List[SwingPoint],
    htf_state: StructureState,
    entry_price: Optional[float] = None,
) -> List[str]:
    """
    Возвращает причины, по которым сетап не может
    стать торговым сигналом.

    Эта функция предназначена прежде всего для debug/logging.

    В отличие от generate_signal(), она НЕ скрывает
    причины отказа.
    """

    reasons: List[str] = []

    if sfp is None:

        reasons.append(
            "No valid SFP"
        )

        return reasons

    if mss is None:

        reasons.append(
            "No valid MSS after SFP"
        )

        return reasons

    try:

        validate_signal_components(
            sfp=sfp,
            mss=mss,
        )

    except ValueError as exc:

        reasons.append(
            str(exc)
        )

        return reasons

    result = evaluate_setup(
        df=df,
        sfp=sfp,
        mss=mss,
        structure=structure,
        htf_state=htf_state,
        entry_price=entry_price,
    )

    reasons.extend(
        result.reasons
    )

    return reasons


# ============================================================
# RISK PREVIEW
# ============================================================

def preview_risk(
    direction: str,
    balance: float,
    risk_percent: float,
    entry: float,
    stop_loss: float,
) -> RiskCalculation:
    """
    Отдельный preview risk calculation.

    Не создаёт торговый сигнал.

    Полезно для Telegram-команды /risk
    и будущего пользовательского интерфейса.
    """

    validate_risk_percent_for_signal(
        risk_percent=risk_percent,
    )

    result = calculate_risk(
        direction=direction,
        balance=balance,
        risk_percent=risk_percent,
        entry=entry,
        stop_loss=stop_loss,
        tp1_rr=TP1_R_MULTIPLE,
        tp2_rr=TP2_R_MULTIPLE,
    )

    validate_position_notional(
        balance=balance,
        position_notional=result.position_notional,
    )

    return result
