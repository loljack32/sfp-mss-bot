# ============================================================
# SFP + MSS SIGNAL ENGINE
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from config import (
    TP1_R_MULTIPLE,
    TP2_R_MULTIPLE,
    MIN_RR,
    MAX_POSITION_BALANCE_MULTIPLE,
)

from core.structure import (
    SwingPoint,
    StructureState,
    analyze_structure,
)

from core.sfp import (
    SFPSetup,
    find_latest_sfp,
    is_sfp_invalidated,
)

from core.mss import (
    MSSSetup,
    find_mss_after_sfp,
    is_mss_invalidated,
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

    Этот объект уже можно передавать в Telegram.
    """

    symbol: str

    direction: str

    timeframe: str

    signal_timestamp: pd.Timestamp

    entry: float

    stop_loss: float

    tp1: float

    tp2: float

    risk_calculation: RiskCalculation

    sfp: SFPSetup

    mss: MSSSetup

    score: float

    rr_tp1: float

    rr_tp2: float

    htf_trend: str

    structure_label: str

    liquidity_target: Optional[float]

    warnings: tuple[str, ...]


# ============================================================
# HELPERS
# ============================================================

def _get_entry_price(
    df: pd.DataFrame,
    mss: MSSSetup,
) -> float:
    """
    Определяет цену входа.

    Для подтверждённого MSS используем close
    MSS-свечи.

    Это позволяет не отправлять сигнал
    по случайной цене внутри свечи.
    """

    if mss.break_index < 0:

        raise ValueError(
            "Invalid MSS break index."
        )

    if mss.break_index >= len(df):

        raise ValueError(
            "MSS break index is outside DataFrame."
        )

    entry = float(
        df.iloc[
            mss.break_index
        ]["close"]
    )

    if entry <= 0:

        raise ValueError(
            "Entry price must be greater than zero."
        )

    return entry


# ============================================================
# RISK LIMIT
# ============================================================

def _check_position_limit(
    risk: RiskCalculation,
) -> tuple[bool, str]:
    """
    Защита от ненормально большого номинала позиции.

    Например:

        balance = 1000
        max multiple = 10

    Максимальный notional:

        10000
    """

    max_notional = (
        risk.balance
        * MAX_POSITION_BALANCE_MULTIPLE
    )

    if risk.position_notional > max_notional:

        return (
            False,
            (
                "Position notional "
                f"{risk.position_notional:.4f} "
                "exceeds configured maximum "
                f"{MAX_POSITION_BALANCE_MULTIPLE:.2f}x "
                "balance."
            ),
        )

    return True, "Position size is within limits."


# ============================================================
# SIGNAL VALIDATION
# ============================================================

def _validate_signal_prices(
    direction: str,
    entry: float,
    stop_loss: float,
    tp1: float,
    tp2: float,
) -> None:
    """
    Проверяет геометрию LONG/SHORT сигнала.
    """

    if direction == "LONG":

        if not (
            stop_loss < entry
            < tp1
            < tp2
        ):

            raise ValueError(
                "Invalid LONG signal price structure: "
                "expected SL < Entry < TP1 < TP2."
            )

    elif direction == "SHORT":

        if not (
            tp2
            < tp1
            < entry
            < stop_loss
        ):

            raise ValueError(
                "Invalid SHORT signal price structure: "
                "expected TP2 < TP1 < Entry < SL."
            )

    else:

        raise ValueError(
            f"Unknown signal direction: {direction}"
        )


# ============================================================
# BUILD SIGNAL
# ============================================================

def build_signal(
    symbol: str,
    entry_df: pd.DataFrame,
    htf_df: pd.DataFrame,
    balance: float,
    risk_percent: float,
    timeframe: str = "15m",
) -> Optional[TradingSignal]:
    """
    Главная функция создания торгового сигнала.

    Pipeline:

        1. HTF structure
        2. Entry structure
        3. SFP
        4. MSS
        5. Filters
        6. Entry
        7. SL
        8. TP1 / TP2
        9. Position size
        10. Final validation

    Если сетап не проходит хотя бы один
    обязательный фильтр — возвращается None.
    """

    # ========================================================
    # BASIC VALIDATION
    # ========================================================

    if entry_df.empty:

        return None

    if htf_df.empty:

        return None

    if balance <= 0:

        raise ValueError(
            "Balance must be greater than zero."
        )

    if risk_percent <= 0:

        raise ValueError(
            "Risk percent must be greater than zero."
        )

    # ========================================================
    # STRUCTURE
    # ========================================================

    entry_structure, entry_state = (
        analyze_structure(
            entry_df,
            htf=False,
        )
    )

    htf_structure, htf_state = (
        analyze_structure(
            htf_df,
            htf=True,
        )
    )

    # --------------------------------------------------------
    # HTF STRUCTURE MUST BE CLEAR
    # --------------------------------------------------------

    if htf_state.trend not in {
        "BULLISH",
        "BEARISH",
    }:

        return None

    # ========================================================
    # SFP
    # ========================================================

    sfp = find_latest_sfp(
        df=entry_df,
        structure=entry_structure,
    )

    if sfp is None:

        return None

    # ========================================================
    # SFP INVALIDATION
    # ========================================================

    if is_sfp_invalidated(
        df=entry_df,
        sfp=sfp,
    ):

        return None

    # ========================================================
    # MSS
    # ========================================================

    mss = find_mss_after_sfp(
        df=entry_df,
        sfp=sfp,
        structure=entry_structure,
    )

    if mss is None:

        return None

    # ========================================================
    # MSS INVALIDATION
    # ========================================================

    if is_mss_invalidated(
        df=entry_df,
        sfp=sfp,
        mss=mss,
    ):

        return None

    # ========================================================
    # SIGNAL MUST BE FROM LATEST CLOSED DATA
    # ========================================================

    latest_index = (
        len(entry_df) - 1
    )

    if mss.break_index > latest_index:

        return None

    # ========================================================
    # ENTRY
    # ========================================================

    entry = _get_entry_price(
        df=entry_df,
        mss=mss,
    )

    # ========================================================
    # FILTERS
    # ========================================================

    filter_result: FilterResult = (
        evaluate_setup(
            df=entry_df,
            sfp=sfp,
            mss=mss,
            structure=entry_structure,
            htf_state=htf_state,
            entry_price=entry,
        )
    )

    # --------------------------------------------------------
    # FILTER RESULT
    # --------------------------------------------------------

    if not filter_result.passed:

        return None

    # ========================================================
    # EXTRACT SL
    # ========================================================

    stop_loss_value = (
        filter_result.metrics.get(
            "stop_loss"
        )
    )

    if stop_loss_value is None:

        return None

    stop_loss = float(
        stop_loss_value
    )

    # ========================================================
    # STOP VALIDATION
    # ========================================================

    if stop_loss <= 0:

        return None

    # ========================================================
    # RISK CALCULATION
    # ========================================================

    try:

        risk = calculate_risk(
            direction=sfp.direction,
            balance=balance,
            risk_percent=risk_percent,
            entry=entry,
            stop_loss=stop_loss,
            tp1_rr=TP1_R_MULTIPLE,
            tp2_rr=TP2_R_MULTIPLE,
        )

    except ValueError:

        return None

    # ========================================================
    # TP
    # ========================================================

    tp1 = float(
        risk.tp1
    )

    tp2 = float(
        risk.tp2
    )

    # ========================================================
    # PRICE STRUCTURE VALIDATION
    # ========================================================

    try:

        _validate_signal_prices(
            direction=sfp.direction,
            entry=entry,
            stop_loss=stop_loss,
            tp1=tp1,
            tp2=tp2,
        )

    except ValueError:

        return None

    # ========================================================
    # RR VALIDATION
    # ========================================================

    if risk.rr_tp2 < MIN_RR:

        return None

    # ========================================================
    # POSITION LIMIT
    # ========================================================

    position_ok, _ = (
        _check_position_limit(
            risk
        )
    )

    if not position_ok:

        return None

    # ========================================================
    # LIQUIDITY TARGET
    # ========================================================

    liquidity_target = (
        filter_result.metrics.get(
            "liquidity_target"
        )
    )

    if liquidity_target is not None:

        liquidity_target = float(
            liquidity_target
        )

    # ========================================================
    # SIGNAL TIMESTAMP
    # ========================================================

    signal_timestamp = (
        entry_df.iloc[
            mss.break_index
        ]["timestamp"]
    )

    # ========================================================
    # WARNINGS
    # ========================================================

    warnings = tuple(
        filter_result.warnings
    )

    # ========================================================
    # BUILD
    # ========================================================

    return TradingSignal(
        symbol=symbol,

        direction=sfp.direction,

        timeframe=timeframe,

        signal_timestamp=signal_timestamp,

        entry=entry,

        stop_loss=stop_loss,

        tp1=tp1,

        tp2=tp2,

        risk_calculation=risk,

        sfp=sfp,

        mss=mss,

        score=float(
            filter_result.score
        ),

        rr_tp1=float(
            risk.rr_tp1
        ),

        rr_tp2=float(
            risk.rr_tp2
        ),

        htf_trend=htf_state.trend,

        structure_label=(
            mss.structure_label
        ),

        liquidity_target=(
            liquidity_target
        ),

        warnings=warnings,
    )


# ============================================================
# SERIALIZATION
# ============================================================

def signal_to_dict(
    signal: TradingSignal,
) -> dict:
    """
    Преобразует сигнал в JSON-compatible dict.
    """

    risk = (
        signal.risk_calculation
    )

    return {
        "symbol": signal.symbol,

        "direction": signal.direction,

        "timeframe": signal.timeframe,

        "signal_timestamp": str(
            signal.signal_timestamp
        ),

        "entry": signal.entry,

        "stop_loss": signal.stop_loss,

        "tp1": signal.tp1,

        "tp2": signal.tp2,

        "balance": risk.balance,

        "risk_percent": (
            risk.risk_percent
        ),

        "risk_amount": (
            risk.risk_amount
        ),

        "position_size": (
            risk.position_size
        ),

        "position_notional": (
            risk.position_notional
        ),

        "stop_distance": (
            risk.stop_distance
        ),

        "stop_distance_percent": (
            risk.stop_distance_percent
        ),

        "rr_tp1": signal.rr_tp1,

        "rr_tp2": signal.rr_tp2,

        "score": signal.score,

        "htf_trend": signal.htf_trend,

        "structure_label": (
            signal.structure_label
        ),

        "liquidity_target": (
            signal.liquidity_target
        ),

        "sfp": {
            "direction": (
                signal.sfp.direction
            ),

            "level": (
                signal.sfp.level
            ),

            "sweep_extreme": (
                signal.sfp.sweep_extreme
            ),

            "sweep_index": (
                signal.sfp.sweep_index
            ),

            "sweep_timestamp": str(
                signal.sfp.sweep_timestamp
            ),

            "sweep_atr_ratio": (
                signal.sfp.sweep_atr_ratio
            ),

            "reclaim_ratio": (
                signal.sfp.reclaim_ratio
            ),

            "score": (
                signal.sfp.score
            ),
        },

        "mss": {
            "direction": (
                signal.mss.direction
            ),

            "structure_level": (
                signal.mss.structure_level
            ),

            "structure_label": (
                signal.mss.structure_label
            ),

            "break_index": (
                signal.mss.break_index
            ),

            "break_timestamp": str(
                signal.mss.break_timestamp
            ),

            "break_price": (
                signal.mss.break_price
            ),

            "displacement": (
                signal.mss.displacement
            ),

            "displacement_atr_ratio": (
                signal.mss.displacement_atr_ratio
            ),

            "body_ratio": (
                signal.mss.body_ratio
            ),

            "score": (
                signal.mss.score
            ),
        },

        "warnings": list(
            signal.warnings
        ),
    }


# ============================================================
# TELEGRAM SUMMARY
# ============================================================

def signal_summary(
    signal: TradingSignal,
) -> str:
    """
    Формирует краткое представление сигнала.

    Основное Telegram-сообщение будет строиться
    отдельно в telegram.py.
    """

    direction = (
        "LONG"
        if signal.direction == "LONG"
        else "SHORT"
    )

    return (
        f"{signal.symbol} "
        f"{direction} "
        f"SFP+MSS | "
        f"Score {signal.score:.1f} | "
        f"RR 1:{signal.rr_tp2:.2f}"
    )
