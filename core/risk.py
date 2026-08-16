# ============================================================
# RISK MANAGEMENT
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ============================================================
# DATA CLASS
# ============================================================

@dataclass(frozen=True)
class RiskCalculation:
    """
    Полный результат расчёта торговой позиции.

    Все значения относятся к одной конкретной сделке.
    """

    balance: float

    risk_percent: float
    risk_amount: float

    entry: float
    stop_loss: float

    stop_distance: float
    stop_distance_percent: float

    position_size: float
    position_notional: float

    tp1: float
    tp2: float

    rr_tp1: float
    rr_tp2: float


# ============================================================
# VALIDATION
# ============================================================

def validate_balance(
    balance: float,
) -> None:
    """
    Проверяет баланс пользователя.
    """

    if balance <= 0:
        raise ValueError(
            "Balance must be greater than zero."
        )


def validate_risk_percent(
    risk_percent: float,
) -> None:
    """
    Проверяет процент риска.

    Например:

        1.0  = 1%
        0.5  = 0.5%
        2.0  = 2%
    """

    if risk_percent <= 0:
        raise ValueError(
            "Risk percent must be greater than zero."
        )

    if risk_percent > 100:
        raise ValueError(
            "Risk percent cannot exceed 100%."
        )


def validate_prices(
    entry: float,
    stop_loss: float,
) -> None:
    """
    Проверяет корректность Entry и SL.
    """

    if entry <= 0:
        raise ValueError(
            "Entry price must be greater than zero."
        )

    if stop_loss <= 0:
        raise ValueError(
            "Stop loss must be greater than zero."
        )


# ============================================================
# RISK AMOUNT
# ============================================================

def calculate_risk_amount(
    balance: float,
    risk_percent: float,
) -> float:
    """
    Сколько денег пользователь готов потерять
    при полном срабатывании SL.

    Формула:

        risk_amount =
            balance × risk_percent / 100
    """

    validate_balance(
        balance
    )

    validate_risk_percent(
        risk_percent
    )

    return (
        balance
        * risk_percent
        / 100.0
    )


# ============================================================
# STOP DISTANCE
# ============================================================

def calculate_stop_distance(
    direction: str,
    entry: float,
    stop_loss: float,
) -> float:
    """
    Расстояние от Entry до SL.

    LONG:

        Entry - SL

    SHORT:

        SL - Entry
    """

    validate_prices(
        entry,
        stop_loss,
    )

    if direction == "LONG":

        distance = (
            entry - stop_loss
        )

    elif direction == "SHORT":

        distance = (
            stop_loss - entry
        )

    else:

        raise ValueError(
            f"Unknown direction: {direction}"
        )

    if distance <= 0:
        raise ValueError(
            "Stop loss must be on the correct "
            "side of the entry."
        )

    return distance


# ============================================================
# STOP DISTANCE %
# ============================================================

def calculate_stop_distance_percent(
    entry: float,
    stop_distance: float,
) -> float:
    """
    Процент расстояния Entry → SL.
    """

    if entry <= 0:
        raise ValueError(
            "Entry must be greater than zero."
        )

    return (
        stop_distance
        / entry
        * 100.0
    )


# ============================================================
# POSITION SIZE
# ============================================================

def calculate_position_size(
    risk_amount: float,
    stop_distance: float,
) -> float:
    """
    Рассчитывает количество базового актива.

    Формула:

        position_size =
            risk_amount / stop_distance

    Пример:

        risk = 10 USDT
        Entry = 100
        SL = 98

        distance = 2

        position = 10 / 2 = 5
    """

    if risk_amount <= 0:
        raise ValueError(
            "Risk amount must be greater than zero."
        )

    if stop_distance <= 0:
        raise ValueError(
            "Stop distance must be greater than zero."
        )

    return (
        risk_amount
        / stop_distance
    )


# ============================================================
# POSITION NOTIONAL
# ============================================================

def calculate_position_notional(
    entry: float,
    position_size: float,
) -> float:
    """
    Номинальная стоимость позиции.

    Формула:

        position_size × entry
    """

    if entry <= 0:
        raise ValueError(
            "Entry must be greater than zero."
        )

    if position_size <= 0:
        raise ValueError(
            "Position size must be greater than zero."
        )

    return (
        entry
        * position_size
    )


# ============================================================
# TP BY R
# ============================================================

def calculate_take_profit(
    direction: str,
    entry: float,
    stop_distance: float,
    rr: float,
) -> float:
    """
    Рассчитывает TP на заданном R.

    LONG:

        TP = Entry + RiskDistance × RR

    SHORT:

        TP = Entry - RiskDistance × RR
    """

    if entry <= 0:
        raise ValueError(
            "Entry must be greater than zero."
        )

    if stop_distance <= 0:
        raise ValueError(
            "Stop distance must be greater than zero."
        )

    if rr <= 0:
        raise ValueError(
            "RR must be greater than zero."
        )

    if direction == "LONG":

        return (
            entry
            + stop_distance * rr
        )

    if direction == "SHORT":

        return (
            entry
            - stop_distance * rr
        )

    raise ValueError(
        f"Unknown direction: {direction}"
    )


# ============================================================
# RR BETWEEN PRICES
# ============================================================

def calculate_rr(
    direction: str,
    entry: float,
    stop_loss: float,
    target: float,
) -> Optional[float]:
    """
    Рассчитывает фактический RR между Entry, SL и TP.
    """

    stop_distance = calculate_stop_distance(
        direction=direction,
        entry=entry,
        stop_loss=stop_loss,
    )

    if direction == "LONG":

        reward = (
            target - entry
        )

    elif direction == "SHORT":

        reward = (
            entry - target
        )

    else:

        return None

    if reward <= 0:
        return None

    return (
        reward
        / stop_distance
    )


# ============================================================
# MAIN CALCULATION
# ============================================================

def calculate_risk(
    direction: str,
    balance: float,
    risk_percent: float,
    entry: float,
    stop_loss: float,
    tp1_rr: float = 1.0,
    tp2_rr: float = 2.0,
) -> RiskCalculation:
    """
    Главная функция risk management.

    На вход:

        direction
        balance
        risk %
        entry
        SL

    На выход:

        risk amount
        position size
        position notional
        TP1
        TP2
        фактические RR
    """

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validate_balance(
        balance
    )

    validate_risk_percent(
        risk_percent
    )

    validate_prices(
        entry,
        stop_loss,
    )

    # --------------------------------------------------------
    # RISK AMOUNT
    # --------------------------------------------------------

    risk_amount = calculate_risk_amount(
        balance=balance,
        risk_percent=risk_percent,
    )

    # --------------------------------------------------------
    # STOP DISTANCE
    # --------------------------------------------------------

    stop_distance = calculate_stop_distance(
        direction=direction,
        entry=entry,
        stop_loss=stop_loss,
    )

    stop_distance_percent = (
        calculate_stop_distance_percent(
            entry=entry,
            stop_distance=stop_distance,
        )
    )

    # --------------------------------------------------------
    # POSITION SIZE
    # --------------------------------------------------------

    position_size = calculate_position_size(
        risk_amount=risk_amount,
        stop_distance=stop_distance,
    )

    # --------------------------------------------------------
    # NOTIONAL
    # --------------------------------------------------------

    position_notional = (
        calculate_position_notional(
            entry=entry,
            position_size=position_size,
        )
    )

    # --------------------------------------------------------
    # TAKE PROFITS
    # --------------------------------------------------------

    tp1 = calculate_take_profit(
        direction=direction,
        entry=entry,
        stop_distance=stop_distance,
        rr=tp1_rr,
    )

    tp2 = calculate_take_profit(
        direction=direction,
        entry=entry,
        stop_distance=stop_distance,
        rr=tp2_rr,
    )

    # --------------------------------------------------------
    # ACTUAL RR
    # --------------------------------------------------------

    rr_tp1 = calculate_rr(
        direction=direction,
        entry=entry,
        stop_loss=stop_loss,
        target=tp1,
    )

    rr_tp2 = calculate_rr(
        direction=direction,
        entry=entry,
        stop_loss=stop_loss,
        target=tp2,
    )

    if rr_tp1 is None:
        raise ValueError(
            "Unable to calculate TP1 RR."
        )

    if rr_tp2 is None:
        raise ValueError(
            "Unable to calculate TP2 RR."
        )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    return RiskCalculation(
        balance=balance,
        risk_percent=risk_percent,
        risk_amount=risk_amount,
        entry=entry,
        stop_loss=stop_loss,
        stop_distance=stop_distance,
        stop_distance_percent=(
            stop_distance_percent
        ),
        position_size=position_size,
        position_notional=position_notional,
        tp1=tp1,
        tp2=tp2,
        rr_tp1=rr_tp1,
        rr_tp2=rr_tp2,
    )


# ============================================================
# SERIALIZATION
# ============================================================

def risk_to_dict(
    result: RiskCalculation,
) -> dict:
    """
    Преобразует результат расчёта в dict.

    Используется для Telegram / JSON / логирования.
    """

    return {
        "balance": result.balance,
        "risk_percent": result.risk_percent,
        "risk_amount": result.risk_amount,
        "entry": result.entry,
        "stop_loss": result.stop_loss,
        "stop_distance": result.stop_distance,
        "stop_distance_percent": (
            result.stop_distance_percent
        ),
        "position_size": result.position_size,
        "position_notional": (
            result.position_notional
        ),
        "tp1": result.tp1,
        "tp2": result.tp2,
        "rr_tp1": result.rr_tp1,
        "rr_tp2": result.rr_tp2,
    }
