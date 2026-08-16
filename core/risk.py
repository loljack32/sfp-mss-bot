# ============================================================
# RISK MANAGEMENT WITH LEVERAGE & MARGIN
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config import DEFAULT_LEVERAGE


# ============================================================
# DATA CLASS
# ============================================================

@dataclass(frozen=True)
class RiskCalculation:
    """
    Полный результат расчёта позиции с учётом кредитного плеча и маржи.
    """

    balance: float
    risk_percent: float
    risk_amount: float
    leverage: int

    margin_required: float          # Сколько своих $ нужно в сделку
    effective_leverage: float       # Фактическое используемое плечо
    estimated_liquidation: float    # Расчетная цена ликвидации

    entry: float
    stop_loss: float
    stop_distance: float
    stop_distance_percent: float

    position_size: float            # Количество монет
    position_notional: float        # Полная стоимость позиции в $

    tp1: float
    tp2: float
    rr_tp1: float
    rr_tp2: float


# ============================================================
# VALIDATION
# ============================================================

def validate_balance(balance: float) -> None:
    if balance <= 0:
        raise ValueError("Balance must be greater than zero.")


def validate_risk_percent(risk_percent: float) -> None:
    if risk_percent <= 0:
        raise ValueError("Risk percent must be greater than zero.")
    if risk_percent > 100:
        raise ValueError("Risk percent cannot exceed 100%.")


def validate_prices(entry: float, stop_loss: float) -> None:
    if entry <= 0:
        raise ValueError("Entry price must be greater than zero.")
    if stop_loss <= 0:
        raise ValueError("Stop loss must be greater than zero.")


# ============================================================
# CALCULATIONS
# ============================================================

def calculate_risk_amount(balance: float, risk_percent: float) -> float:
    validate_balance(balance)
    validate_risk_percent(risk_percent)
    return balance * risk_percent / 100.0


def calculate_stop_distance(direction: str, entry: float, stop_loss: float) -> float:
    validate_prices(entry, stop_loss)
    if direction == "LONG":
        distance = entry - stop_loss
    elif direction == "SHORT":
        distance = stop_loss - entry
    else:
        raise ValueError(f"Unknown direction: {direction}")

    if distance <= 0:
        raise ValueError("Stop loss must be on the correct side of the entry.")
    return distance


def calculate_stop_distance_percent(entry: float, stop_distance: float) -> float:
    if entry <= 0:
        raise ValueError("Entry must be greater than zero.")
    return (stop_distance / entry) * 100.0


def calculate_position_size(risk_amount: float, stop_distance: float) -> float:
    if risk_amount <= 0 or stop_distance <= 0:
        raise ValueError("Risk amount and stop distance must be greater than zero.")
    return risk_amount / stop_distance


def calculate_position_notional(entry: float, position_size: float) -> float:
    return entry * position_size


def calculate_take_profit(
    direction: str,
    entry: float,
    stop_distance: float,
    rr: float,
) -> float:
    if direction == "LONG":
        return entry + stop_distance * rr
    if direction == "SHORT":
        return entry - stop_distance * rr
    raise ValueError(f"Unknown direction: {direction}")


def calculate_rr(
    direction: str,
    entry: float,
    stop_loss: float,
    target: float,
) -> Optional[float]:
    stop_distance = calculate_stop_distance(
        direction=direction,
        entry=entry,
        stop_loss=stop_loss,
    )
    if direction == "LONG":
        reward = target - entry
    elif direction == "SHORT":
        reward = entry - target
    else:
        return None

    if reward <= 0:
        return None
    return reward / stop_distance


def estimate_liquidation_price(
    direction: str,
    entry: float,
    leverage: int,
    maintenance_margin_rate: float = 0.005,  # 0.5% стандартная ставка поддержки на бирже
) -> float:
    """
    Расчёт ориентировочной цены ликвидации позиции с изолированным плечом.
    """
    if leverage <= 1:
        return 0.0 if direction == "LONG" else entry * 2.0

    if direction == "LONG":
        # Для лонга цена падает до ликвидации
        liq_distance_pct = (1.0 / leverage) - maintenance_margin_rate
        return max(0.0, entry * (1.0 - liq_distance_pct))
    else:
        # Для шорта цена растет до ликвидации
        liq_distance_pct = (1.0 / leverage) - maintenance_margin_rate
        return entry * (1.0 + liq_distance_pct)


# ============================================================
# MAIN CALCULATION
# ============================================================

def calculate_risk(
    direction: str,
    balance: float,
    risk_percent: float,
    entry: float,
    stop_loss: float,
    leverage: int = DEFAULT_LEVERAGE,
    tp1_rr: float = 1.0,
    tp2_rr: float = 2.0,
) -> RiskCalculation:
    validate_balance(balance)
    validate_risk_percent(risk_percent)
    validate_prices(entry, stop_loss)

    leverage = max(1, int(leverage))

    # 1. Сумма риска
    risk_amount = calculate_risk_amount(balance=balance, risk_percent=risk_percent)

    # 2. Дистанция стопа
    stop_distance = calculate_stop_distance(direction=direction, entry=entry, stop_loss=stop_loss)
    stop_distance_percent = calculate_stop_distance_percent(entry=entry, stop_distance=stop_distance)

    # 3. Размер позиции
    position_size = calculate_position_size(risk_amount=risk_amount, stop_distance=stop_distance)
    position_notional = calculate_position_notional(entry=entry, position_size=position_size)

    # 4. Расчёт маржи и плеча
    margin_required = position_notional / leverage
    effective_leverage = position_notional / balance if balance > 0 else 0.0

    # 5. Ликвидация
    estimated_liquidation = estimate_liquidation_price(
        direction=direction,
        entry=entry,
        leverage=leverage,
    )

    # 6. Тейк-профиты
    tp1 = calculate_take_profit(direction=direction, entry=entry, stop_distance=stop_distance, rr=tp1_rr)
    tp2 = calculate_take_profit(direction=direction, entry=entry, stop_distance=stop_distance, rr=tp2_rr)

    rr_tp1 = calculate_rr(direction=direction, entry=entry, stop_loss=stop_loss, target=tp1)
    rr_tp2 = calculate_rr(direction=direction, entry=entry, stop_loss=stop_loss, target=tp2)

    if rr_tp1 is None or rr_tp2 is None:
        raise ValueError("Unable to calculate TP RR.")

    return RiskCalculation(
        balance=balance,
        risk_percent=risk_percent,
        risk_amount=risk_amount,
        leverage=leverage,
        margin_required=margin_required,
        effective_leverage=effective_leverage,
        estimated_liquidation=estimated_liquidation,
        entry=entry,
        stop_loss=stop_loss,
        stop_distance=stop_distance,
        stop_distance_percent=stop_distance_percent,
        position_size=position_size,
        position_notional=position_notional,
        tp1=tp1,
        tp2=tp2,
        rr_tp1=rr_tp1,
        rr_tp2=rr_tp2,
    )


def risk_to_dict(result: RiskCalculation) -> dict:
    return {
        "balance": result.balance,
        "risk_percent": result.risk_percent,
        "risk_amount": result.risk_amount,
        "leverage": result.leverage,
        "margin_required": result.margin_required,
        "effective_leverage": result.effective_leverage,
        "estimated_liquidation": result.estimated_liquidation,
        "entry": result.entry,
        "stop_loss": result.stop_loss,
        "stop_distance": result.stop_distance,
        "stop_distance_percent": result.stop_distance_percent,
        "position_size": result.position_size,
        "position_notional": result.position_notional,
        "tp1": result.tp1,
        "tp2": result.tp2,
        "rr_tp1": result.rr_tp1,
        "rr_tp2": result.rr_tp2,
    }
