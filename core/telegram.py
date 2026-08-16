# ============================================================
# TELEGRAM NOTIFIER (WITH LEVERAGE & MARGIN DISPLAY)
# ============================================================

from __future__ import annotations

import os
from typing import Optional

import requests

from config import (
    TELEGRAM_BOT_TOKEN_ENV,
    TELEGRAM_CHAT_ID_ENV,
    REQUEST_TIMEOUT,
)

from core.signals import TradingSignal


# ============================================================
# MESSAGE FORMATTING
# ============================================================

def format_signal_message(signal: TradingSignal) -> str:
    """
    Формирует красивое HTML-сообщение для Telegram с учётом плеча и маржи.
    """
    is_long = signal.direction == "LONG"
    direction_icon = "🟢" if is_long else "🔴"
    trend_icon = (
        "📈"
        if signal.htf_trend == "BULLISH"
        else "📉"
        if signal.htf_trend == "BEARISH"
        else "⚖️"
    )

    if signal.setup_type == "TREND":
        tactic_badge = "🎯 <b>Тактика:</b> <code>SFP + MSS (По тренду 4H)</code>"
        header_title = f"{direction_icon} <b>NEW SIGNAL: {signal.symbol} | {signal.direction}</b>"
    else:
        tactic_badge = "⚡️ <b>Тактика:</b> <code>SFP + MSS (Контртренд / Откат 4H ⚠️)</code>"
        header_title = f"⚡️ <b>NEW SIGNAL: {signal.symbol} | {signal.direction} [PULLBACK]</b>"

    def fmt_price(price: float) -> str:
        if price >= 100:
            return f"{price:,.2f}"
        if price >= 1:
            return f"{price:,.4f}"
        return f"{price:,.6f}"

    lines = [
        header_title,
        tactic_badge,
        f"⏱ <b>Timeframe:</b> <code>{signal.timeframe}</code>",
        f"⭐ <b>Signal Score:</b> <code>{signal.signal_score:.1f} / 100</code>",
        "",
        "📊 <b>Market Context:</b>",
        f"• HTF Trend (4H): <b>{signal.htf_trend}</b> {trend_icon}",
        f"• SFP Score: <code>{signal.sfp_score:.1f}</code> (Sweep: <code>{fmt_price(signal.sfp.sweep_extreme)}</code>)",
        f"• MSS Score: <code>{signal.mss_score:.1f}</code> (Break: <code>{fmt_price(signal.mss.structure_level)}</code> [{signal.mss.structure_label}])",
        "",
        "🎯 <b>Trade Parameters:</b>",
        f"• <b>Entry:</b> <code>{fmt_price(signal.entry)}</code>",
        f"• <b>Stop Loss:</b> <code>{fmt_price(signal.stop_loss)}</code> (<code>{signal.stop_distance_percent:.2f}%</code>)",
        f"• <b>TP1 (1R):</b> <code>{fmt_price(signal.tp1)}</code> [RR {signal.rr_tp1:.2f}]",
        f"• <b>TP2 (2R):</b> <code>{fmt_price(signal.tp2)}</code> [RR {signal.rr_tp2:.2f}]",
    ]

    if signal.liquidity_target is not None:
        target_label = (
            f" ({signal.liquidity_target_label})"
            if signal.liquidity_target_label
            else ""
        )
        lines.append(
            f"• <b>Liquidity Target:</b> <code>{fmt_price(signal.liquidity_target)}</code>{target_label}"
        )

    lines.extend([
        "",
        "💼 <b>Margin & Position Size (Фьючерсы):</b>",
        f"• ⚡️ <b>Кредитное плечо:</b> <code>{signal.leverage}x</code> (Isolated)",
        f"• 💵 <b>Маржа (свои деньги):</b> <code>${signal.margin_required:,.2f} USDT</code>",
        f"• 📦 <b>Номинал позиции:</b> <code>${signal.position_notional:,.2f}</code> ({signal.position_size:,.4f} монет)",
        f"• ⚠️ <b>Риск при стопе:</b> <code>{signal.risk_percent:.2f}%</code> (<code>${signal.risk_amount:,.2f}</code>)",
        f"• ☠️ <b>Оценка ликвидации:</b> <code>{fmt_price(signal.estimated_liquidation)}</code>",
        "",
        f"🕒 <i>Candle Time: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</i>",
    ])

    return "\n".join(lines)


# ============================================================
# TELEGRAM CLIENT
# ============================================================

class TelegramNotifier:
    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        timeout: int = REQUEST_TIMEOUT,
    ) -> None:
        self.bot_token = bot_token or os.getenv(TELEGRAM_BOT_TOKEN_ENV, "")
        self.chat_id = chat_id or os.getenv(TELEGRAM_CHAT_ID_ENV, "")
        self.timeout = timeout

        if not self.bot_token:
            print(f"[WARN] Telegram bot token is not set (env: {TELEGRAM_BOT_TOKEN_ENV}).")
        if not self.chat_id:
            print(f"[WARN] Telegram chat ID is not set (env: {TELEGRAM_CHAT_ID_ENV}).")

    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def send_message(
        self,
        text: str,
        parse_mode: str = "HTML",
        disable_web_page_preview: bool = True,
    ) -> bool:
        if not self.is_configured:
            print("[WARN] Telegram is not configured. Skipping notification.")
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": disable_web_page_preview,
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            data = response.json()
            if not data.get("ok"):
                print(f"[ERROR] Telegram API error: {data.get('description')}")
                return False
            return True
        except Exception as exc:
            print(f"[ERROR] Failed to send Telegram message: {exc}")
            return False

    def send_signal(self, signal: TradingSignal) -> bool:
        message = format_signal_message(signal)
        return self.send_message(message)


def send_signal(signal: TradingSignal) -> bool:
    notifier = TelegramNotifier()
    return notifier.send_signal(signal)
