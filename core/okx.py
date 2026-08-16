# ============================================================
# OKX MARKET DATA CLIENT
# SFP + MSS BOT
# ============================================================

from __future__ import annotations

import time
from typing import List, Optional

import pandas as pd
import requests

from config import (
    OKX_BASE_URL,
    CANDLE_LIMIT,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    EXCLUDED_BASE_CURRENCIES,
    MIN_24H_VOLUME_USDT,
)


# ============================================================
# CONSTANTS
# ============================================================

CANDLES_ENDPOINT = "/api/v5/market/candles"
TICKERS_ENDPOINT = "/api/v5/market/tickers"

SUPPORTED_TIMEFRAMES = {
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1H",
    "2H",
    "4H",
    "6H",
    "12H",
    "1D",
    "2D",
    "3D",
    "1W",
    "1M",
    "3M",
}


# ============================================================
# EXCEPTIONS
# ============================================================

class OKXError(Exception):
    """Базовая ошибка OKX API."""


class OKXRequestError(OKXError):
    """Ошибка HTTP-запроса."""


class OKXResponseError(OKXError):
    """OKX вернул code != 0."""


class OKXDataError(OKXError):
    """Некорректные или пустые данные от OKX."""


# ============================================================
# CLIENT
# ============================================================

class OKXClient:
    """
    Клиент публичного Market Data API OKX.
    """

    def __init__(
        self,
        base_url: str = OKX_BASE_URL,
        timeout: int = REQUEST_TIMEOUT,
        max_retries: int = MAX_RETRIES,
    ) -> None:

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max(1, int(max_retries))
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "SFP-MSS-Bot/1.0",
            }
        )

    # ========================================================
    # VALIDATIONS
    # ========================================================

    @staticmethod
    def validate_timeframe(timeframe: str) -> None:
        if timeframe not in SUPPORTED_TIMEFRAMES:
            raise ValueError(f"Unsupported OKX timeframe: {timeframe}")

    @staticmethod
    def validate_symbol(symbol: str) -> None:
        if not symbol or "-" not in symbol:
            raise ValueError(f"Invalid OKX instrument ID: {symbol}")

    # ========================================================
    # HTTP REQUEST
    # ========================================================

    def _request(self, endpoint: str, params: dict) -> dict:
        url = self.base_url + endpoint
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.timeout,
                )

                if response.status_code in {429, 500, 502, 503, 504}:
                    last_error = OKXRequestError(
                        f"Temporary OKX HTTP error {response.status_code}"
                    )
                    if attempt < self.max_retries:
                        time.sleep(min(2 ** (attempt - 1), 5))
                        continue
                    raise last_error

                if not response.ok:
                    raise OKXRequestError(
                        f"OKX HTTP error {response.status_code}: {response.text[:500]}"
                    )

                try:
                    payload = response.json()
                except ValueError as exc:
                    raise OKXRequestError("OKX returned invalid JSON.") from exc

                code = payload.get("code")
                if code != "0":
                    message = payload.get("msg", "Unknown OKX API error.")
                    raise OKXResponseError(
                        f"OKX API error code={code}, message={message}"
                    )

                return payload

            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise OKXRequestError(
                        f"Unable to connect to OKX after {self.max_retries} attempts."
                    ) from exc
                time.sleep(min(2 ** (attempt - 1), 5))

            except OKXError:
                raise

            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise OKXRequestError("Unexpected OKX request error.") from exc
                time.sleep(min(2 ** (attempt - 1), 5))

        raise OKXRequestError("OKX request failed.") from last_error

    # ========================================================
    # DYNAMIC TOP SYMBOLS (BY 24H VOLUME)
    # ========================================================

    def get_top_volume_symbols(
        self,
        top_n: int = 200,
        quote_currency: str = "USDT",
        min_volume_usdt: float = MIN_24H_VOLUME_USDT,
    ) -> List[str]:
        """
        Получает топ N инструментов по объёму торгов за 24 часа.
        Фильтрует стейблкоины и неликвидные пары.
        """
        params = {"instType": "SPOT"}
        payload = self._request(TICKERS_ENDPOINT, params=params)
        data = payload.get("data", [])

        valid_tickers = []
        suffix = f"-{quote_currency}"

        for item in data:
            inst_id = item.get("instId", "")
            if not inst_id.endswith(suffix):
                continue

            base = inst_id.split("-")[0]
            if base in EXCLUDED_BASE_CURRENCIES:
                continue

            try:
                vol_usdt = float(item.get("volCcy24h", 0.0))
            except (ValueError, TypeError):
                vol_usdt = 0.0

            if vol_usdt >= min_volume_usdt:
                valid_tickers.append((inst_id, vol_usdt))

        # Сортируем по убыванию суточного объёма в USDT
        valid_tickers.sort(key=lambda x: x[1], reverse=True)

        return [t[0] for t in valid_tickers[:top_n]]

    # ========================================================
    # CANDLES → DATAFRAME
    # ========================================================

    def get_raw_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = CANDLE_LIMIT,
    ) -> list[list[str]]:
        self.validate_symbol(symbol)
        self.validate_timeframe(timeframe)

        limit = min(max(1, int(limit)), 300)
        params = {
            "instId": symbol,
            "bar": timeframe,
            "limit": str(limit),
        }

        payload = self._request(CANDLES_ENDPOINT, params=params)
        data = payload.get("data")

        if not isinstance(data, list) or not data:
            raise OKXDataError(f"No candles returned for {symbol} {timeframe}.")

        return data

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = CANDLE_LIMIT,
        closed_only: bool = True,
    ) -> pd.DataFrame:
        raw = self.get_raw_candles(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
        )

        rows = []
        for candle in raw:
            if len(candle) < 9:
                continue
            try:
                rows.append(
                    {
                        "timestamp": pd.to_datetime(
                            int(candle[0]),
                            unit="ms",
                            utc=True,
                        ),
                        "open": float(candle[1]),
                        "high": float(candle[2]),
                        "low": float(candle[3]),
                        "close": float(candle[4]),
                        "volume": float(candle[5]),
                        "confirm": str(candle[8]),
                    }
                )
            except (TypeError, ValueError):
                continue

        if not rows:
            raise OKXDataError(f"No valid candles for {symbol} {timeframe}.")

        df = pd.DataFrame(rows)
        if closed_only:
            df = df[df["confirm"] == "1"]

        df = (
            df.sort_values("timestamp")
            .drop_duplicates(subset=["timestamp"], keep="last")
            .reset_index(drop=True)
        )

        if df.empty:
            raise OKXDataError(f"No closed candles for {symbol} {timeframe}.")

        return df

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "OKXClient":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def get_market_data(
    symbol: str,
    timeframe: str,
    limit: int = CANDLE_LIMIT,
) -> pd.DataFrame:
    with OKXClient() as client:
        return client.get_candles(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
            closed_only=True,
        )
