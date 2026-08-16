# ============================================================
# OKX MARKET DATA CLIENT
# SFP + MSS BOT
# ============================================================

from __future__ import annotations

import time
from typing import Optional

import pandas as pd
import requests

from config import (
    OKX_BASE_URL,
    CANDLE_LIMIT,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
)


# ============================================================
# CONSTANTS
# ============================================================

CANDLES_ENDPOINT = "/api/v5/market/candles"

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
    """
    Базовая ошибка OKX API.
    """


class OKXRequestError(OKXError):
    """
    Ошибка HTTP-запроса.
    """


class OKXResponseError(OKXError):
    """
    OKX вернул code != 0.
    """


class OKXDataError(OKXError):
    """
    Некорректные или пустые данные от OKX.
    """


# ============================================================
# CLIENT
# ============================================================

class OKXClient:
    """
    Клиент публичного Market Data API OKX.

    Используется только для получения рыночных данных.

    Авторизация API-ключом для market candles НЕ требуется.

    Основной endpoint:

        GET /api/v5/market/candles
    """

    def __init__(
        self,
        base_url: str = OKX_BASE_URL,
        timeout: int = REQUEST_TIMEOUT,
        max_retries: int = MAX_RETRIES,
    ) -> None:

        self.base_url = (
            base_url.rstrip("/")
        )

        self.timeout = timeout

        self.max_retries = max(
            1,
            int(max_retries),
        )

        self.session = requests.Session()

        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": (
                    "SFP-MSS-Bot/1.0"
                ),
            }
        )


    # ========================================================
    # TIMEFRAME VALIDATION
    # ========================================================

    @staticmethod
    def validate_timeframe(
        timeframe: str,
    ) -> None:
        """
        Проверяет допустимость timeframe.
        """

        if timeframe not in SUPPORTED_TIMEFRAMES:

            raise ValueError(
                "Unsupported OKX timeframe: "
                f"{timeframe}"
            )


    # ========================================================
    # SYMBOL VALIDATION
    # ========================================================

    @staticmethod
    def validate_symbol(
        symbol: str,
    ) -> None:
        """
        Проверяет формат торгового инструмента.

        Например:

            BTC-USDT
            ETH-USDT
        """

        if not symbol:
            raise ValueError(
                "Symbol cannot be empty."
            )

        if "-" not in symbol:
            raise ValueError(
                "Invalid OKX instrument ID: "
                f"{symbol}"
            )


    # ========================================================
    # HTTP REQUEST
    # ========================================================

    def _request(
        self,
        params: dict,
    ) -> dict:
        """
        Выполняет GET-запрос к OKX.

        Реализует несколько попыток при временных
        сетевых ошибках / HTTP 429 / 5xx.
        """

        url = (
            self.base_url
            + CANDLES_ENDPOINT
        )

        last_error: Optional[Exception] = None

        for attempt in range(
            1,
            self.max_retries + 1,
        ):

            try:

                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.timeout,
                )

                # ------------------------------------------------
                # RETRYABLE HTTP STATUS
                # ------------------------------------------------

                if response.status_code in {
                    429,
                    500,
                    502,
                    503,
                    504,
                }:

                    last_error = (
                        OKXRequestError(
                            "Temporary OKX HTTP error "
                            f"{response.status_code}"
                        )
                    )

                    if attempt < self.max_retries:

                        time.sleep(
                            min(
                                2 ** (attempt - 1),
                                5,
                            )
                        )

                        continue

                    raise last_error

                # ------------------------------------------------
                # OTHER HTTP ERROR
                # ------------------------------------------------

                if not response.ok:

                    raise OKXRequestError(
                        "OKX HTTP error "
                        f"{response.status_code}: "
                        f"{response.text[:500]}"
                    )

                # ------------------------------------------------
                # JSON
                # ------------------------------------------------

                try:

                    payload = response.json()

                except ValueError as exc:

                    raise OKXRequestError(
                        "OKX returned invalid JSON."
                    ) from exc

                # ------------------------------------------------
                # API CODE
                # ------------------------------------------------

                code = payload.get(
                    "code"
                )

                if code != "0":

                    message = payload.get(
                        "msg",
                        "Unknown OKX API error.",
                    )

                    raise OKXResponseError(
                        f"OKX API error "
                        f"code={code}, "
                        f"message={message}"
                    )

                return payload

            except (
                requests.Timeout,
                requests.ConnectionError,
            ) as exc:

                last_error = exc

                if attempt >= self.max_retries:

                    raise OKXRequestError(
                        "Unable to connect to OKX "
                        f"after {self.max_retries} attempts."
                    ) from exc

                time.sleep(
                    min(
                        2 ** (attempt - 1),
                        5,
                    )
                )

            except OKXError:
                raise

            except requests.RequestException as exc:

                last_error = exc

                if attempt >= self.max_retries:

                    raise OKXRequestError(
                        "Unexpected OKX request error."
                    ) from exc

                time.sleep(
                    min(
                        2 ** (attempt - 1),
                        5,
                    )
                )

        raise OKXRequestError(
            "OKX request failed."
        ) from last_error


    # ========================================================
    # RAW CANDLES
    # ========================================================

    def get_raw_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = CANDLE_LIMIT,
    ) -> list[list[str]]:
        """
        Получает сырые свечи OKX.

        OKX возвращает свечу примерно в формате:

            [
                timestamp,
                open,
                high,
                low,
                close,
                volume,
                volume_currency,
                volume_currency_quote,
                confirm
            ]

        confirm:

            "1" = свеча закрыта
            "0" = свеча ещё формируется
        """

        self.validate_symbol(
            symbol
        )

        self.validate_timeframe(
            timeframe
        )

        if limit < 1:
            raise ValueError(
                "limit must be >= 1"
            )

        # OKX maximum for candles endpoint
        # is 300.
        limit = min(
            int(limit),
            300,
        )

        params = {
            "instId": symbol,
            "bar": timeframe,
            "limit": str(limit),
        }

        payload = self._request(
            params=params
        )

        data = payload.get(
            "data"
        )

        if not isinstance(
            data,
            list,
        ):

            raise OKXDataError(
                "OKX response does not contain "
                "a valid data array."
            )

        if not data:

            raise OKXDataError(
                f"OKX returned no candles for "
                f"{symbol} {timeframe}."
            )

        return data


    # ========================================================
    # CANDLES → DATAFRAME
    # ========================================================

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = CANDLE_LIMIT,
        closed_only: bool = True,
    ) -> pd.DataFrame:
        """
        Получает OHLCV и возвращает DataFrame.

        ВАЖНО:

        При closed_only=True незакрытая текущая свеча
        удаляется.

        Это обязательная защита стратегии от анализа
        формирующейся свечи.
        """

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

                timestamp_ms = int(
                    candle[0]
                )

                open_price = float(
                    candle[1]
                )

                high = float(
                    candle[2]
                )

                low = float(
                    candle[3]
                )

                close = float(
                    candle[4]
                )

                volume = float(
                    candle[5]
                )

                volume_currency = float(
                    candle[6]
                )

                volume_quote = float(
                    candle[7]
                )

                confirm = str(
                    candle[8]
                )

            except (
                TypeError,
                ValueError,
            ):

                continue

            rows.append(
                {
                    "timestamp": pd.to_datetime(
                        timestamp_ms,
                        unit="ms",
                        utc=True,
                    ),

                    "open": open_price,

                    "high": high,

                    "low": low,

                    "close": close,

                    "volume": volume,

                    "volume_currency": (
                        volume_currency
                    ),

                    "volume_quote": (
                        volume_quote
                    ),

                    "confirm": confirm,
                }
            )

        if not rows:

            raise OKXDataError(
                f"No valid candles returned "
                f"for {symbol} {timeframe}."
            )

        df = pd.DataFrame(
            rows
        )

        # ----------------------------------------------------
        # REMOVE INVALID PRICES
        # ----------------------------------------------------

        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "volume_currency",
            "volume_quote",
        ]

        for column in numeric_columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        df = df.dropna(
            subset=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )

        # ----------------------------------------------------
        # OHLC VALIDATION
        # ----------------------------------------------------

        df = df[
            (df["high"] >= df["low"])
            &
            (df["high"] >= df["open"])
            &
            (df["high"] >= df["close"])
            &
            (df["low"] <= df["open"])
            &
            (df["low"] <= df["close"])
            &
            (df["volume"] >= 0)
        ]

        # ----------------------------------------------------
        # CLOSED CANDLES ONLY
        # ----------------------------------------------------

        if closed_only:

            df = df[
                df["confirm"] == "1"
            ]

        # ----------------------------------------------------
        # SORT CHRONOLOGICALLY
        # ----------------------------------------------------

        df = (
            df.sort_values(
                "timestamp"
            )
            .drop_duplicates(
                subset=["timestamp"],
                keep="last",
            )
            .reset_index(
                drop=True
            )
        )

        if df.empty:

            raise OKXDataError(
                f"No closed valid candles "
                f"available for "
                f"{symbol} {timeframe}."
            )

        return df


    # ========================================================
    # LATEST PRICE
    # ========================================================

    def get_latest_close(
        self,
        symbol: str,
        timeframe: str = "15m",
    ) -> float:
        """
        Возвращает close последней закрытой свечи.
        """

        df = self.get_candles(
            symbol=symbol,
            timeframe=timeframe,
            limit=5,
            closed_only=True,
        )

        if df.empty:

            raise OKXDataError(
                f"No closed candles for "
                f"{symbol}."
            )

        return float(
            df.iloc[-1]["close"]
        )


    # ========================================================
    # MULTI-TIMEFRAME DATA
    # ========================================================

    def get_multi_timeframe(
        self,
        symbol: str,
        timeframes: list[str],
        limit: int = CANDLE_LIMIT,
    ) -> dict[str, pd.DataFrame]:
        """
        Получает несколько таймфреймов для одного инструмента.

        Например:

            {
                "4H": DataFrame,
                "1H": DataFrame,
                "15m": DataFrame,
            }
        """

        result: dict[
            str,
            pd.DataFrame
        ] = {}

        for timeframe in timeframes:

            result[timeframe] = (
                self.get_candles(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=limit,
                    closed_only=True,
                )
            )

        return result


    # ========================================================
    # HEALTH CHECK
    # ========================================================

    def health_check(
        self,
        symbol: str = "BTC-USDT",
    ) -> bool:
        """
        Простая проверка доступности OKX API.
        """

        try:

            self.get_candles(
                symbol=symbol,
                timeframe="15m",
                limit=5,
                closed_only=True,
            )

            return True

        except OKXError:

            return False


    # ========================================================
    # CLOSE SESSION
    # ========================================================

    def close(self) -> None:
        """
        Закрывает HTTP session.
        """

        self.session.close()


    # ========================================================
    # CONTEXT MANAGER
    # ========================================================

    def __enter__(
        self,
    ) -> "OKXClient":

        return self


    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:

        self.close()


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def get_market_data(
    symbol: str,
    timeframe: str,
    limit: int = CANDLE_LIMIT,
) -> pd.DataFrame:
    """
    Удобная функция для получения свечей
    без ручного создания OKXClient.
    """

    with OKXClient() as client:

        return client.get_candles(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
            closed_only=True,
        )
