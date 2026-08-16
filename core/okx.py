# ============================================================
# OKX MARKET DATA CLIENT
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


class OKXClient:
    """
    Клиент для получения публичных market data с OKX.

    Авторизация для свечей не требуется.
    """

    def __init__(
        self,
        base_url: str = OKX_BASE_URL,
        timeout: int = REQUEST_TIMEOUT,
        max_retries: int = MAX_RETRIES,
    ) -> None:

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

        self.session = requests.Session()

        self.session.headers.update(
            {
                "User-Agent": "SFP-MSS-Bot/1.0",
                "Accept": "application/json",
            }
        )

    # --------------------------------------------------------
    # REQUEST
    # --------------------------------------------------------

    def _get(
        self,
        endpoint: str,
        params: dict,
    ) -> Optional[dict]:

        url = f"{self.base_url}{endpoint}"

        last_error = None

        for attempt in range(1, self.max_retries + 1):

            try:

                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.timeout,
                )

                response.raise_for_status()

                data = response.json()

                if data.get("code") != "0":
                    raise RuntimeError(
                        f"OKX API error: "
                        f"code={data.get('code')} "
                        f"message={data.get('msg')}"
                    )

                return data

            except Exception as exc:

                last_error = exc

                if attempt < self.max_retries:
                    time.sleep(attempt)

        print(
            f"[OKX ERROR] request failed: "
            f"{url} params={params} error={last_error}"
        )

        return None

    # --------------------------------------------------------
    # CANDLES
    # --------------------------------------------------------

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = CANDLE_LIMIT,
    ) -> pd.DataFrame:

        params = {
            "instId": symbol,
            "bar": timeframe,
            "limit": str(limit),
        }

        response = self._get(
            "/api/v5/market/candles",
            params,
        )

        if not response:
            return pd.DataFrame()

        rows = response.get("data", [])

        if not rows:
            return pd.DataFrame()

        # OKX candle format:
        #
        # [
        #   ts,
        #   open,
        #   high,
        #   low,
        #   close,
        #   volume,
        #   volCcy,
        #   volCcyQuote,
        #   confirm
        # ]

        columns = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "vol_ccy",
            "vol_ccy_quote",
            "confirm",
        ]

        df = pd.DataFrame(
            rows,
            columns=columns,
        )

        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "vol_ccy",
            "vol_ccy_quote",
        ]

        for column in numeric_columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        df["timestamp"] = pd.to_numeric(
            df["timestamp"],
            errors="coerce",
        )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            unit="ms",
            utc=True,
        )

        df["confirm"] = df["confirm"].astype(str)

        # OKX обычно возвращает свечи от новых к старым.
        # Нам нужен хронологический порядок.
        df = df.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        # Удаляем полностью битые строки.
        df = df.dropna(
            subset=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        ).reset_index(drop=True)

        return df

    # --------------------------------------------------------
    # LATEST PRICE
    # --------------------------------------------------------

    def get_ticker_price(
        self,
        symbol: str,
    ) -> Optional[float]:

        params = {
            "instId": symbol,
        }

        response = self._get(
            "/api/v5/market/ticker",
            params,
        )

        if not response:
            return None

        rows = response.get("data", [])

        if not rows:
            return None

        try:
            return float(rows[0]["last"])

        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            return None


# ============================================================
# HELPER
# ============================================================

def create_okx_client() -> OKXClient:
    """
    Создаёт стандартный OKX клиент.
    """

    return OKXClient()
