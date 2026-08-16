# ============================================================
# SFP + MSS BOT — CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# OKX & SCANNER SETTINGS
# ------------------------------------------------------------

OKX_BASE_URL = "https://www.okx.com"

# Включить динамический выбор топ-монет по объёму торгов
DYNAMIC_TOP_SYMBOLS = True

# Сколько монет из топа сканировать
TOP_SYMBOLS_COUNT = 200

# Разрешить контртрендовые сигналы (откаты против 4H)
ENABLE_COUNTER_TREND_SIGNALS = True

# Минимальный суточный объём торгов (в USDT)
MIN_24H_VOLUME_USDT = 500_000

# Стейблкоины и нежелательные пары
EXCLUDED_BASE_CURRENCIES = {
    "USDC",
    "USDK",
    "TUSD",
    "DAI",
    "EUR",
    "EURT",
    "BUSD",
    "FDUSD",
    "USDE",
    "USDD",
    "USDG",
}

# Резервный список монет
SYMBOLS = [
    "BTC-USDT",
    "ETH-USDT",
    "SOL-USDT",
    "XRP-USDT",
    "DOGE-USDT",
    "AVAX-USDT",
    "LINK-USDT",
    "ADA-USDT",
    "DOT-USDT",
    "SUI-USDT",
    "NEAR-USDT",
    "APT-USDT",
]

# Таймфреймы
HTF_TIMEFRAME = "4H"
MTF_TIMEFRAME = "1H"
ENTRY_TIMEFRAME = "15m"

# Количество свечей для анализа
CANDLE_LIMIT = 300


# ------------------------------------------------------------
# MARKET STRUCTURE
# ------------------------------------------------------------

SWING_LEFT = 3
SWING_RIGHT = 3

HTF_SWING_LEFT = 4
HTF_SWING_RIGHT = 4


# ------------------------------------------------------------
# SFP
# ------------------------------------------------------------

MIN_SFP_SWEEP_ATR = 0.05
MAX_SFP_SWEEP_ATR = 0.80
MIN_SFP_RECLAIM_RATIO = 0.50


# ------------------------------------------------------------
# MSS
# ------------------------------------------------------------

MIN_MSS_DISPLACEMENT_ATR = 0.20
MIN_MSS_BODY_RATIO = 0.60
MSS_REQUIRE_CLOSE = True


# ------------------------------------------------------------
# VOLUME
# ------------------------------------------------------------

MIN_VOLUME_RATIO = 1.20
VOLUME_LOOKBACK = 20


# ------------------------------------------------------------
# ATR
# ------------------------------------------------------------

ATR_PERIOD = 14
MIN_ATR_PERCENT = 0.0010


# ------------------------------------------------------------
# SIGNAL SCORE
# ------------------------------------------------------------

MIN_SIGNAL_SCORE = 68  # Порог скоринга под RR от 1.6

SIGNAL_SCORE_WEIGHTS = {
    "htf_structure": 20,
    "liquidity": 15,
    "sfp": 20,
    "mss": 20,
    "displacement": 10,
    "volume": 5,
    "risk_reward": 10,
}


# ------------------------------------------------------------
# RISK MANAGEMENT
# ------------------------------------------------------------

DEFAULT_RISK_PERCENT = 1.0
MIN_RISK_PERCENT = 0.10
MAX_RISK_PERCENT = 5.0

# Минимальный RR для отправки сигнала (теперь от 1.6)
MIN_RR = 1.60

TP1_R_MULTIPLE = 1.0
TP2_R_MULTIPLE = 2.0

MAX_POSITION_BALANCE_MULTIPLE = 10.0


# ------------------------------------------------------------
# SIGNAL COOLDOWN
# ------------------------------------------------------------

SIGNAL_COOLDOWN_MINUTES = 60


# ------------------------------------------------------------
# GITHUB / LOCAL DATA
# ------------------------------------------------------------

DATA_DIR = "data"
SIGNAL_STATE_FILE = "data/signals_state.json"


# ------------------------------------------------------------
# TELEGRAM
# ------------------------------------------------------------

TELEGRAM_BOT_TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID_ENV = "TELEGRAM_CHAT_ID"


# ------------------------------------------------------------
# HTTP & API
# ------------------------------------------------------------

REQUEST_TIMEOUT = 15
MAX_RETRIES = 3
