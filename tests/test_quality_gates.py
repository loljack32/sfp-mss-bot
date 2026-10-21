# ============================================================
# QUALITY GATE TESTS
# ============================================================
# Проверяют, что доработанные фильтры действительно отсекают
# слабые, устаревшие и нереалистичные сетапы.
#
# Запуск из корня репозитория:  python -m unittest discover tests
# ============================================================

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from core.sfp import SFPSetup
from core.mss import MSSSetup, detect_bullish_mss
from core.structure import SwingPoint, StructureState
from core.filters import (
    evaluate_setup,
    check_local_structure,
    find_nearest_liquidity_target,
)
from core.signals import generate_signal
from core.okx import normalize_instrument_id

import main as main_module


# ============================================================
# FIXTURES
# ============================================================

def make_df(
    n: int = 80,
    liquidity_price: float = None,
    liquidity_index: int = 40,
) -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [100.0] * n,
            "high": [101.0] * n,
            "low": [99.0] * n,
            "close": [100.0] * n,
            "volume": [1000.0] * n,
            "confirm": ["1"] * n,
        }
    )
    if liquidity_price is not None and liquidity_index < n:
        # Крупный свинг (окно 10/10): единственный экстремум выше всех
        # соседей - уровень ликвидности для evaluate_setup.
        df.loc[liquidity_index, "high"] = liquidity_price
    return df


def make_sfp(df: pd.DataFrame, sweep_index: int, **overrides) -> SFPSetup:
    params = dict(
        direction="LONG",
        level=99.5,
        sweep_extreme=99.0,
        sweep_index=sweep_index,
        sweep_timestamp=df.iloc[sweep_index]["timestamp"],
        entry_reference=100.0,
        atr=1.0,
        sweep_distance=0.5,
        sweep_atr_ratio=0.5,
        reclaim_ratio=0.85,
        candle_body_ratio=0.7,
        score=90.0,
        source_swing_index=10,
        source_swing_label="LL",
    )
    params.update(overrides)
    return SFPSetup(**params)


def make_mss(df: pd.DataFrame, break_index: int, **overrides) -> MSSSetup:
    params = dict(
        direction="LONG",
        structure_level=99.8,
        structure_label="LH",
        structure_index=break_index - 8,
        break_index=break_index,
        break_timestamp=df.iloc[break_index]["timestamp"],
        break_price=100.2,
        displacement=0.8,
        displacement_atr_ratio=0.8,
        body_ratio=0.8,
        candle_range=1.0,
        score=90.0,
    )
    params.update(overrides)
    return MSSSetup(**params)


def make_htf_state() -> StructureState:
    return StructureState(
        trend="BULLISH",
        last_high=None,
        previous_high=None,
        last_low=None,
        previous_low=None,
        last_swing=None,
    )


def far_liquidity_structure(df: pd.DataFrame, price: float = 130.0):
    return [
        SwingPoint(
            index=10,
            timestamp=df.iloc[10]["timestamp"],
            price=price,
            kind="HIGH",
            label="HH",
        )
    ]


# ============================================================
# TESTS
# ============================================================

class TestQualityGates(unittest.TestCase):
    def test_valid_fresh_setup_passes(self):
        df = make_df(liquidity_price=130.0)
        break_index = len(df) - 2
        df.loc[break_index, "volume"] = 2000.0

        result = evaluate_setup(
            df=df,
            sfp=make_sfp(df, break_index - 5),
            mss=make_mss(df, break_index),
            structure=far_liquidity_structure(df),
            htf_state=make_htf_state(),
        )

        self.assertTrue(result.passed, msg=f"reasons: {result.reasons}")

    def test_stale_setup_rejected(self):
        df = make_df(liquidity_price=130.0)
        break_index = 60  # 19 баров назад при лимите свежести 2

        result = evaluate_setup(
            df=df,
            sfp=make_sfp(df, break_index - 5),
            mss=make_mss(df, break_index),
            structure=far_liquidity_structure(df),
            htf_state=make_htf_state(),
        )

        self.assertFalse(result.passed)
        self.assertTrue(
            any("Stale setup" in r for r in result.reasons),
            msg=f"reasons: {result.reasons}",
        )

    def test_weak_volume_is_hard_rejection(self):
        df = make_df(liquidity_price=130.0)
        break_index = len(df) - 2
        df.loc[break_index, "volume"] = 500.0  # 0.5x от среднего

        result = evaluate_setup(
            df=df,
            sfp=make_sfp(df, break_index - 5),
            mss=make_mss(df, break_index),
            structure=far_liquidity_structure(df),
            htf_state=make_htf_state(),
        )

        self.assertFalse(result.passed)
        self.assertTrue(
            any("Volume confirmation too weak" in r for r in result.reasons),
            msg=f"reasons: {result.reasons}",
        )

    def test_hh_break_is_not_a_bullish_mss(self):
        df = make_df()
        ok, _ = check_local_structure(
            "LONG",
            make_sfp(df, 70),
            make_mss(df, 75, structure_label="HH"),
            [],
        )
        self.assertFalse(ok)

        ok, _ = check_local_structure(
            "LONG",
            make_sfp(df, 70),
            make_mss(df, 75, structure_label="LH"),
            [],
        )
        self.assertTrue(ok)

    def test_liquidity_target_selection(self):
        df = make_df()
        too_close = SwingPoint(
            index=10,
            timestamp=df.iloc[10]["timestamp"],
            price=101.0,  # RR около 0.95 - целью быть не может
            kind="HIGH",
            label="HH",
        )
        valid = SwingPoint(
            index=12,
            timestamp=df.iloc[12]["timestamp"],
            price=130.0,
            kind="HIGH",
            label="HH",
        )

        # Слишком близкий уровень пропускается, целью становится
        # первый уровень с приемлемым RR.
        target = find_nearest_liquidity_target(
            direction="LONG",
            current_index=79,
            current_price=100.0,
            structure=[too_close, valid],
            stop_loss=98.95,
            min_rr=1.6,
        )
        self.assertIsNotNone(target)
        self.assertEqual(target.price, 130.0)

        # Если приемлемого уровня нет вообще - цели нет.
        target = find_nearest_liquidity_target(
            direction="LONG",
            current_index=79,
            current_price=100.0,
            structure=[too_close],
            stop_loss=98.95,
            min_rr=1.6,
        )
        self.assertIsNone(target)


class TestMSSLevelSelection(unittest.TestCase):
    def test_mss_uses_lh_after_swept_low_not_any_high(self):
        df = make_df(40)
        # Сильная бычья свеча пробоя на индексе 25.
        df.loc[25, ["open", "high", "low", "close"]] = [100.0, 106.5, 99.5, 106.0]

        structure = [
            SwingPoint(
                index=5,
                timestamp=df.iloc[5]["timestamp"],
                price=90.0,
                kind="LOW",
                label="LL",
            ),
            SwingPoint(
                index=8,
                timestamp=df.iloc[8]["timestamp"],
                price=110.0,
                kind="HIGH",
                label="HH",
            ),
            SwingPoint(
                index=12,
                timestamp=df.iloc[12]["timestamp"],
                price=105.0,
                kind="HIGH",
                label="LH",
            ),
        ]

        sfp = make_sfp(
            df,
            sweep_index=20,
            level=90.0,
            sweep_extreme=89.5,
            source_swing_index=5,
        )

        mss = detect_bullish_mss(df, sfp, structure)

        self.assertIsNotNone(mss)
        self.assertEqual(mss.structure_label, "LH")
        self.assertEqual(mss.structure_level, 105.0)


class TestRiskGuards(unittest.TestCase):
    def _kwargs(self, df):
        break_index = len(df) - 2
        df.loc[break_index, "volume"] = 2000.0
        return dict(
            symbol="BTC-USDT-SWAP",
            timeframe="15m",
            df=df,
            sfp=make_sfp(df, break_index - 5),
            mss=make_mss(df, break_index),
            structure=far_liquidity_structure(df),
            htf_state=make_htf_state(),
            balance=10000.0,
        )

    def test_margin_above_balance_rejected(self):
        df = make_df(liquidity_price=130.0)
        # Риск 5% при плече 1x: маржа = весь номинал >> баланса.
        signal = generate_signal(
            risk_percent=5.0, leverage=1, **self._kwargs(df)
        )
        self.assertIsNone(signal)

        df = make_df(liquidity_price=130.0)
        signal = generate_signal(
            risk_percent=5.0, leverage=10, **self._kwargs(df)
        )
        self.assertIsNotNone(signal)

    def test_tp2_uses_2r_multiple(self):
        # Цели v2: TP1 = 1R, TP2 = 2R. Ликвидность на 2.5R -
        # TP2 встаёт на 2R, не дотягиваясь до пула.
        df = make_df(liquidity_price=102.625)
        kwargs = self._kwargs(df)

        signal = generate_signal(risk_percent=1.0, leverage=10, **kwargs)

        self.assertIsNotNone(signal)
        self.assertAlmostEqual(signal.rr_tp1, 1.0, places=2)
        self.assertAlmostEqual(signal.rr_tp2, 2.0, places=2)
        self.assertAlmostEqual(signal.tp2, 102.10, places=2)


class TestDedupeAndInstruments(unittest.TestCase):
    def test_setup_never_resent_but_new_setup_allowed(self):
        df = make_df()
        sfp = make_sfp(df, 70)
        mss = make_mss(df, 75)

        state = {}
        main_module.record_signal_sent(
            "BTC-USDT-SWAP", "LONG", state, sfp=sfp, mss=mss
        )

        # Тот же сетап блокируется даже при нулевом кулдауне по символу.
        self.assertTrue(
            main_module.is_on_cooldown(
                "BTC-USDT-SWAP", "LONG", state,
                cooldown_minutes=0, sfp=sfp, mss=mss,
            )
        )

        # Другой сетап при нулевом кулдауне не блокируется.
        other_mss = make_mss(df, 76)
        self.assertFalse(
            main_module.is_on_cooldown(
                "BTC-USDT-SWAP", "LONG", state,
                cooldown_minutes=0, sfp=sfp, mss=other_mss,
            )
        )

    def test_normalize_instrument_id(self):
        self.assertEqual(
            normalize_instrument_id("BTC-USDT"), "BTC-USDT-SWAP"
        )
        self.assertEqual(
            normalize_instrument_id("BTC-USDT-SWAP"), "BTC-USDT-SWAP"
        )


class TestSituationalFilters(unittest.TestCase):
    """Фильтры по итогам разбора убыточных сделок (бэктест 6 мес.)."""

    def test_chase_entry_rejected(self):
        df = make_df(liquidity_price=160.0)
        n = len(df)
        # Цена уже прошла +30 при ATR=1 за последние 21 бар - погоня.
        for i in range(n - 21, n):
            c = 100.0 + (i - (n - 21)) * 1.5
            df.loc[i, "close"] = c
            df.loc[i, "open"] = c
            df.loc[i, "high"] = c + 0.5
            df.loc[i, "low"] = c - 0.5
        break_index = n - 2
        df.loc[break_index, "volume"] = 2000.0

        result = evaluate_setup(
            df=df,
            sfp=make_sfp(df, break_index - 5),
            mss=make_mss(df, break_index),
            structure=far_liquidity_structure(df, price=160.0),
            htf_state=make_htf_state(),
        )

        self.assertFalse(result.passed)
        self.assertTrue(
            any("Entry chase" in r for r in result.reasons),
            msg=f"reasons: {result.reasons}",
        )

    def _evaluate_short(self, regime_uptrend):
        df = make_df()
        break_index = len(df) - 2
        df.loc[break_index, "volume"] = 2000.0
        structure = [
            SwingPoint(
                index=10,
                timestamp=df.iloc[10]["timestamp"],
                price=70.0,
                kind="LOW",
                label="LL",
            )
        ]

        return evaluate_setup(
            df=df,
            sfp=make_sfp(
                df,
                break_index - 5,
                direction="SHORT",
                level=100.5,
                sweep_extreme=101.0,
                source_swing_label="HH",
            ),
            mss=make_mss(
                df,
                break_index,
                direction="SHORT",
                structure_level=100.2,
                structure_label="HL",
                break_price=99.8,
            ),
            structure=structure,
            htf_state=make_htf_state(),
            regime_uptrend=regime_uptrend,
        )

    def test_short_not_blocked_by_direction_gate(self):
        # Шорты разрешены конфигом: блокировать их может режим
        # рынка или другие гейты, но не сам факт направления.
        result = self._evaluate_short(regime_uptrend=False)
        self.assertFalse(
            any("Short signals disabled" in r for r in result.reasons),
            msg=f"reasons: {result.reasons}",
        )

    def test_short_blocked_in_uptrend_regime(self):
        result = self._evaluate_short(regime_uptrend=True)
        self.assertFalse(result.passed)
        self.assertTrue(
            any("Market regime" in r for r in result.reasons),
            msg=f"reasons: {result.reasons}",
        )


class TestHtfDistanceWindow(unittest.TestCase):
    """Окно дистанции до EMA50 на 4H: [-0.5, +1.5] ATR."""

    def _evaluate(self, dist):
        df = make_df(liquidity_price=130.0)
        break_index = len(df) - 2
        df.loc[break_index, "volume"] = 2000.0
        return evaluate_setup(
            df=df,
            sfp=make_sfp(df, break_index - 5),
            mss=make_mss(df, break_index),
            structure=far_liquidity_structure(df),
            htf_state=make_htf_state(),
            htf_dist_atr=dist,
        )

    def test_inside_window_passes(self):
        result = self._evaluate(0.7)
        self.assertTrue(result.passed, msg=f"reasons: {result.reasons}")

    def test_overheated_htf_rejected(self):
        result = self._evaluate(2.8)
        self.assertFalse(result.passed)
        self.assertTrue(
            any("HTF distance" in r for r in result.reasons),
            msg=f"reasons: {result.reasons}",
        )

    def test_deep_countertrend_rejected(self):
        result = self._evaluate(-1.9)
        self.assertFalse(result.passed)
        self.assertTrue(
            any("HTF distance" in r for r in result.reasons),
            msg=f"reasons: {result.reasons}",
        )


class TestRegimeFilter(unittest.TestCase):
    """Фильтр режима рынка: лонг только при бенчмарке выше EMA."""

    def _evaluate(self, uptrend):
        df = make_df(liquidity_price=130.0)
        break_index = len(df) - 2
        df.loc[break_index, "volume"] = 2000.0
        return evaluate_setup(
            df=df,
            sfp=make_sfp(df, break_index - 5),
            mss=make_mss(df, break_index),
            structure=far_liquidity_structure(df),
            htf_state=make_htf_state(),
            htf_dist_atr=0.7,
            regime_uptrend=uptrend,
        )

    def test_uptrend_passes(self):
        result = self._evaluate(True)
        self.assertTrue(result.passed, msg=f"reasons: {result.reasons}")

    def test_downtrend_blocks_long(self):
        result = self._evaluate(False)
        self.assertFalse(result.passed)
        self.assertTrue(
            any("Market regime" in r for r in result.reasons),
            msg=f"reasons: {result.reasons}",
        )

    def test_unknown_regime_passes(self):
        result = self._evaluate(None)
        self.assertTrue(result.passed, msg=f"reasons: {result.reasons}")


class TestBreakevenOutcome(unittest.TestCase):
    def test_breakeven_exit_after_plus_1r(self):
        from types import SimpleNamespace

        from backtest import _simulate_outcome

        signal = SimpleNamespace(
            entry=100.0,
            stop_loss=95.0,
            tp1=110.0,
            tp2=115.0,
            rr_tp1=2.0,
            rr_tp2=3.0,
            direction="LONG",
        )
        df = pd.DataFrame(
            [
                (101.0, 99.0, 100.0),   # сигнальный бар
                (106.0, 99.0, 105.0),   # +1.2R: безубыток взводится
                (103.0, 99.5, 102.0),   # откат в точку входа: выход BE
                (112.0, 101.0, 111.0),  # недостижим
            ],
            columns=["high", "low", "close"],
        )

        out = _simulate_outcome(df, signal, 0, 10)

        self.assertEqual(out["exit"], "BE")
        self.assertAlmostEqual(out["result_r"], -0.05, places=6)


if __name__ == "__main__":
    unittest.main()
