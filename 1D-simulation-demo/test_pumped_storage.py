# -*- coding: utf-8 -*-
"""pumped_storage.py（揚水発電所の1Dモデル）の確認。実行: python -m pytest -q   （このフォルダで。pytest が必要）
確認しているのは、水の出入りの帳尻が合うこと、台数を変えても結論の向きがおかしくならないこと、記事の数値。"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from onedsim import simulate   # noqa: E402
from pumped_storage import DAY, PumpedStorage, make_demand, make_rain, roundtrip   # noqa: E402


def article_week(**kw):
    """記事の基準の1週間: 水曜（2日目）が猛暑日で 940 MW、土曜（5日目）に 30 mm/h × 12 時間の豪雨"""
    m = PumpedStorage(rain=make_rain(storm_day=5), demand=make_demand(heatwave_day=2, heatwave_MW=940.0), **kw)
    return simulate(m, 7 * DAY, 60.0, record_every=1)


def water_balance_error(r):
    """（上池 + 下池 + あふれた水 + 蒸発した水 − 降った雨）の変化 [m3]。0 なら水が湧きも消えもしていない。
    雨・蒸発・越流の量は、池の水量と同じ積分で積算した値（W_rain など）を使う"""
    total = r["V_u"] + r["V_l"] + r["W_spill_u"] + r["W_spill_l"] + r["W_evap"] - r["W_rain"]
    return total[-1] - total[0]


@pytest.fixture(scope="module")
def week():
    return article_week()


def test_water_is_conserved_in_the_article_week(week):
    assert abs(water_balance_error(week)) < 1.0


def test_lower_reservoir_overflow_is_counted_instead_of_deleted():
    """雨・高い発電要求・猛暑が重なって下池が満水になっても、あふれた水は数えられ、消えない"""
    m = PumpedStorage(rain=make_rain(storm_day=0, mm_per_h=50.0, hours=24.0),
                      demand=make_demand(peak_MW=1200.0, heatwave_day=2))
    r = simulate(m, 7 * DAY, 60.0, record_every=1)
    assert r["W_spill_l"][-1] > 1e5                       # このケースでは下池があふれる
    assert abs(water_balance_error(r)) < 1.0


def test_running_more_units_does_not_make_the_plant_worse_for_the_same_demand(week):
    """同じ需要なら、6 台の発電所は 2 台より効率が落ちたり、不足が増えたりしない（必要な台数だけ動かす）"""
    six = article_week(n_units=6)
    assert roundtrip(six) > roundtrip(week) - 0.005
    assert six["E_short"][-1] <= week["E_short"][-1] + 1.0


def test_a_heatwave_demand_of_zero_megawatts_is_taken_literally():
    d = make_demand(heatwave_day=0, heatwave_MW=0.0)
    assert d(14 * 3600.0) == 0.0                          # 0 日目の 14 時（昼ピーク）


def test_the_article_week_numbers(week):
    """記事 1 ページ目の数値（供給不足 約100 MWh、往復効率 72 %、最低水位 5.2 m、最小の落差 659 m）"""
    assert week["E_short"][-1] == pytest.approx(97.4, abs=0.5)
    assert roundtrip(week) == pytest.approx(0.720, abs=0.002)
    assert week["level_u"].min() == pytest.approx(5.19, abs=0.02)
    assert week["H"].min() == pytest.approx(659.0, abs=0.5)
