# -*- coding: utf-8 -*-
"""pumped_storage.py（揚水発電所の1Dモデル）の確認。実行: python -m pytest -q   （このフォルダで。pytest が必要）
確認しているのは、池の形と落差が公表水位に合っていること、水路の損失の決め方、水の出入りの帳尻が合うこと、
越流で失った電力量の換算、台数を変えても結論の向きがおかしくならないこと、記事の数値。"""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from onedsim import simulate   # noqa: E402
from pumped_storage import DAY, G, RHO, PumpedStorage, make_demand, make_rain, roundtrip   # noqa: E402

V = 12.67e6                                               # 有効貯水量 [m3]（上池・下池とも）


def article_week(**kw):
    """記事の基準の1週間: 水曜（2日目）が猛暑日で 940 MW、土曜（5日目）に 30 mm/h × 12 時間の豪雨"""
    m = PumpedStorage(rain=make_rain(storm_day=5), demand=make_demand(heatwave_day=2, heatwave_MW=940.0), **kw)
    return simulate(m, 7 * DAY, 60.0, record_every=1)


def water_balance_error(r):
    """（上池 + 下池 + あふれた水 + 蒸発した水 − 降った雨）の変化 [m3]。0 なら水が湧きも消えもしていない。
    雨・蒸発・越流の量は、池の水量と同じ積分で積算した値（W_rain など）を使う"""
    total = r["V_u"] + r["V_l"] + r["W_spill_u"] + r["W_spill_l"] + r["W_evap"] - r["W_rain"]
    return total[-1] - total[0]


def state(m, V_u, V_l):
    s = m.initial_state()
    s["V_u"], s["V_l"] = V_u, V_l
    return s


def chain_J_per_m3(m, H, q):
    """水 1 m3 を 1 台あたり流量 q で落としたときの発電量 [J/m3]。モデルの発電の式を、ここで独立に書いたもの"""
    return RHO * G * (H - m.k_loss * q ** 2) * float(m.turb_eta(q)) * m.eta_gen


@pytest.fixture(scope="module")
def week():
    return article_week()


# ----------------------------------------------------------------------------- 池の形と落差（公表水位）
@pytest.mark.parametrize("V_u, V_l, H", [
    (V, V, 683.6),            # 両池とも満水: 1,527.0 − 843.4
    (0.0, 0.0, 686.0),        # 両池とも最低水位: 1,500.0 − 814.0
    (V / 2, V / 2, 684.8),    # 両池とも有効貯水量の半分
    (V, 0.0, 713.0),          # 最大: 上池が満水、下池が最低水位
    (0.0, V, 656.6),          # 最小: 上池が最低水位、下池が満水
])
def test_static_head_is_the_difference_of_the_published_water_levels(V_u, V_l, H):
    assert PumpedStorage().head(V_u, V_l) == pytest.approx(H, abs=0.05)


def test_pond_levels_run_from_zero_to_the_published_usable_depths():
    """水位の出力は「最低水位からの高さ」: 上池 0〜27.0 m、下池 0〜29.4 m。面積は 有効貯水量 ÷ 利用水深"""
    m = PumpedStorage()
    lo, hi = m.outputs(0.0, state(m, 0.0, 0.0)), m.outputs(0.0, state(m, V, V))
    assert (lo["level_u"], lo["level_l"]) == (0.0, 0.0)
    assert hi["level_u"] == pytest.approx(27.0, abs=1e-9) and hi["level_l"] == pytest.approx(29.4, abs=1e-9)
    assert m.A_u == pytest.approx(V / 27.0) and m.A_l == pytest.approx(V / 29.4)
    assert m.A_u / 1e4 == pytest.approx(46.9, abs=0.05) and m.A_l / 1e4 == pytest.approx(43.1, abs=0.05)   # ha


def test_the_old_single_area_and_base_head_arguments_are_rejected():
    for kw in (dict(A_res=4.0e5), dict(H_base=680.0)):
        with pytest.raises(TypeError):
            PumpedStorage(**kw)


@pytest.mark.parametrize("kw", [
    dict(z_u=(1500.0, 1527.0)),            # 満水位が最低水位より低い
    dict(z_l=(843.4, 843.4)),              # 利用水深が 0
    dict(z_u=(840.0, 830.0)),              # 上池が下池より低い
    dict(z_u=(1490.0, 1470.0)),            # 両池が半分のときの落差が 653 m を下回り、損失の決め方が成り立たない
    dict(k_loss=-0.001),
])
def test_impossible_geometry_or_loss_is_an_error(kw):
    with pytest.raises(ValueError):
        PumpedStorage(**kw)


# ----------------------------------------------------------------------------- 水路の損失の決め方（仮定）と定格点
def test_waterway_loss_gives_653_m_effective_head_at_85_m3s_with_both_ponds_half_full():
    m = PumpedStorage()
    H_half = m.head(V / 2, V / 2)
    assert H_half - m.k_loss * 85.0 ** 2 == pytest.approx(653.0, abs=1e-9)
    assert m.k_loss * 85.0 ** 2 == pytest.approx(31.8, abs=0.05)
    assert m.k_loss == pytest.approx(0.004401, abs=5e-7)


def test_waterway_loss_is_computed_from_the_geometry_and_can_be_overridden():
    m = PumpedStorage(z_u=(1537.0, 1510.0))                # 上池を 10 m 高くすると、損失も 10 m 増えて 653 m を保つ
    assert m.head(V / 2, V / 2) - m.k_loss * 85.0 ** 2 == pytest.approx(653.0, abs=1e-9)
    assert m.k_loss * 85.0 ** 2 == pytest.approx(41.8, abs=0.05)
    assert PumpedStorage(k_loss=0.003).k_loss == 0.003
    assert PumpedStorage(k_loss=0.0).k_loss == 0.0         # 0 は「損失なし」で、決め直しの合図ではない


def test_rated_point_reproduces_the_published_turbine_and_generator_output():
    """両池が半分・85 m3/s（有効落差 653 m）で、水車出力 482 MW、発電機出力 470 MW（1台あたり）"""
    m = PumpedStorage()
    H_half, Qm = m.head(V / 2, V / 2), m.Q_turb_max
    assert Qm == 85.0
    P_gen = chain_J_per_m3(m, H_half, Qm) * Qm / 1e6
    assert P_gen / m.eta_gen == pytest.approx(482.0, abs=0.5)
    assert P_gen == pytest.approx(470.0, abs=0.5) and P_gen <= m.P_unit
    assert m.turbine_flow(P_gen, H_half) == pytest.approx(Qm, abs=1e-6)


# ----------------------------------------------------------------------------- 水の収支
def test_water_is_conserved_in_the_article_week(week):
    assert abs(water_balance_error(week)) < 1.0


def test_lower_reservoir_overflow_is_counted_instead_of_deleted():
    """雨・高い発電要求・猛暑が重なって下池が満水になっても、あふれた水は数えられ、消えない"""
    m = PumpedStorage(rain=make_rain(storm_day=0, mm_per_h=50.0, hours=24.0),
                      demand=make_demand(peak_MW=1200.0, heatwave_day=2))
    r = simulate(m, 7 * DAY, 60.0, record_every=1)
    assert r["W_spill_l"][-1] > 1e5                       # このケースでは下池があふれる
    assert abs(water_balance_error(r)) < 1.0


def test_each_pond_evaporates_from_its_own_area_and_only_the_upper_pond_collects_rain():
    """運転していないときの水量の変化: 上池 = 雨 − 蒸発（上池の面積）、下池 = − 蒸発（下池の面積）。下池に自然流入はない（仮定）"""
    m = PumpedStorage(rain=lambda t: 30.0, demand=lambda t: 0.0)
    d = m.derivatives(0.0, m.initial_state())
    evap = 4.0 / 1000 / DAY                               # 4 mm/日 [m/s]
    rain = 30.0 / 1000 / 3600 * 6.2e6 * 0.8               # 30 mm/h × 集水域 6.2 km2 × 流出率 0.8 [m3/s]
    assert d["V_u"] == pytest.approx(rain - evap * V / 27.0, rel=1e-12)
    assert d["V_l"] == pytest.approx(-evap * V / 29.4, rel=1e-12)
    assert d["W_evap"] == pytest.approx(evap * (V / 27.0 + V / 29.4), rel=1e-12)
    assert d["W_rain"] == pytest.approx(rain, rel=1e-12)


# 池の範囲（0〜V_max）の端で、clamp() が水を作ったり消したりしないこと（7 日間、独立レビューの再現ケース）
@pytest.mark.parametrize("kw, dt", [
    (dict(V_u0_frac=0.0, demand=lambda t: 0.0), 60.0),                          # 空の上池から蒸発を数えない
    (dict(V_u0_frac=1.0, rain=lambda t: 30.0, demand=lambda t: 0.0), 60.0),     # 空の下池から蒸発を数えない
    (dict(rain=lambda t: 50.0, demand=lambda t: -900.0), 300.0),                # 1 ステップで V_u が V_max を越えても、越えた分を数える
])
def test_water_balance_holds_when_a_pond_hits_its_limit(kw, dt):
    m = PumpedStorage(**kw)
    r = simulate(m, 7 * DAY, dt, record_every=1)
    assert abs(water_balance_error(r)) < 1e-4          # 丸め誤差の積み重ね（池の水量 1e7 m3 の桁で 1e-6〜1e-5 m3）より上、バグ（1e3〜1e4 m3）よりずっと下
    assert r["V_u"].min() >= 0.0 and r["V_l"].min() >= 0.0
    assert r["V_u"].max() <= m.V_max and r["V_l"].max() <= m.V_max


def test_water_pushed_past_a_full_pond_in_one_step_is_counted_as_spill():
    m = PumpedStorage()
    s = state(m, V + 5000.0, V - 5000.0)
    c = m.clamp(dict(s))
    assert c["V_u"] == V and c["W_spill_u"] == pytest.approx(5000.0, abs=1e-9)
    assert c["E_spill"] == pytest.approx(5000.0 * chain_J_per_m3(m, m.head(V, V - 5000.0), m.Q_turb_max) / 3.6e9, rel=1e-12)
    c = m.clamp(dict(state(m, V - 5000.0, V + 5000.0)))          # 下池: 越えた分は下流への放流として数える（発電機会の換算はしない）
    assert c["V_l"] == V and c["W_spill_l"] == pytest.approx(5000.0, abs=1e-9) and c["E_spill"] == 0.0


def test_evaporation_an_empty_pond_could_not_supply_is_taken_back_out_of_the_count():
    m = PumpedStorage()
    s = state(m, -30.0, V / 2)
    s["W_evap"] = 100.0
    c = m.clamp(s)
    assert c["V_u"] == 0.0 and c["W_evap"] == pytest.approx(70.0, abs=1e-12)


# ----------------------------------------------------------------------------- 越流で失った電力量、発電可能量
@pytest.mark.parametrize("V_l", [0.05 * V, 0.5 * V, 0.9 * V])
def test_spilled_energy_per_m3_is_the_generating_chain_at_maximum_flow(V_l):
    """あふれた水 1 m3 の電力量 = ρ g (H − k Qmax²) η水車(Qmax) η発電機。モデルのいちばん良い発電の点を超えない"""
    m = PumpedStorage(rain=lambda t: 50.0, demand=lambda t: 0.0)
    d = m.derivatives(0.0, state(m, V, V_l))
    assert d["W_spill_u"] > 50.0 and d["V_u"] == 0.0       # 満水の上池に雨が入り、その分があふれている
    per_m3 = d["E_spill"] * 3.6e9 / d["W_spill_u"]         # [J/m3]
    H = m.head(V, V_l)
    assert per_m3 == pytest.approx(chain_J_per_m3(m, H, m.Q_turb_max), rel=1e-12)
    best = max(chain_J_per_m3(m, H, q) for q in np.linspace(1.0, m.Q_turb_max, 400))
    assert per_m3 <= best
    assert per_m3 < 0.88 * RHO * G * H                     # 静落差に一律の効率 0.9 を掛けた値にはならない


def test_spilled_energy_of_a_whole_run_does_not_exceed_the_best_generating_chain():
    m = PumpedStorage(rain=make_rain(storm_day=5, mm_per_h=80.0, hours=24.0))
    r = simulate(m, 7 * DAY, 60.0, record_every=1)
    assert r["W_spill_u"][-1] > 1e6
    best = max(chain_J_per_m3(m, r["H"].max(), q) for q in np.linspace(1.0, m.Q_turb_max, 400))
    assert 0.0 < r["E_spill"][-1] * 3.6e9 <= best * r["W_spill_u"][-1]


def test_stored_energy_uses_the_same_chain_as_spilled_energy():
    m = PumpedStorage()
    o = m.outputs(0.0, state(m, 0.7 * V, 0.3 * V))
    expect = (0.7 * V - m.V_min) * chain_J_per_m3(m, m.head(0.7 * V, 0.3 * V), m.Q_turb_max) / 3.6e9
    assert o["E_stored"] == pytest.approx(expect, rel=1e-12)


# ----------------------------------------------------------------------------- ポンプ水車の表と公表仕様の関係
def pump_points(m, n=400):
    """雨のない運転（V_u + V_l = V_max）で、両池とも運用の下限より上にある範囲の、ポンプの動作点 (流量 m3/s, 全揚程 m, 軸入力 MW)"""
    pts = []
    for V_u in np.linspace(m.V_min, m.V_max - m.V_min, n):
        Q = m.pump_operating_point(m.head(V_u, m.V_max - V_u))
        H_p = float(m.pump_H(Q))
        pts.append((Q, H_p, RHO * G * Q * H_p / float(m.pump_eta(Q)) / 1e6))
    return np.array(pts)


def test_pump_total_head_stays_inside_the_published_range():
    """公表の全揚程は 677〜728 m（東芝レビュー 2006）"""
    H_p = pump_points(PumpedStorage())[:, 1]
    assert 677.0 <= H_p.min() and H_p.max() <= 728.0


def test_pump_shaft_power_is_within_half_a_percent_of_the_published_maximum(week):
    """公表の最大軸入力は 464 MW（東芝レビュー 2006）。仮定したポンプ曲線は公表値に合わせ込んでいないので、上池の水位が運用の下限に近いときは
    これを少し上回る（1.4 MW、0.3 %）。記事の 1 週間の運転では、その水位まで下がらないので、464 MW 以内に収まる"""
    m = PumpedStorage()
    assert 464.0 < pump_points(m)[:, 2].max() < 464.0 * 1.005
    units = np.where(week["P_pump"] > 1.5 * m.P_pump_nominal, 2, 1)     # その時刻に揚水している台数（週末は 1 台）
    assert (week["P_pump"] / units * m.eta_motor).max() <= 464.0        # 1 台あたりの軸入力 = 電動機の入力 × 電動機効率


# ----------------------------------------------------------------------------- 運用
def test_running_more_units_does_not_make_the_plant_worse_for_the_same_demand(week):
    """同じ需要なら、6 台の発電所は 2 台より効率が落ちたり、不足が増えたりしない（必要な台数だけ動かす）"""
    six = article_week(n_units=6)
    assert roundtrip(six) > roundtrip(week) - 0.005
    assert six["E_short"][-1] <= week["E_short"][-1] + 1.0


def test_a_heatwave_demand_of_zero_megawatts_is_taken_literally():
    d = make_demand(heatwave_day=0, heatwave_MW=0.0)
    assert d(14 * 3600.0) == 0.0                          # 0 日目の 14 時（昼ピーク）


def test_a_larger_waterway_loss_means_more_shortage_and_a_lower_round_trip(week):
    more = article_week(k_loss=PumpedStorage().k_loss + 10.0 / 85.0 ** 2)   # 85 m3/s での損失を 10 m 増やす
    assert more["E_short"][-1] > week["E_short"][-1] and roundtrip(more) < roundtrip(week)


def test_levels_and_head_stay_inside_the_range_the_geometry_allows(week):
    assert 0.0 <= week["level_u"].min() and week["level_u"].max() <= 27.0
    assert 0.0 <= week["level_l"].min() and week["level_l"].max() <= 29.4
    assert 656.6 <= week["H"].min() and week["H"].max() <= 713.0


def test_the_article_week_numbers(week):
    """記事 1 ページ目の数値（供給不足、往復効率、上池の最低水位、最小の落差）"""
    assert week["E_short"][-1] == pytest.approx(81.85, abs=0.05)
    assert roundtrip(week) == pytest.approx(0.7162, abs=0.0005)
    assert week["level_u"].min() == pytest.approx(3.72, abs=0.01)
    assert week["H"].min() == pytest.approx(664.42, abs=0.05)
    assert week["E_spill"][-1] == 0.0                      # この週は上池があふれない
