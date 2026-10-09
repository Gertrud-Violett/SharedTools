# -*- coding: utf-8 -*-
"""gg_rocket.py（GGサイクルエンジンの1Dモデル）の起動・停止の確認。
実行: python -m pytest -q   （このフォルダで。pytest が必要）
確認しているのは「仕様どおりか」ではなく「物理として筋が通っているか」: 定常値、保存則、手順を変えたときの向き。"""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from onedsim import simulate          # noqa: E402
import gg_rocket                      # noqa: E402
from gg_rocket import P_ATM, T_AD, GGEngine, crossing_time, design_point, gg_gas, turbine_eta_ratio   # noqa: E402

trapezoid = getattr(np, "trapezoid", None) or np.trapz   # 台形則の積分。NumPy 2 では trapezoid、1.x では trapz

D = design_point()
RPM_D = D["rpm"]                 # 設計回転数 [rpm]
PD_D = D["Pd"] / 1e6             # 設計のポンプ吐出圧 [MPa]
PC_D = D["Pc"] / 1e6             # 設計の燃焼圧 [MPa]
F_D = D["F_sl"] / 1e3            # 設計の海面推力 [kN]
T_GG_D = D["T_gg"]               # 設計の GG 出口温度 [K]
DT = 2e-4                        # 時間刻み [s]（記事の起動計算と同じ）
T_CUT = 1.5                      # 停止の指令を出す時刻 [s]（起動が落ち着いたあと）


def run(t_end, record_every=5, **kw):
    return simulate(GGEngine(**kw), t_end, DT, record_every=record_every)


def after(r, t0):
    """時刻 t0 以降だけを取り出すマスク"""
    return r["t"] >= t0 - 1e-12


# ----------------------------------------------------------------------------- 応答時間の読み取り
def test_crossing_time_interpolates_between_samples_on_the_way_up():
    assert crossing_time([0.0, 1.0, 2.0], [0.0, 10.0, 20.0], 15.0) == pytest.approx(1.5)


def test_crossing_time_interpolates_between_samples_on_the_way_down():
    assert crossing_time([0.0, 1.0, 2.0], [20.0, 10.0, 0.0], 5.0, rising=False) == pytest.approx(1.5)


def test_crossing_time_is_none_when_the_level_is_never_reached():
    assert crossing_time([0.0, 1.0], [0.0, 1.0], 2.0) is None
    assert crossing_time([0.0, 1.0], [1.0, 0.5], 0.2, rising=False) is None


def test_crossing_time_is_the_first_sample_when_already_past_the_level():
    assert crossing_time([3.0, 4.0], [5.0, 6.0], 1.0) == 3.0


# ----------------------------------------------------------------------------- 燃焼ガスの表
def test_gg_gas_properties_inside_the_textbook_table_are_the_textbook_values():
    """Huzel & Huang の表の行（O/F 0.408: cp 0.653 Btu/lb·°F = 2,734 J/kg/K、比熱比 1.124）"""
    cp, gam, R = gg_gas(0.408 / 1.408)
    assert cp == pytest.approx(2734.0, abs=0.5) and gam == pytest.approx(1.124, abs=1e-4)
    assert R == pytest.approx(cp * (gam - 1) / gam, rel=1e-12)


def test_gg_gas_properties_beyond_the_textbook_table_follow_the_equilibrium_values():
    """教科書の表より上の O/F では、端の値のままにせず、化学平衡計算の分子量と比熱比を使う（O/F 3 で分子量 24.75、比熱比 1.132）"""
    cp, gam, R = gg_gas(3.0 / 4.0)
    assert R == pytest.approx(8314.46 / 24.75, rel=1e-6) and gam == pytest.approx(1.132, abs=1e-6)
    assert cp == pytest.approx(gam * R / (gam - 1), rel=1e-12)


def test_gg_gas_at_the_oxygen_end_is_nearly_oxygen():
    """ほとんど酸素だけのガスのガス定数は、酸素の値（8314.46 ÷ 32.0 = 259.8 J/kg/K）に近い"""
    cp, gam, R = gg_gas(0.99)
    assert R == pytest.approx(8314.46 / 31.999, rel=0.01) and gam > 1.25


def test_gg_gas_properties_are_continuous_at_the_end_of_the_textbook_table():
    lo, hi = gg_gas(0.516 / 1.516 - 1e-9), gg_gas(0.516 / 1.516 + 1e-9)
    assert hi == pytest.approx(lo, rel=1e-6)


def test_embedded_combustion_tables_match_rocketcea():
    """RocketCEA が入っている環境でだけ実行する（なければスキップ）。gg_rocket.py に埋め込んだ表が、化学平衡計算の結果と丸めの範囲で一致する"""
    cea_obj = pytest.importorskip("rocketcea.cea_obj")
    import cea_tables
    cea = cea_obj.CEA_Obj(oxName="LOX", fuelName="RP-1")
    pc = cea_tables.PC_BAR * cea_tables.PSI_PER_BAR
    assert cea_tables.OFS == gg_rocket._OF_CEA
    for of in cea_tables.OFS:
        ivac, cstar_ft, tc_R = cea.get_IvacCstrTc(Pc=pc, MR=of, eps=cea_tables.EPS)
        assert float(T_AD(float(of))) == pytest.approx(tc_R / 1.8, abs=0.51)
        assert float(gg_rocket.CSTAR(float(of))) == pytest.approx(cstar_ft * 0.3048, abs=0.51)
        mw, gam = cea.get_Chamber_MolWt_gamma(Pc=cea_tables.GG_PRESSURE_RATIO * pc, MR=of, eps=cea_tables.EPS)
        assert float(gg_rocket.GG_R(float(of))) == pytest.approx(8314.46 / mw, rel=5e-4)
        assert float(gg_rocket.GG_GAMMA(float(of))) == pytest.approx(gam, abs=5.1e-4)
    for of, cf in zip(gg_rocket.CF_VAC.x, gg_rocket.CF_VAC.y):
        ivac, cstar_ft, _ = cea.get_IvacCstrTc(Pc=pc, MR=float(of), eps=cea_tables.EPS)
        assert cf == pytest.approx(ivac * 9.80665 / (cstar_ft * 0.3048), abs=5.1e-4)


# ----------------------------------------------------------------------------- タービンの効率マップ
@pytest.mark.parametrize("x", [0.1, 0.6, 0.9, 0.99, 1.01, 1.128, 1.6])
def test_turbine_efficiency_map_is_the_parabola_at_any_speed_ratio(x):
    """衝動タービンの効率（設計点との比）は速度比 x の放物線 2x − x²。0.25 刻みの点のあいだでも式どおりの値になる"""
    assert turbine_eta_ratio(x) == pytest.approx(2 * x - x * x, rel=1e-12)


def test_turbine_efficiency_map_has_a_smooth_peak_at_the_design_point():
    """設計点（x = 1）が頂点。左右対称で、そこで折れ曲がらない（頂点からの落ち方は 2 次）"""
    d = 1e-3
    assert turbine_eta_ratio(1.0) == 1.0
    assert turbine_eta_ratio(1 - d) == pytest.approx(turbine_eta_ratio(1 + d), rel=1e-12)
    assert 1.0 - turbine_eta_ratio(1 + d) == pytest.approx(d * d, rel=1e-6)


def test_turbine_stall_torque_is_twice_the_design_torque():
    """ガスの流量と熱落差が同じなら、止まっているとき（x → 0）のトルクは設計点の 2 倍（衝動タービンの理論）。
    トルク ∝ 効率 ÷ 回転数 なので、設計点との比は (η/η_d) / x"""
    x = 1e-6
    assert turbine_eta_ratio(x) / x == pytest.approx(2.0, rel=1e-5)


def test_turbine_efficiency_map_is_zero_outside_zero_to_twice_the_design_speed_ratio():
    for x in (-0.5, 0.0, 2.0, 3.0):
        assert turbine_eta_ratio(x) == 0.0


# ----------------------------------------------------------------------------- 起動
@pytest.fixture(scope="module")
def base():
    return run(3.0)


def test_old_start_sequence_regression():
    """主弁を 0.3〜0.9 s で開く以前の手順の結果が、意図せず変わっていないことの確認（回帰テスト）。
    値は、起動トルクが回転数とともに下がる版（2026-10）で計算したもの。モデルの式を変えたら、確かめたうえで更新する"""
    r = run(3.0, main_valve_t=(0.30, 0.90))
    assert r["rpm"].max() == pytest.approx(44967.94248662225, rel=1e-9)
    assert r["Pd_o_MPa"].max() == pytest.approx(24.594729271064736, rel=1e-9)
    assert r["t"][np.argmax(r["F_kN"] > 0.9 * r["F_kN"][-1])] == pytest.approx(0.586, abs=1e-9)


def test_base_start_settles_at_the_design_point(base):
    assert base["F_kN"][-1] == pytest.approx(F_D, abs=0.01)
    assert base["rpm"][-1] == pytest.approx(RPM_D, abs=1.0)
    assert base["Pc_MPa"][-1] == pytest.approx(PC_D, abs=1e-4)
    assert base["Pgg_MPa"][-1] == pytest.approx(D["Pgg"] / 1e6, abs=1e-4)
    assert base["T_gg_K"][-1] == pytest.approx(T_GG_D, abs=0.01)


def test_base_start_speed_overshoot_is_within_3_percent(base):
    assert base["rpm"].max() / RPM_D - 1 < 0.03


def test_base_start_pump_discharge_overshoot_is_within_5_percent(base):
    assert base["Pd_o_MPa"].max() / PD_D - 1 < 0.05
    assert base["Pd_f_MPa"].max() / PD_D - 1 < 0.05


def test_base_start_chamber_pressure_and_thrust_overshoot_within_3_percent(base):
    assert base["Pc_MPa"].max() / PC_D - 1 < 0.03
    assert base["F_kN"].max() / F_D - 1 < 0.03


def test_base_start_time_to_90_percent_thrust_does_not_depend_on_the_recording_interval(base):
    """推力が定常値の 90 % に達する時刻は、5 ステップごとの記録からでも、毎ステップの記録からでも同じ値に読める"""
    fine = run(1.0, record_every=1)
    t_coarse = crossing_time(base["t"], base["F_kN"], 0.9 * F_D)
    t_fine = crossing_time(fine["t"], fine["F_kN"], 0.9 * F_D)
    assert t_coarse == pytest.approx(t_fine, abs=2e-5)
    assert 0.4537 < t_fine < 0.4547


def test_base_start_keeps_gg_temperature_near_design(base):
    assert base["T_gg_K"].max() < T_GG_D + 50.0


def test_base_start_keeps_main_chamber_mixture_ratio_near_design(base):
    burning = base["mdot_c"] > 30.0
    assert 2.30 < base["OF_mcc"][burning].min() and base["OF_mcc"][burning].max() < 2.45


def test_starter_torque_falls_linearly_with_speed_and_is_zero_at_design_speed():
    """スピンスタート（外部のガスでタービンを回す）のトルク: 止まっているとき最大で、回転数とともに下がり、設計回転数で 0。spin_time で止める"""
    eng = GGEngine()
    w_d = eng.omega_d
    assert eng.spin(0.0, 0.0) == eng.spin_torque > 0.0
    assert eng.spin(0.1, 0.5 * w_d) == pytest.approx(0.5 * eng.spin_torque, rel=1e-12)
    assert eng.spin(0.1, w_d) == 0.0 and eng.spin(0.1, 1.2 * w_d) == 0.0          # 設計回転数より上では、押しも引きもしない
    assert eng.spin(eng.spin_time, 0.5 * w_d) == 0.0 and eng.spin(eng.spin_time - 1e-9, 0.5 * w_d) > 0.0


def test_starter_alone_cannot_take_the_shaft_above_design_speed():
    """GG に点火しなければ（弁を開かなければ）、起動トルクだけでは設計回転数を超えない"""
    r = run(1.0, gg_valve_t=10.0, main_valve_t=(10.0, 10.3))
    assert r["W_t_MW"].max() == 0.0
    assert 5000.0 < r["rpm"].max() <= RPM_D


def test_base_start_does_not_overshoot_in_speed(base):
    assert base["rpm"].max() / RPM_D - 1 < 0.001


def test_rotor_energy_balance_during_start():
    """軸の回転エネルギーの増加 = ∫(タービン出力 − ポンプ動力 + 起動トルクの動力) dt"""
    eng = GGEngine()
    r = simulate(eng, 1.5, DT, record_every=1)
    t, w = r["t"], r["omega"]
    assert r["W_spin_MW"].max() > 0.2 and np.all(r["W_spin_MW"][t >= eng.spin_time] == 0.0)
    gained = 0.5 * eng.J * (w[-1] ** 2 - w[0] ** 2)
    supplied = trapezoid((r["W_t_MW"] - r["W_p_MW"] + r["W_spin_MW"]) * 1e6, t)
    assert gained == pytest.approx(supplied, rel=2e-3)


def test_propellant_mass_is_conserved_during_start():
    """タンクから減った質量 = ∫(主燃焼室への流量 + GG への流量) dt"""
    r = simulate(GGEngine(), 1.5, DT, record_every=1)
    drained = (r["m_f"][0] + r["m_o"][0]) - (r["m_f"][-1] + r["m_o"][-1])
    flowed = trapezoid(r["mdot_c"] + r["mdot_gg"], r["t"])
    assert drained == pytest.approx(flowed, rel=1e-3)


# ----------------------------------------------------------------------------- 停止
@pytest.fixture(scope="module")
def cut():
    """基準の停止手順（GG 弁 → 主弁、どちらも LOX 弁が先）"""
    return run(T_CUT + 1.5, cutoff_t=T_CUT)


def test_without_cutoff_nothing_changes_before_and_after(base):
    """cutoff_t を与えなければ、停止の機能は結果に影響しない"""
    r = run(3.0, cutoff_t=None)
    for k in ("rpm", "Pc_MPa", "F_kN", "T_gg_K"):
        assert np.array_equal(r[k], base[k])


def test_cutoff_does_not_change_anything_before_the_command(base, cut):
    before = cut["t"] < T_CUT - 1e-9
    n = int(before.sum())
    for k in ("rpm", "Pc_MPa", "F_kN", "T_gg_K"):
        assert np.array_equal(cut[k][:n], base[k][:n])


def test_cutoff_lags_never_move_a_valve_before_the_command(base):
    """LOX 弁を先に閉じる設定（負の lag）を大きくしても、弁が指令より前に動くことはない"""
    r = run(T_CUT + 0.1, cutoff_t=T_CUT, gg_lox_cut_lag=-0.04, main_lox_cut_lag=-0.05)
    n = int((r["t"] <= T_CUT + 1e-9).sum())
    for k in ("rpm", "Pc_MPa", "F_kN", "T_gg_K", "valve_main_lox", "valve_gg_lox"):
        assert np.array_equal(r[k][:n], base[k][:n]) if k in base else np.all(r[k][:n][r["t"][:n] > 0.6] == 1.0)


def test_cutoff_brings_thrust_and_chamber_pressure_down(cut):
    assert cut["F_kN"][-1] < 0.005 * F_D
    assert cut["Pc_MPa"][-1] < 0.12                     # ほぼ大気圧
    assert cut["mdot_c"][-1] == 0.0 and cut["mdot_gg"][-1] == 0.0


def test_cutoff_base_sequence_does_not_overspeed(cut):
    m = after(cut, T_CUT)
    assert cut["rpm"][m].max() <= cut["rpm"][m][0] * 1.002
    assert cut["rpm"][-1] < 0.5 * RPM_D                 # 1.5 s 後には半分以下まで減速している


def test_cutoff_base_sequence_stays_fuel_rich(cut):
    m = after(cut, T_CUT)
    burning = m & (cut["mdot_c"] > 30.0)
    assert cut["OF_mcc"][burning].max() < D["OF_c"] + 0.02      # 主燃焼室は酸素過剰側に振れない
    assert cut["T_gg_K"][m].max() < T_GG_D + 20.0               # GG の温度は上がらない
    assert cut["T_c_K"][m].max() < cut["T_c_K"][m][0] + 5.0     # 主燃焼室の温度も上がらない


def test_cutoff_with_gg_valves_closing_after_main_valves_overspeeds():
    """主弁を先に閉じ、GG 弁を 0.1 s 遅らせると、負荷を失ったポンプをタービンが回し続けて回転数が跳ね上がる"""
    g0, g1 = GGEngine().gg_cut_t
    m0, m1 = GGEngine().main_cut_t
    late = (m1, m1 + (g1 - g0))                                  # 主弁が閉じ切ってから GG 弁を閉じ始める
    r = run(T_CUT + 0.6, cutoff_t=T_CUT, gg_cut_t=late)
    assert r["rpm"][after(r, T_CUT)].max() / RPM_D - 1 > 0.10


def test_cutoff_with_gg_fuel_valve_closing_first_spikes_gg_temperature():
    """GG の燃料弁が先に閉じると酸素過剰を通過して温度が跳ね上がり、LOX 弁が先なら上がらない。
    （同時に閉じる場合は見ない: 主 LOX 弁が先に閉じて LOX ポンプの吐出圧が少し上がるぶん、GG の O/F がわずかに上がる）"""
    peak = {}
    for lag in (-0.02, 0.02):                                    # LOX 弁の遅れ [s]（− は LOX 弁が先、+ は燃料弁が先に閉じる）
        r = run(T_CUT + 0.5, record_every=1, cutoff_t=T_CUT, gg_lox_cut_lag=lag)
        peak[lag] = r["T_gg_K"][after(r, T_CUT)].max()
    assert peak[-0.02] < T_GG_D + 1.0
    assert peak[0.02] > peak[-0.02] + 200.0


def test_cutoff_impulse_grows_when_main_valves_close_slowly():
    """主弁をゆっくり閉じるほど、停止の指令のあとに出る力積（推力の時間積分）が増える"""
    def impulse(dur):
        m0, _ = GGEngine().main_cut_t
        r = run(T_CUT + 1.5, record_every=1, cutoff_t=T_CUT, main_cut_t=(m0, m0 + dur))
        m = after(r, T_CUT)
        return trapezoid(r["F_kN"][m], r["t"][m])             # [kN·s]
    fast, slow = impulse(0.05), impulse(0.30)
    assert slow > 1.2 * fast
    assert 0.0 < fast < F_D * 0.5                                # 全開の推力 0.5 秒分より小さい


def test_propellant_mass_is_conserved_through_cutoff():
    r = run(T_CUT + 1.0, record_every=1, cutoff_t=T_CUT)
    drained = (r["m_f"][0] + r["m_o"][0]) - (r["m_f"][-1] + r["m_o"][-1])
    flowed = trapezoid(r["mdot_c"] + r["mdot_gg"], r["t"])
    assert drained == pytest.approx(flowed, rel=1e-3)


# ----------------------------------------------------------------------------- GG の中のガスの温度（中身の O/F で決まる）
@pytest.fixture(scope="module")
def cut_fuel_first():
    """GG の燃料弁が LOX 弁より 10 ms 先に閉じる停止"""
    return simulate(GGEngine(cutoff_t=T_CUT, gg_lox_cut_lag=0.01), T_CUT + 0.2, 1e-4, record_every=1)


def test_gg_starts_filled_with_nitrogen_at_ambient_pressure_and_temperature():
    eng = GGEngine()
    r = simulate(eng, 0.01, 1e-4, record_every=1)
    assert r["Pgg_MPa"][0] == pytest.approx(P_ATM / 1e6, rel=1e-12) and r["T_gg_K"][0] == pytest.approx(eng.T_init, rel=1e-12)
    assert r["mi_gg"][0] == r["m_gg"][0] > 0.0 and r["mo_gg"][0] == 0.0


def test_gg_temperature_is_the_flame_temperature_at_the_mixture_ratio_of_its_own_gas(cut_fuel_first):
    """反応が速ければ、GG の中のガスは「中身の O/F」の燃焼温度になる。酸素と燃料過剰のガスが、反応しないまま同居することはない
    （停止のころには、起動前に入っていた窒素はすべて流れ出ているので、薄まる分は考えなくてよい）"""
    r, m = cut_fuel_first, after(cut_fuel_first, T_CUT)
    T_mix = np.array([float(T_AD(float(of))) for of in r["OF_gg_gas"][m]])
    assert np.max(np.abs(r["T_gg_K"][m] - T_mix)) < 1.0


def test_lox_alone_does_not_cool_fuel_rich_gas_in_the_gg(cut_fuel_first):
    """燃料弁が閉じて LOX だけが入ってくるあいだ、中身がまだ燃料過剰（O/F 2.9 未満。燃焼温度が最高になる手前）なら、
    入ってきた酸素は中のガスと燃えるので、温度は下がらない"""
    r = cut_fuel_first
    m = after(r, T_CUT) & (r["valve_gg"] == 0.0) & (r["valve_gg_lox"] > 0.0) & (r["OF_gg_gas"] < 2.9)
    assert m.sum() > 50
    assert np.all(np.diff(r["T_gg_K"][m]) >= -0.5)


def test_nothing_burns_in_the_gg_before_any_fuel_is_admitted():
    """LOX 弁だけを 40 ms 先に開く。燃料弁が動くまで GG の中にあるのは窒素と酸素だけなので、温度は上がらない"""
    r = simulate(GGEngine(gg_lox_lag=-0.04), 0.15, 1e-4, record_every=1)
    assert np.all(r["valve_gg"] == 0.0) and r["valve_gg_lox"].max() > 0.0
    assert r["T_gg_K"].max() < 400.0


def test_lox_lead_temperature_peak_depends_on_how_much_nitrogen_is_in_the_gg_before_start():
    """LOX 弁が先に開いたときの温度のピークは、起動前から入っている窒素が多いほど低い（薄まるため）。
    停止のときは窒素がもう残っていないので、ピークは窒素の量によらない"""
    start = [simulate(GGEngine(gg_lox_lag=-0.01, gg_n2_fill=x), 0.4, 1e-4, record_every=2)["T_gg_K"].max() for x in (0.5, 1.0, 2.0)]
    assert start[0] > start[1] + 200.0 and start[1] > start[2] + 200.0
    stop = [simulate(GGEngine(cutoff_t=T_CUT, gg_lox_cut_lag=0.01, gg_n2_fill=x), T_CUT + 0.2, 1e-4, record_every=2)["T_gg_K"].max() for x in (0.5, 2.0)]
    assert stop[0] == pytest.approx(stop[1], abs=1.0)


def test_nitrogen_in_the_gg_is_conserved_until_gas_starts_to_leave_through_the_turbine():
    """LOX が入り始めても、タービンへガスが流れ出すまでは、起動前から入っていた窒素の量は変わらない"""
    r = simulate(GGEngine(gg_lox_lag=-0.01), 0.16, 1e-4, record_every=1)
    m = r["mdot_t"] == 0.0
    assert m.sum() > 1000 and r["mdot_gg"][m].max() > 0.0
    assert np.ptp(r["mi_gg"][m]) < 1e-12


def test_gg_temperature_with_nitrogen_left_is_the_constant_pressure_mixing_temperature():
    """窒素が質量で 30 % 残っている GG（推進剤からできたガスは O/F 3）。混ぜる前後でエンタルピー（質量 × cp × T）の合計が同じになる温度で、
    質量の割合で平均した温度（2,720.6 K）ではない。O/F 3 のガスは、表の比熱比 1.132・分子量 24.75・燃焼温度 3,758 K"""
    eng = GGEngine()
    g = eng.gg_state(dict(m_gg=0.05, mi_gg=0.015, mo_gg=0.02625))
    cp_p = 1.132 * (8314.46 / 24.75) / 0.132
    cp, R = 0.7 * cp_p + 0.3 * 1150.0, 0.7 * 8314.46 / 24.75 + 0.3 * 296.8
    T = (0.7 * cp_p * 3758.0 + 0.3 * 1150.0 * 300.0) / cp
    assert T == pytest.approx(3252.8, abs=0.1) and T - (0.7 * 3758.0 + 0.3 * 300.0) > 500.0
    assert g["Y"] == pytest.approx(0.75, rel=1e-12) and g["cp"] == pytest.approx(cp, rel=1e-9) and g["T"] == pytest.approx(T, rel=1e-9)
    assert g["R"] == pytest.approx(R, rel=1e-9) and g["gam"] == pytest.approx(cp / (cp - R), rel=1e-9)
    assert g["P"] == pytest.approx(0.05 * R * T / eng.V_gg, rel=1e-9)


def test_gg_state_without_nitrogen_at_the_design_mixture_ratio_is_the_design_point():
    """窒素がなく、中身が設計の O/F なら、設計の温度と圧力になる"""
    eng = GGEngine()
    m = D["Pgg"] * eng.V_gg / (D["R_g"] * T_GG_D)
    g = eng.gg_state(dict(m_gg=m, mi_gg=0.0, mo_gg=m * D["OF_g"] / (1 + D["OF_g"])))
    assert g["T"] == pytest.approx(T_GG_D, rel=1e-9) and g["P"] == pytest.approx(D["Pgg"], rel=1e-9)


@pytest.fixture(scope="module")
def lox_lead_states():
    """GG の LOX 弁が 20 ms 先に開く起動。記録した状態量から、各時刻の流量を計算し直す"""
    eng = GGEngine(gg_lox_lag=-0.02)
    r = simulate(eng, 1.0, 1e-4, record_every=1)
    states = ("m_f", "m_o", "omega", "Pc", "Tc", "m_gg", "mo_gg", "mi_gg")
    flows = [eng.flows(float(t), {k: float(r[k][i]) for k in states}) for i, t in enumerate(r["t"])]
    return eng, r, {k: np.array([f[k] for f in flows]) for k in ("mf_g", "mo_g", "m_t")}


def test_each_kind_of_gas_in_the_gg_is_conserved_when_the_lox_valve_leads(lox_lead_states):
    """酸化剤からできた分・燃料からできた分・窒素のそれぞれで、∫(流入 − 流出) dt = 中の質量の変化。
    流出は「中身と同じ組成」。燃料の分は状態量ではない（全体 − 酸化剤 − 窒素）ので、3 本の式と clamp() がかみ合っていないと合わない"""
    _, r, f = lox_lead_states
    out = f["m_t"] / r["m_gg"]
    kinds = {"酸化剤": (f["mo_g"], r["mo_gg"]), "燃料": (f["mf_g"], r["m_gg"] - r["mo_gg"] - r["mi_gg"]), "窒素": (0.0 * out, r["mi_gg"])}
    for name, (inflow, m) in kinds.items():
        resid = trapezoid(inflow - out * m, r["t"]) - (m[-1] - m[0])
        assert abs(resid) < 1e-5, f"{name}: {resid:.3g} kg"
        assert m.min() > -1e-12, name


def test_gg_pressure_temperature_and_mass_satisfy_the_gas_law_at_every_recorded_step(lox_lead_states):
    """記録された GG の圧力・温度と、3 つの質量から決まるガス定数（質量の割合で平均）が、P V = m R T を満たす"""
    eng, r, _ = lox_lead_states
    mp = r["m_gg"] - r["mi_gg"]
    R_p = np.array([gg_gas(min(float(mo / m), 0.999) if m > 1e-12 else 0.0)[2] for mo, m in zip(r["mo_gg"], mp)])
    R = (mp * R_p + r["mi_gg"] * 296.8) / r["m_gg"]
    assert np.max(np.abs(r["Pgg_MPa"] * 1e6 * eng.V_gg / (r["m_gg"] * R * r["T_gg_K"]) - 1.0)) < 1e-9


@pytest.mark.parametrize("fill", [0.0, -1.0])
def test_gg_must_hold_some_gas_before_start(fill):
    """起動前の GG が空（または負の量）だと、ガスの組成が決まらない。黙って計算を進めずにエラーにする"""
    with pytest.raises(ValueError, match="gg_n2_fill"):
        GGEngine(gg_n2_fill=fill)


def test_nitrogen_above_the_turbine_exit_pressure_leaks_out_before_the_valves_open():
    """起動前の窒素を 5 気圧ぶん入れても、タービン出口の圧力（0.2 MPa）より高い分は、弁が開く前にタービンのノズルから抜ける。
    残るのは 0.2 MPa ぶんで、gg_n2_fill を約 1.97（= 0.2 MPa ÷ 1 気圧）より大きくしても結果は変わらない"""
    eng = GGEngine(gg_n2_fill=5.0)
    r = simulate(eng, 0.14, 1e-4, record_every=1)
    left = eng.P_exh * eng.V_gg / (296.8 * eng.T_init)
    assert r["mi_gg"][0] == pytest.approx(5.0 * P_ATM * eng.V_gg / (296.8 * eng.T_init), rel=1e-12)
    assert r["mi_gg"][-1] == pytest.approx(left, rel=2e-3) and r["Pgg_MPa"][-1] * 1e6 == pytest.approx(eng.P_exh, rel=2e-3)


# ----------------------------------------------------------------------------- 容器の中のガスの質量保存
def gas_balance(r, key_in, key_out, key_m):
    """∫(流入 − 流出) dt − 容器の中のガスの質量の変化 [kg]。0 ならガスが湧きも消えもしていない"""
    return trapezoid(r[key_in] - r[key_out], r["t"]) - (r[key_m][-1] - r[key_m][0])


@pytest.mark.parametrize("kw, t_end", [
    (dict(gg_lox_lag=-0.02), 1.0),                               # 起動: GG の LOX 弁が 20 ms 先に開く（GG の O/F が大きく振れる）
    (dict(cutoff_t=T_CUT, gg_lox_cut_lag=0.02), T_CUT + 0.4),    # 停止: GG の燃料弁が 20 ms 先に閉じる
])
def test_gas_mass_is_conserved_in_both_volumes_when_the_gg_mixture_ratio_swings(kw, t_end):
    r = run(t_end, record_every=1, **kw)
    assert abs(gas_balance(r, "mdot_c", "mdot_c_out", "m_gas_c")) < 0.01 * r["m_gas_c"].max()
    assert abs(gas_balance(r, "mdot_gg", "mdot_t", "m_gas_gg")) < 0.01 * r["m_gas_gg"].max()


# ----------------------------------------------------------------------------- 設計計算
def test_vacuum_thrust_adds_the_back_pressure_on_both_the_nozzle_and_the_turbine_exhaust():
    """真空では、主ノズルとタービン排気の出口の両方で、大気圧が押し返していた分だけ推力が増える"""
    assert D["A_exh"] > 0.0
    assert D["F_vac"] - D["F_sl"] == pytest.approx(P_ATM * (D["A_e"] + D["A_exh"]), rel=1e-12)


def test_design_calculation_refuses_a_chamber_pressure_at_which_the_nozzle_flow_would_separate():
    """設計計算の推力の式は、流れが出口まで壁に沿っている場合のもの。開口比 16 のまま燃焼圧を約 5.7 MPa より下げると海面ではがれるので、
    そのまま計算すると、時間積分のモデル（はく離を考える）と推力が食い違う（5 MPa で 845 kN のつもりが 856 kN）。エラーにして知らせる"""
    with pytest.raises(ValueError, match="はがれ"):
        design_point(Pc=5e6)
    assert design_point(Pc=6e6)["F_sl"] == D["F_sl"]


# ----------------------------------------------------------------------------- 主ノズル（海面での流れのはく離）
def full_flow_thrust(eng, Pc, of_c=None):
    """はく離を考えない式: F = η CF_vac Pc At − Pa Ae"""
    return eng.d["eta_cf"] * float(gg_rocket.CF_VAC(D["OF_c"] if of_c is None else of_c)) * Pc * eng.A_t - P_ATM * eng.A_e


def test_nozzle_flows_full_at_the_design_point_so_the_thrust_formula_is_unchanged_there():
    eng = GGEngine()
    assert eng.main_thrust(D["Pc"], D["OF_c"]) == pytest.approx(full_flow_thrust(eng, D["Pc"]), rel=1e-12)
    assert gg_rocket.nozzle(eng.pr_exit)[0] == pytest.approx(D["eps"], rel=1e-9)        # 出口の圧力比のところで、面積比がちょうど開口比になる


def test_nozzle_separation_starts_where_the_exit_pressure_falls_to_the_assumed_fraction_of_ambient():
    """出口の圧力が大気圧の K_SEP 倍になる燃焼圧を境に、それより上は従来の式、下ははく離を考えた式。境目で推力は連続"""
    eng = GGEngine()
    Pc_sep = gg_rocket.K_SEP * P_ATM / eng.pr_exit
    assert 4e6 < Pc_sep < 7e6
    assert eng.main_thrust(Pc_sep * 1.001, D["OF_c"]) == pytest.approx(full_flow_thrust(eng, Pc_sep * 1.001), rel=1e-12)
    assert eng.main_thrust(Pc_sep * 0.999, D["OF_c"]) == pytest.approx(full_flow_thrust(eng, Pc_sep * 0.999), rel=1e-3)


@pytest.mark.parametrize("Pc", [5e6, 3e6, 1e6, 0.5e6, 0.2e6])
def test_separated_nozzle_gives_more_thrust_than_the_full_flow_formula(Pc):
    """はがれた先の壁には大気圧がかかるので、満流のまま過膨張させた式より推力は大きい。燃焼圧が大気圧より高いあいだは、推力は正"""
    eng = GGEngine()
    assert eng.main_thrust(Pc, D["OF_c"]) > max(full_flow_thrust(eng, Pc), 0.0)


def test_main_thrust_is_zero_when_the_chamber_is_at_ambient_pressure():
    assert GGEngine().main_thrust(P_ATM, D["OF_c"]) == 0.0


def test_main_thrust_is_zero_instead_of_an_error_below_ambient_pressure():
    """燃焼圧が大気圧より低いと、はがれる断面の圧力比が 1 を超えて式が計算できない。流れないのだから 0 を返す"""
    eng = GGEngine()
    assert eng.main_thrust(0.3 * P_ATM, D["OF_c"]) == 0.0 and eng.main_thrust(0.0, D["OF_c"]) == 0.0


@pytest.mark.parametrize("M, pr, eps", [(2.0, 0.127805, 1.6875), (3.0, 0.027224, 4.2346)])
def test_nozzle_relations_reproduce_the_textbook_isentropic_table(M, pr, eps):
    """圧縮性流れの教科書の表（比熱比 1.4）: マッハ 2 と 3 の圧力比 P/P0 と面積比 A/A*。
    そこで切ったノズルの真空推力係数は（運動量 + 圧力）÷（Pc At）= (P/P0)(A/A*)(1 + γM²)。式の符号を取り違えると合わない"""
    e, cf = gg_rocket.nozzle(pr, 1.4)
    assert e == pytest.approx(eps, rel=2e-4)
    assert cf == pytest.approx(pr * eps * (1 + 1.4 * M * M), rel=2e-4)


SEPARATED = [(5e6, 1.48140, 0.9626), (3e6, 1.39633, 0.9563), (1e6, 1.16718, 0.9357), (0.5e6, 0.97495, 0.9137), (0.2e6, 0.61755, 0.8616)]


@pytest.mark.parametrize("Pc, cf_ref, ratio_ref", SEPARATED)
def test_separated_thrust_is_the_nozzle_cut_at_the_separation_section_with_ambient_pressure_behind_it(Pc, cf_ref, ratio_ref):
    """はがれたときの推力 =（はがれる断面で切ったノズルの真空推力）−（大気圧 × その断面の面積）。
    cf_ref は F ÷（Pc At）、ratio_ref は「大気圧までちょうど膨張させたノズル」の推力に対する比（どちらもレビューで別に計算した値）。
    はがれる目安が大気圧の 0.4 倍なので、ちょうど膨張させたノズルより推力は小さい"""
    eng = GGEngine()
    cf_design = eng.d["eta_cf"] * float(gg_rocket.CF_VAC(D["OF_c"])) / gg_rocket.nozzle(eng.pr_exit)[1]   # 表の推力係数 ÷ 理想気体の式の値
    cut = lambda pr: cf_design * gg_rocket.nozzle(pr)[1] - P_ATM / Pc * gg_rocket.nozzle(pr)[0]           # 圧力比 pr の断面で切ったノズルの F ÷（Pc At）
    cf = eng.main_thrust(Pc, D["OF_c"]) / (Pc * eng.A_t)
    assert cf == pytest.approx(cut(gg_rocket.K_SEP * P_ATM / Pc), rel=1e-12)
    assert cf == pytest.approx(cf_ref, rel=2e-5)
    assert cf < cut(P_ATM / Pc) and cf / cut(P_ATM / Pc) == pytest.approx(ratio_ref, abs=2e-4)


def test_main_thrust_rises_steadily_with_chamber_pressure_through_the_separation_onset():
    """燃焼圧を大気圧から 12 MPa まで上げていくと、推力は途中で下がることなく増え続ける（はく離の始まる 5.67 MPa にも段差がない）"""
    eng = GGEngine()
    assert 139.0 < 1.0 / eng.pr_exit < 141.0                            # 開口比 16 の圧力比 Pc/Pe（化学平衡計算では 139.8）
    assert gg_rocket.K_SEP * P_ATM / eng.pr_exit == pytest.approx(5.674e6, rel=1e-3)
    Pc = np.linspace(1.001 * P_ATM, 12e6, 6001)
    F = np.array([eng.main_thrust(float(p), D["OF_c"]) for p in Pc])
    assert np.all(np.diff(F) > 0.0)


def test_a_choked_nozzle_still_pushes_during_shutdown(cut):
    """停止の途中、燃焼圧が 0.5 MPa（約 5 気圧）まで下がった時点でも、スロートはチョークしていてガスが出ているので、推力は 0 ではない"""
    m = after(cut, T_CUT)
    i = int(np.argmax(cut["Pc_MPa"][m] <= 0.5))
    assert cut["mdot_c_out"][m][i] > 1.0
    assert cut["F_kN"][m][i] > 5.0


def test_thrust_is_gone_only_when_the_chamber_is_nearly_empty(cut):
    """推力が停止前の 1 % まで下がるのは、燃焼圧が 3 気圧より下がってから"""
    m = after(cut, T_CUT)
    i = int(np.argmax(cut["F_kN"][m] <= 0.01 * F_D))
    assert cut["Pc_MPa"][m][i] * 1e6 < 3.0 * P_ATM


# ----------------------------------------------------------------------------- 容器のモデル（部品の単体テスト）
def test_chamber_temperature_follows_with_the_gas_mass_divided_by_the_inflow():
    """τ = 容器の中のガスの質量 ÷ 流入量。定常では流入 = 流出なので、流出で割る誤りは、流入と流出が違う状態でないと見つからない"""
    P, T, V, R, m_in, m_out, T_target = 5e6, 2000.0, 0.4, 350.0, 80.0, 30.0, 3500.0
    dP, dT = GGEngine.chamber(P, T, V, R, m_in, m_out, T_target)
    tau = (P * V / (R * T)) / m_in
    assert dT == pytest.approx((T_target - T) / tau, rel=1e-12)
    assert dP == pytest.approx(R * T / V * (m_in - m_out) + P / T * dT, rel=1e-12)


# ----------------------------------------------------------------------------- 時間刻みと安定性（記事 4 ページ目）
STATES_6 = ["omega", "Pc", "Tc", "m_gg", "mo_gg", "mi_gg"]       # タンクの残量を除く状態量


@pytest.fixture(scope="module")
def eigenvalues():
    """定常点のまわりで線形化した状態方程式の固有値（中心差分。run_studies.py と同じ求め方）"""
    eng = GGEngine()
    r = simulate(eng, 2.0, 5e-4, record_every=4)
    s_ss = {k: float(r[k][-1]) for k in ("m_f", "m_o", *STATES_6)}

    def rhs(v):
        d = eng.derivatives(3.0, dict(s_ss, **dict(zip(STATES_6, v))))
        return np.array([d[k] for k in STATES_6])
    x0 = np.array([s_ss[k] for k in STATES_6])
    scale = np.array([s_ss["m_gg"] if k.endswith("_gg") else s_ss[k] for k in STATES_6])   # 窒素の量は定常では 0 なので、GG の質量は全体の量を基準に振る
    assert np.max(np.abs(rhs(x0)) / scale) < 1e-3, "定常に達していない"
    J = np.zeros((6, 6))
    for j in range(6):
        h = scale[j] * 1e-5
        xp, xm = x0.copy(), x0.copy()
        xp[j] += h
        xm[j] -= h
        J[:, j] = (rhs(xp) - rhs(xm)) / (2 * h)
    return np.linalg.eigvals(J)


def test_steady_state_eigenvalues_are_real_and_negative(eigenvalues):
    """「安定に解ける条件」の式は、固有値が負の実数であることを前提にしている"""
    assert np.max(np.abs(eigenvalues.imag)) < 1e-6 * np.max(np.abs(eigenvalues.real))
    assert np.all(eigenvalues.real < 0)


def test_fastest_eigenvalue_is_about_minus_1170_per_second(eigenvalues):
    assert eigenvalues.real.min() == pytest.approx(-1166.0, rel=0.02)


def pc_error(method, dt, t_end, ref):
    """燃焼圧の、基準の計算（4段・0.2 ms）からのずれの最大値 [MPa]"""
    q = simulate(GGEngine(), t_end, dt, method=method)
    return float(np.max(np.abs(q["Pc_MPa"] - np.interp(q["t"], ref["t"], ref["Pc_MPa"]))))


@pytest.fixture(scope="module")
def ref_start():
    return simulate(GGEngine(), 0.7, 2e-4, method="rk4")


def test_euler_is_stable_at_1_7_ms_and_diverges_at_2_0_ms(ref_start):
    """Δt·|λ| ≤ 2 → Δt ≤ 1.7 ms。燃焼圧が立ち上がる 0.2〜0.5 秒を含む短い計算で確かめる"""
    assert pc_error("euler", 1.7e-3, 0.68, ref_start) < 0.2          # 限界の手前: 誤差は大きめだが発散しない
    assert pc_error("euler", 2.0e-3, 0.60, ref_start) > 1.0          # 限界の外: 燃焼圧が 1 MPa 以上ずれる


def test_rk4_is_still_accurate_at_2_0_ms(ref_start):
    assert pc_error("rk4", 2.0e-3, 0.60, ref_start) < 0.02


def start_outputs(dt):
    r = simulate(GGEngine(), 1.0, dt, record_every=round(2e-4 / dt))    # 記録の間隔は 0.2 ms にそろえる
    return dict(rpm_peak=float(r["rpm"].max()), Pd_o_peak=float(r["Pd_o_MPa"].max()), Pd_f_peak=float(r["Pd_f_MPa"].max()),
                Pc_peak=float(r["Pc_MPa"].max()), Tgg_peak=float(r["T_gg_K"].max()), t90=crossing_time(r["t"], r["F_kN"], 0.9 * F_D))


@pytest.fixture(scope="module")
def halved():
    return start_outputs(2e-4), start_outputs(1e-4)


# 許容値は、実際の差（ピークの値で 2〜3×10⁻⁵、GG の温度で 3×10⁻⁸、90 % 到達時刻で 1×10⁻⁶）の 3 倍以上
@pytest.mark.parametrize("key, rel", [("rpm_peak", 1e-4), ("Pd_o_peak", 1e-4), ("Pd_f_peak", 1e-4), ("Pc_peak", 1e-4), ("Tgg_peak", 1e-6), ("t90", 1e-5)])
def test_start_transient_results_do_not_change_when_dt_is_halved(halved, key, rel):
    """記事の起動の計算（Δt = 0.2 ms）の結果は、Δt を半分にしても変わらない"""
    coarse, fine = halved
    assert coarse[key] == pytest.approx(fine[key], rel=rel)
