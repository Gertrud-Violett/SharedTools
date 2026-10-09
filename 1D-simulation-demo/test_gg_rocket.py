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
from gg_rocket import P_ATM, GGEngine, crossing_time, design_point, turbine_eta_ratio   # noqa: E402

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
    値は、タービンの効率マップを式（放物線）で計算する版（2026-10）で計算したもの。モデルの式を変えたら、確かめたうえで更新する"""
    r = run(3.0, main_valve_t=(0.30, 0.90))
    assert r["rpm"].max() == pytest.approx(45369.85021466805, rel=1e-9)
    assert r["Pd_o_MPa"].max() == pytest.approx(25.14157759599409, rel=1e-9)
    assert r["t"][np.argmax(r["F_kN"] > 0.9 * r["F_kN"][-1])] == pytest.approx(0.583, abs=1e-9)


def test_base_start_settles_at_the_design_point(base):
    assert base["F_kN"][-1] == pytest.approx(F_D, abs=0.01)
    assert base["rpm"][-1] == pytest.approx(RPM_D, abs=1.0)
    assert base["Pc_MPa"][-1] == pytest.approx(PC_D, abs=1e-4)


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
    assert 0.4605 < t_fine < 0.4615


def test_base_start_keeps_gg_temperature_near_design(base):
    assert base["T_gg_K"].max() < T_GG_D + 50.0


def test_base_start_keeps_main_chamber_mixture_ratio_near_design(base):
    burning = base["mdot_c"] > 30.0
    assert 2.30 < base["OF_mcc"][burning].min() and base["OF_mcc"][burning].max() < 2.45


def test_rotor_energy_balance_during_start():
    """軸の回転エネルギーの増加 = ∫(タービン出力 − ポンプ動力) dt + ∫(起動トルク × 角速度) dt"""
    eng = GGEngine()
    r = simulate(eng, 1.5, DT, record_every=1)
    t, w = r["t"], r["omega"]
    spin = np.where(t < eng.spin_time, eng.spin_torque, 0.0) * w
    gained = 0.5 * eng.J * (w[-1] ** 2 - w[0] ** 2)
    supplied = trapezoid((r["W_t_MW"] - r["W_p_MW"]) * 1e6 + spin, t)
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


# ----------------------------------------------------------------------------- 容器のモデル（部品の単体テスト）
def test_chamber_temperature_follows_with_the_gas_mass_divided_by_the_inflow():
    """τ = 容器の中のガスの質量 ÷ 流入量。定常では流入 = 流出なので、流出で割る誤りは、流入と流出が違う状態でないと見つからない"""
    P, T, V, R, m_in, m_out, T_target = 5e6, 2000.0, 0.4, 350.0, 80.0, 30.0, 3500.0
    dP, dT = GGEngine.chamber(P, T, V, R, m_in, m_out, T_target)
    tau = (P * V / (R * T)) / m_in
    assert dT == pytest.approx((T_target - T) / tau, rel=1e-12)
    assert dP == pytest.approx(R * T / V * (m_in - m_out) + P / T * dT, rel=1e-12)


# ----------------------------------------------------------------------------- 時間刻みと安定性（記事 4 ページ目）
STATES_6 = ["omega", "Pc", "Pgg", "Tc", "Tgg", "Ygg"]            # タンクの残量を除く状態量


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
    assert np.max(np.abs(rhs(x0)) / np.abs(x0)) < 1e-3, "定常に達していない"
    J = np.zeros((6, 6))
    for j in range(6):
        h = x0[j] * 1e-5
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
