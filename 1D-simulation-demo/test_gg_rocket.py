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
from gg_rocket import P_ATM, GGEngine, crossing_time, design_point   # noqa: E402

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


# ----------------------------------------------------------------------------- 起動
@pytest.fixture(scope="module")
def base():
    return run(3.0)


def test_old_start_sequence_regression():
    """主弁を 0.3〜0.9 s で開く以前の手順の結果が、意図せず変わっていないことの確認（回帰テスト）。
    値は、GG の中のガスの組成を状態量に加えた版（2026-10）で計算したもの。モデルの式を変えたら、確かめたうえで更新する"""
    r = run(3.0, main_valve_t=(0.30, 0.90))
    assert r["rpm"].max() == pytest.approx(44950.100823028275, rel=1e-9)
    assert r["Pd_o_MPa"].max() == pytest.approx(24.660025968206444, rel=1e-9)
    assert r["t"][np.argmax(r["F_kN"] > 0.9 * r["F_kN"][-1])] == pytest.approx(0.589, abs=1e-9)


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
    assert 0.4640 < t_fine < 0.4650


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
