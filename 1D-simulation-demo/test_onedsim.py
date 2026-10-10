# -*- coding: utf-8 -*-
"""onedsim.py（ソルバー本体）の確認。実行: python -m pytest -q   （このフォルダで。pytest が必要）
確認しているのは、積分法が教科書どおりの精度で解けることと、おかしな入力や発散を黙って通さないこと。"""
import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from onedsim import Model, Table1D, bisect, simulate, sweep   # noqa: E402


class Growth(Model):
    """ds/dt = s（s(0) = 1）。厳密解は e^t。記事の「1ステップの比較」と同じ問題"""
    def initial_state(self):
        return {"s": 1.0}

    def derivatives(self, t, s):
        return {"s": s["s"]}


class Wavy(Model):
    """dy/dt = y cos t（y(0) = 1）。厳密解は exp(sin t)。滑らかなので、各積分法の理論どおりの次数が出るはず"""
    def initial_state(self):
        return {"y": 1.0}

    def derivatives(self, t, s):
        return {"y": s["y"] * math.cos(t)}


# ----------------------------------------------------------------------------- 積分法の精度
@pytest.mark.parametrize("method, expected", [("euler", 2.0), ("rk2", 2.5), ("rk4", 1 + 1 + 1 / 2 + 1 / 6 + 1 / 24)])
def test_one_step_of_ds_dt_equals_s_gives_the_values_in_the_article(method, expected):
    r = simulate(Growth(), t_end=1.0, dt=1.0, method=method)
    assert r["s"][-1] == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("method, order", [("euler", 1), ("rk2", 2), ("rk4", 4)])
def test_error_shrinks_with_the_theoretical_order_on_a_smooth_problem(method, order):
    exact = math.exp(math.sin(2.0))
    err = [abs(simulate(Wavy(), t_end=2.0, dt=dt, method=method)["y"][-1] - exact) for dt in (0.02, 0.01)]
    assert math.log2(err[0] / err[1]) == pytest.approx(order, abs=0.1)


# ----------------------------------------------------------------------------- 時刻と記録
def test_t_end_that_is_not_a_multiple_of_dt_is_rejected():
    with pytest.raises(ValueError):
        simulate(Growth(), t_end=1.0, dt=0.3)


def test_the_last_state_is_recorded_even_when_record_every_does_not_divide_the_steps():
    r = simulate(Growth(), t_end=1.0, dt=0.1, record_every=3)
    assert r["t"][-1] == pytest.approx(1.0, abs=1e-12)
    assert r["s"][-1] == pytest.approx(simulate(Growth(), t_end=1.0, dt=0.1)["s"][-1], rel=1e-12)


# ----------------------------------------------------------------------------- 発散とモデルの誤り
class SqrtOfNegative(Model):
    """式の書き間違い（負の数の平方根）を含むモデル"""
    def initial_state(self):
        return {"x": 1.0}

    def derivatives(self, t, s):
        return {"x": math.sqrt(-s["x"])}


def test_an_error_inside_the_model_is_reported_with_its_original_exception():
    with pytest.raises(FloatingPointError) as e:
        simulate(SqrtOfNegative(), t_end=1.0, dt=0.1)
    assert isinstance(e.value.__cause__, ValueError)          # 元の例外（math domain error）がたどれる
    assert "ValueError" in str(e.value)
    assert len(e.value.result["t"]) == 1                      # 止まる直前までの記録（初期値だけ）


class BlowsUp(Model):
    """2 ステップ目で状態量が無限大になる。clamp は上限で切るので、clamp の後で調べると見逃す"""
    def initial_state(self):
        return {"x": 1.0}

    def derivatives(self, t, s):
        return {"x": math.inf if t > 0.05 else 0.0}

    def clamp(self, s):
        s["x"] = min(s["x"], 10.0)
        return s


def test_an_infinite_state_is_caught_before_clamp_hides_it():
    with pytest.raises(FloatingPointError):
        simulate(BlowsUp(), t_end=1.0, dt=0.1, method="euler")


# ----------------------------------------------------------------------------- 部品
def test_bisect_rejects_an_interval_that_does_not_bracket_a_root():
    with pytest.raises(ValueError):
        bisect(lambda x: x - 10.0, 0.0, 1.0)


def test_bisect_accepts_a_root_at_the_end_of_the_interval():
    assert bisect(lambda x: x, 0.0, 1.0) == pytest.approx(0.0, abs=1e-9)


def test_bisect_finds_the_root_inside_the_interval():
    assert bisect(lambda x: x * x - 2.0, 0.0, 2.0) == pytest.approx(math.sqrt(2.0), abs=1e-9)


def test_table1d_interpolates_like_numpy_and_holds_the_end_values():
    t = Table1D([0.0, 1.0, 3.0], [0.0, 10.0, 0.0])
    xq = np.linspace(-1.0, 4.0, 11)
    assert np.allclose(t(xq), np.interp(xq, [0.0, 1.0, 3.0], [0.0, 10.0, 0.0]))
    assert [t(float(x)) for x in xq] == pytest.approx(list(t(xq)))
    assert t(-5.0) == 0.0 and t(9.0) == 0.0


def test_table1d_rejects_x_and_y_of_different_lengths():
    with pytest.raises(ValueError):
        Table1D([0.0, 1.0, 2.0], [0.0, 1.0])


def test_table1d_rejects_an_empty_table():
    with pytest.raises(ValueError):
        Table1D([], [])


def test_table1d_returns_nan_for_a_nan_input_like_numpy():
    t = Table1D([0.0, 1.0, 3.0], [0.0, 10.0, 0.0])
    assert math.isnan(t(float("nan")))
    assert np.isnan(t(np.array([float("nan")]))[0])


def test_bisect_compares_signs_so_that_tiny_function_values_do_not_underflow():
    """f(a)·f(m) の積で符号を調べると、値がごく小さいとき積が 0 に丸まって、間違った側を選んでしまう"""
    assert bisect(lambda x: 1e-170 * (x - 0.75), 0.0, 1.0) == pytest.approx(0.75, abs=1e-9)
    assert bisect(lambda x: -1e-170 * (x - 0.75), 0.0, 1.0) == pytest.approx(0.75, abs=1e-9)


def test_bisect_rejects_a_function_that_returns_nan():
    with pytest.raises(ValueError):
        bisect(lambda x: float("nan"), 0.0, 1.0)


def test_bisect_rejects_a_nan_that_appears_only_inside_the_interval():
    """両端では値が出るのに、途中で NaN になる関数。黙って上端の近くを返さず、エラーにする"""
    with pytest.raises(ValueError, match="NaN"):
        bisect(lambda x: float("nan") if 0.2 < x < 0.9 else x - 0.5, 0.0, 1.0)


def test_table1d_with_a_single_point_is_a_constant_like_numpy_even_for_nan():
    t = Table1D([2.0], [7.0])
    assert t(1.0) == t(2.0) == t(3.0) == 7.0
    assert t(float("nan")) == float(np.interp(float("nan"), [2.0], [7.0]))


def test_bisect_stays_at_the_lower_end_when_the_function_is_zero_there():
    """下端でちょうど 0（例: 弁が全部閉じているときのポンプの流量）なら、中の点の符号がどちらでも下端を返す"""
    assert bisect(lambda x: x, 0.0, 1.0) == pytest.approx(0.0, abs=1e-9)
    assert bisect(lambda x: -x, 0.0, 1.0) == pytest.approx(0.0, abs=1e-9)


# ----------------------------------------------------------------------------- simulate / sweep の引数
def test_record_every_below_one_is_rejected():
    with pytest.raises(ValueError):
        simulate(Growth(), t_end=1.0, dt=0.1, record_every=0)


class OutputNamedLikeAState(Growth):
    def outputs(self, t, s):
        return {"s": 2.0 * s["s"]}


def test_an_output_with_the_same_name_as_a_state_is_rejected():
    """同じ名前だと、1 つの記録に状態量と出力が交互に入ってしまう"""
    with pytest.raises(ValueError):
        simulate(OutputNamedLikeAState(), t_end=1.0, dt=0.1)


def test_stop_when_that_is_already_true_at_the_start_stops_before_the_first_step():
    r = simulate(Growth(), t_end=1.0, dt=0.1, stop_when=lambda t, s: True)
    assert r["stopped"] and list(r["t"]) == [0.0] and list(r["s"]) == [1.0]


def test_stop_when_records_the_state_at_which_it_stopped_even_between_recording_steps():
    r = simulate(Growth(), t_end=1.0, dt=0.1, record_every=4, stop_when=lambda t, s: t > 0.55)
    assert r["stopped"] and r["t"][-1] == pytest.approx(0.6, abs=1e-12)
    assert list(r["t"][:-1]) == pytest.approx([0.0, 0.4])


def test_sweep_accepts_a_generator_of_values():
    class Decay(Model):
        def __init__(self, k):
            self.k = k

        def initial_state(self):
            return {"y": 1.0}

        def derivatives(self, t, s):
            return {"y": -self.k * s["y"]}

    values, finals = sweep(Decay, "k", (k for k in [1.0, 2.0, 3.0]), 1.0, 0.01, lambda r: r["y"][-1])
    assert list(values) == [1.0, 2.0, 3.0]
    assert finals == pytest.approx([math.exp(-1.0), math.exp(-2.0), math.exp(-3.0)], rel=1e-6)


class NanOnce(Model):
    """nan_call 回目の傾きの計算だけ NaN を返す。同じステップの次の傾きは、NaN になった状態量で表を引くことになる"""
    def __init__(self, nan_call):
        self.table, self.nan_call, self.calls = Table1D([0.0, 1.0], [1.0, 2.0]), nan_call, 0

    def initial_state(self):
        return {"x": 0.5}

    def derivatives(self, t, s):
        self.calls += 1
        return {"x": self.table(s["x"]) * (float("nan") if self.calls == self.nan_call else 1.0)}


@pytest.mark.parametrize("method, nan_call", [("euler", 3), ("rk2", 5), ("rk4", 9)])     # どれも 3 ステップ目の最初の傾き
def test_a_nan_that_reaches_a_table_inside_a_step_is_reported_with_the_record_so_far(method, nan_call):
    with pytest.raises(FloatingPointError) as e:
        simulate(NanOnce(nan_call), t_end=1.0, dt=0.01, method=method)
    assert len(e.value.result["t"]) == 3                      # 初期値と、うまくいった 2 ステップ


# ----------------------------------------------------------------------------- 安定関数 R(z)（記事 4 ページ目の「安定に解ける条件」）
class Lin(Model):
    """dy/dt = z y を 1 ステップ（dt = 1）だけ解くと、y(1) は安定関数 R(z) になる"""
    def __init__(self, z):
        self.z = z

    def initial_state(self):
        return {"y": 1.0}

    def derivatives(self, t, s):
        return {"y": self.z * s["y"]}


def R(method, z):
    return simulate(Lin(z), t_end=1.0, dt=1.0, method=method)["y"][-1]


@pytest.mark.parametrize("z", [-0.5, -1.0, -2.0, -3.0])
def test_stability_functions_are_the_taylor_polynomials(z):
    assert R("euler", z) == pytest.approx(1 + z, rel=1e-12)
    assert R("rk2", z) == pytest.approx(1 + z + z ** 2 / 2, rel=1e-12)
    assert R("rk4", z) == pytest.approx(1 + z + z ** 2 / 2 + z ** 3 / 6 + z ** 4 / 24, rel=1e-12)


def test_euler_and_rk2_are_on_the_stability_boundary_at_z_minus_2():
    assert R("euler", -2.0) == pytest.approx(-1.0, abs=1e-12)     # 1 ステップごとに符号が反転し、振幅は変わらない
    assert R("rk2", -2.0) == pytest.approx(1.0, abs=1e-12)        # 振幅が変わらない（符号は反転しない）
    assert abs(R("euler", -1.99)) < 1 < abs(R("euler", -2.01))
    assert abs(R("rk2", -1.99)) < 1 < abs(R("rk2", -2.01))


def test_rk4_real_axis_stability_bound_is_2_785():
    z_star = bisect(lambda z: R("rk4", z) - 1.0, -3.0, -2.5, n=60)   # R(z) = 1 になる負の z
    assert -z_star == pytest.approx(2.785294, abs=1e-6)
    assert abs(R("rk4", -2.78)) < 1 < abs(R("rk4", -2.79))           # 2.79 は境界のわずかに外側


class Stiff(Model):
    """速い現象と遅い現象が同居する 2×2 の線形系。固有値は −1（[1, 1] の向き）と −1000（[1, −1] の向き）"""
    A = np.array([[-500.5, 499.5], [499.5, -500.5]])

    def initial_state(self):
        return {"y": np.array([2.0, 0.0])}

    def derivatives(self, t, s):
        return {"y": self.A @ s["y"]}


def stiff_final(method, z):
    """Δt·|λ| = z（λ = −1000）で 300 ステップ進めた最後の状態と、遅いモードの厳密解 e^(−t)"""
    dt = z / 1000.0
    r = simulate(Stiff(), t_end=300 * dt, dt=dt, method=method)
    return r["y"][-1], math.exp(-300 * dt)


@pytest.mark.parametrize("method, inside, outside", [("euler", 1.9, 2.1), ("rk2", 1.9, 2.1), ("rk4", 2.70, 2.90)])
def test_stiff_system_is_stable_just_inside_and_unstable_just_outside_the_bound(method, inside, outside):
    y, slow = stiff_final(method, inside)
    assert np.all(np.isfinite(y)) and np.max(np.abs(y)) < 2.0           # 速いモードは減衰し、遅いモードだけが残る
    assert y[0] == pytest.approx(slow, rel=0.3, abs=1e-3)
    try:
        y, _ = stiff_final(method, outside)
    except FloatingPointError:                                           # 発散して止まってもよい
        return
    assert np.max(np.abs(y)) > 1e5                                       # 速いモードが 10 万倍以上に膨らんでいる


# ----------------------------------------------------------------------------- 入力に段差があるとき（記事 4 ページ目の「1 次の精度に落ちる」）
class Forced(Model):
    """dy/dt = −y + u(t)。u は cut より前が 1、あとが 0。t0 と y0 は、計算を途中から始めるときの時刻と値"""
    def __init__(self, cut, t0=0.0, y0=0.0):
        self.cut, self.t0, self.y0 = cut, t0, y0

    def initial_state(self):
        return {"y": self.y0}

    def derivatives(self, t, s):
        return {"y": -s["y"] + (1.0 if self.t0 + t < self.cut else 0.0)}


FORCED_EXACT = (1.0 - math.exp(-0.5)) * math.exp(-0.5)                   # u を 0.5 で切ったときの y(1) の厳密解


def error_in_one_run(method, dt):
    return abs(simulate(Forced(cut=0.5), t_end=1.0, dt=dt, method=method)["y"][-1] - FORCED_EXACT)


def error_in_two_runs_split_at_the_step(method, dt):
    """段差の時刻で計算を 2 つに分ける。前半は u = 1 のまま、後半は u = 0 のままなので、どちらの中でも入力は滑らか"""
    a = simulate(Forced(cut=math.inf), t_end=0.5, dt=dt, method=method)
    b = simulate(Forced(cut=-math.inf, t0=0.5, y0=float(a["y"][-1])), t_end=0.5, dt=dt, method=method)
    return abs(b["y"][-1] - FORCED_EXACT)


def observed_order(error, method):
    return math.log2(error(method, 0.02) / error(method, 0.01))


def test_rk4_drops_to_first_order_when_the_input_steps_at_the_end_of_a_step():
    """段差の時刻ちょうどで終わるステップでは、最後の傾き（k4）が段差のあとの入力で計算されるので、4 段でも 1 次になる"""
    assert observed_order(error_in_one_run, "rk4") == pytest.approx(1.0, abs=0.15)


@pytest.mark.parametrize("method, expected", [("rk2", 2.0), ("rk4", 4.0)])
def test_splitting_the_run_at_the_step_in_the_input_restores_the_full_order(method, expected):
    assert observed_order(error_in_two_runs_split_at_the_step, method) == pytest.approx(expected, abs=0.25)
