# -*- coding: utf-8 -*-
"""onedsim.py（ソルバー本体）の確認。実行: python -m pytest -q   （このフォルダで。pytest が必要）
確認しているのは、積分法が教科書どおりの精度で解けることと、おかしな入力や発散を黙って通さないこと。"""
import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from onedsim import Model, Table1D, bisect, simulate   # noqa: E402


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
