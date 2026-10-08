"""
onedsim — まきブログ版・最小1Dシミュレーションエンジン
=====================================================
1Dシミュレーターの中身は、突き詰めると次の3つだけです。

  1. Table1D   : 試験で得た特性曲線（ポンプのQ-H曲線など）をテーブルとして引く
  2. Model     : 「状態量」と「その時間微分」を定義する（＝部品をつないだ系全体の方程式）
  3. simulate  : 微分方程式を時間積分する（オイラー法 / 2段・4段ルンゲ＝クッタ法）
おまけとして、パラメータを振る sweep と、部品どうしの「つり合い点」を探す二分法 bisect が付いています。

商用ソフト（Amesim, GT-SUITE, EcosimPro...）はこれに GUI・部品ライブラリ・陰解法ソルバーを
足したものだと思ってください。依存は numpy のみです。

使い方:
    class MyModel(Model):
        def initial_state(self): return {"x": 0.0}
        def derivatives(self, t, s): return {"x": -s["x"]}
        def outputs(self, t, s):     return {"x2": s["x"] ** 2}     # 記録したい派生量（任意）
    res = simulate(MyModel(), t_end=10, dt=0.01)
    res["t"], res["x"], res["x2"]  -> numpy 配列
"""
from __future__ import annotations
from bisect import bisect_right
import numpy as np


# ----------------------------------------------------------------------------- 1. 特性テーブル
class Table1D:
    """試験データ（x→y）を線形補間で引く。範囲外は端の値で頭打ち（外挿しない）。"""

    def __init__(self, x, y, name=""):
        self.x = np.asarray(x, float)
        self.y = np.asarray(y, float)
        self.name = name
        if self.x.ndim != 1 or self.x.shape != self.y.shape or not np.all(np.isfinite(self.x) & np.isfinite(self.y)) or np.any(np.diff(self.x) <= 0):
            raise ValueError(f"Table1D {name}: x と y は同じ長さの有限の値で、x は単調増加で与えてください")
        self._x, self._y = self.x.tolist(), self.y.tolist()

    def __call__(self, xq):
        if isinstance(xq, float):                    # 数値1個のときは自前で補間（numpy を通すより数倍速い）
            x, y = self._x, self._y
            if xq <= x[0]:
                return y[0]
            if xq >= x[-1]:
                return y[-1]
            i = bisect_right(x, xq)
            return y[i - 1] + (y[i] - y[i - 1]) * (xq - x[i - 1]) / (x[i] - x[i - 1])
        return np.interp(xq, self.x, self.y)         # 配列はまとめて補間


def bisect(f, a, b, n=40):
    """二分法: f(a) と f(b) の符号が違うとき、f(x)=0 になる x を返す。
    ポンプ曲線と配管抵抗の交点（動作点）のような「部品どうしのつり合い」を解くのに使う。"""
    fa, fb = f(a), f(b)
    if not fa * fb <= 0:                         # 解をはさんでいない（端でちょうど 0 になるのはよい）
        raise ValueError(f"bisect: f({a:g}) と f({b:g}) が同じ符号です。解をはさむ区間を与えてください")
    for _ in range(n):
        m = 0.5 * (a + b)
        fm = f(m)
        if fa * fm <= 0:
            b = m
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


# ----------------------------------------------------------------------------- 2. モデル基底
class Model:
    """状態量 s (dict) と、その微分 ds/dt を返す derivatives() を持つ。"""

    def initial_state(self) -> dict:
        raise NotImplementedError

    def derivatives(self, t: float, s: dict) -> dict:
        raise NotImplementedError

    def outputs(self, t: float, s: dict) -> dict:
        """記録したい派生量（流量・出力など）。省略可。"""
        return {}

    def clamp(self, s: dict) -> dict:
        """各ステップ後に呼ばれる。負のタンク質量などを物理的に修正したいときに上書き。"""
        return s


# ----------------------------------------------------------------------------- 3. 時間積分
def _add(s, ds, h):
    return {k: s[k] + h * ds[k] for k in s}


def _step_euler(model, t, s, dt):
    return _add(s, model.derivatives(t, s), dt)


def _step_rk2(model, t, s, dt):
    """2段ルンゲ＝クッタ法（ホイン法、2次精度）"""
    k1 = model.derivatives(t, s)
    k2 = model.derivatives(t + dt, _add(s, k1, dt))
    return {k: s[k] + dt / 2 * (k1[k] + k2[k]) for k in s}


def _step_rk4(model, t, s, dt):
    k1 = model.derivatives(t, s)
    k2 = model.derivatives(t + dt / 2, _add(s, k1, dt / 2))
    k3 = model.derivatives(t + dt / 2, _add(s, k2, dt / 2))
    k4 = model.derivatives(t + dt, _add(s, k3, dt))
    return {k: s[k] + dt / 6 * (k1[k] + 2 * k2[k] + 2 * k3[k] + k4[k]) for k in s}


def simulate(model: Model, t_end: float, dt: float, method: str = "rk4",
             record_every: int = 1, stop_when=None) -> dict:
    """
    model        : Model のインスタンス
    t_end, dt    : 終了時刻と時間刻み [s]
    method       : "rk4"（4段ルンゲ＝クッタ法、推奨）/ "rk2"（2段ルンゲ＝クッタ法）/ "euler"（オイラー法、教材用）
    record_every : N ステップごとに記録（長時間計算でメモリ節約）
    stop_when    : f(t, s) -> True で打ち切り（例：タンク枯渇）
    戻り値       : {"t": array, <状態量>: array, <outputs>: array, "stopped": bool}
    """
    step = {"rk4": _step_rk4, "rk2": _step_rk2, "euler": _step_euler}[method]
    s = dict(model.initial_state())
    t = 0.0
    n = int(round(t_end / dt))
    if n < 1 or abs(n * dt - t_end) > 1e-9 * max(1.0, abs(t_end)):
        raise ValueError(f"t_end（{t_end}）は時間刻み dt（{dt}）の整数倍にしてください")
    rec = {"t": []}
    for k in list(s) + list(model.outputs(0.0, s)):
        rec[k] = []
    stopped = False

    def record(t, s):
        o = model.outputs(t, s)                      # 先に計算する（ここで失敗しても、記録の長さがそろったままになる）
        rec["t"].append(t)
        for k, v in list(s.items()) + list(o.items()):
            rec[k].append(v)

    record(t, s)
    for i in range(1, n + 1):
        try:
            s_new = step(model, t, s, dt)
            if not all(np.isrealobj(v) and np.all(np.isfinite(v)) for v in s_new.values()):   # clamp() で隠れる前に調べる
                raise FloatingPointError("状態量が無限大・NaN・複素数になりました")
            s, t = model.clamp(s_new), i * dt
            if i % record_every == 0:
                record(t, s)
        except (ValueError, OverflowError, ZeroDivisionError, FloatingPointError) as exc:   # 負の数の平方根なども含む
            err = FloatingPointError(f"t={t:.4g}: 計算を続けられません（{type(exc).__name__}: {exc}）。状態量の発散なら、時間刻み dt を"
                                     "小さくするか method を 'rk4' に。モデルの式の誤りでも起きるので、元の例外も確かめてください")
            err.result = {k: np.asarray(v, float) for k, v in rec.items()}     # 止まる直前までの記録
            raise err from exc
        if stop_when is not None and stop_when(t, s):
            stopped = True
            if i % record_every:
                record(t, s)
            break
    if not stopped and n % record_every:             # 最後の状態は必ず記録する
        record(t, s)
    out = {k: np.asarray(v, float) for k, v in rec.items()}
    out["stopped"] = stopped
    return out


# ----------------------------------------------------------------------------- 4. パラメトリックスタディ
def sweep(make_model, param_name: str, values, t_end, dt, metric, **sim_kw):
    """
    make_model(**{param_name: v}) でモデルを作り、metric(res) -> float を values ごとに評価。
    戻り値: (values, metrics) の numpy 配列。
    """
    ms = []
    for v in values:
        res = simulate(make_model(**{param_name: v}), t_end, dt, **sim_kw)
        ms.append(metric(res))
    return np.asarray(values, float), np.asarray(ms, float)
