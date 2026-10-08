"""
揚水発電所の1Dモデル（onedsim デモ①）
=====================================
神流川発電所（有効落差 653 m、水車出力 482 MW/台・発電機出力 470 MW/台、上池・下池 有効貯水量 1,267万 m3）を
参考にした数値で、上池・下池・ポンプ水車・電力需要・降雨をつないだ系を解きます。

部品:
  上池 / 下池      : 0D タンク（体積が状態量）。水位 = 体積 / 面積
  ポンプ水車       : 特性曲線（Q-H, 効率）を Table1D で持つ。動作点は系と連立して求める
  水路             : 損失 = k * Q^2
  運用ロジック     : 需要スケジュールに従って「発電 / 揚水 / 停止」を切替。発電は要求をまかなえる最小の台数、
                     揚水は余った電力で動かせる台数（1台あたり約 450 MW として判定）だけ動かす
  外乱             : 降雨（集水域→流入）、蒸発、需要スパイク

状態量: V_u (上池体積 m3), V_l (下池体積 m3)
        ほかに積算値として E_gen, E_pump, E_spill, E_short [MWh] と、水車・ポンプを通った水量 W_gen, W_pump、
        上池・下池からあふれた水量 W_spill_u, W_spill_l、雨と蒸発の量 W_rain, W_evap [m3]（水の収支の確認用）
"""
from __future__ import annotations
import math
import numpy as np
from onedsim import Model, Table1D, bisect

RHO, G = 1000.0, 9.81
DAY = 86400.0


class PumpedStorage(Model):
    def __init__(self, n_units=2, V_max=12.67e6, A_res=4.0e5, H_base=680.0,
                 k_loss=0.004, catch_up=6.2e6, runoff=0.8, rain=None, demand=None,
                 V_u0_frac=0.5, evap_mm_day=4.0, V_min_frac=0.05, pump_in_rain=True):
        self.n = n_units
        self.pump_in_rain = pump_in_rain             # False にすると、雨が降っているあいだは揚水しない
        self.V_max, self.A, self.H_base, self.k_loss = V_max, A_res, H_base, k_loss
        self.V_min = V_min_frac * V_max
        self.catch_up, self.runoff, self.evap = catch_up, runoff, evap_mm_day
        self.V_u0 = V_u0_frac * V_max
        self.rain = rain or (lambda t: 0.0)          # mm/h
        self.demand = demand or default_demand       # MW（正=発電要求, 負=揚水余剰）
        # ---- ポンプ水車の特性曲線（1台あたり、定格 500 min-1）----
        #   実機の試験曲線は公表されていないので、公表仕様（東芝レビュー 2006）に合うように曲線の形を仮定した。
        #     水車 : 基準有効落差 653 m・最大流量 85 m3/s・最大出力 482 MW  → このときの水車効率は 0.885
        #     ポンプ: 全揚程 677〜728 m・揚水量 60 m3/s・最大軸入力 464 MW
        #     発電所の出力は 2 台で 94万 kW（470 MW/台）= 水車 482 MW × 発電機効率 0.975
        #   有効落差 653 m は水路損失を引いた後の値 → 静落差 680 m、k_loss=0.004 とした（85 m3/s で損失 約 29 m、有効落差 約 651 m）
        #   この表で、基準の落差 680 m で揚水量 60 m3/s、全揚程 677 m 付近で軸入力 約 464 MW になる
        self.pump_H = Table1D([0, 20, 40, 50, 60, 70, 80], [774, 764, 739, 719, 694, 646, 586], "pump Q-H")
        self.pump_eta = Table1D([0, 20, 40, 50, 60, 70, 80], [0.0, 0.55, 0.80, 0.88, 0.91, 0.90, 0.85], "pump eta")
        self.turb_eta = Table1D([0, 20, 40, 60, 75, 85], [0.0, 0.70, 0.85, 0.90, 0.90, 0.885], "turbine eta")
        self.Q_turb_max = 85.0     # m3/s /台
        self.eta_gen, self.eta_motor = 0.975, 0.975   # 発電機 / 電動機効率
        self.P_unit = 470.0        # MW /台（発電機出力の上限 = 水車の最大出力 482 MW × 発電機効率）
        self.P_pump_nominal = 450.0   # MW /台（揚水 1 台のおおよその入力。余った電力で何台回せるかの判定に使う）
        self.V_spill = 0.999 * V_max  # 池がここまで来たら満水（上池はこれ以上の流入が越流、下池は下流の川へ放流）

    # ---------------- 系の幾何 ----------------
    def head(self, V_u, V_l):
        """静落差 [m]：両池が半分のとき H_base。上池が増え下池が減ると落差が増える。"""
        Vm = 0.5 * self.V_max
        return self.H_base + (V_u - Vm) / self.A - (V_l - Vm) / self.A

    # ---------------- 部品：ポンプ動作点 ----------------
    def pump_operating_point(self, H_static):
        """ポンプ曲線 H_pump(Q) と、システムの要求 H_static + k Q^2 の交点（1台あたり）。"""
        f = lambda Q: self.pump_H(Q) - (H_static + self.k_loss * Q ** 2)
        if f(0.0) <= 0:
            return 0.0
        return bisect(f, 0.0, 80.0)

    # ---------------- 部品：水車（ガイドベーン制御） ----------------
    def turbine_flow(self, P_req_unit, H_static):
        """要求出力 [MW/台] を出す流量を探す。上限 Q_turb_max。"""
        P_req_unit = min(P_req_unit, self.P_unit)
        f = lambda Q: RHO * G * Q * (H_static - self.k_loss * Q ** 2) * self.turb_eta(Q) * self.eta_gen / 1e6 - P_req_unit
        if f(self.Q_turb_max) < 0:
            return self.Q_turb_max
        return bisect(f, 0.0, self.Q_turb_max)

    # ---------------- 状態方程式 ----------------
    def initial_state(self):
        return {"V_u": self.V_u0, "V_l": self.V_max - self.V_u0, "E_gen": 0.0, "E_pump": 0.0, "E_spill": 0.0, "E_short": 0.0,
                "W_gen": 0.0, "W_pump": 0.0, "W_spill_u": 0.0, "W_spill_l": 0.0, "W_rain": 0.0, "W_evap": 0.0}

    def flows(self, t, s):
        V_u, V_l = s["V_u"], s["V_l"]
        H = self.head(V_u, V_l)
        P_req = self.demand(t)
        Q_t = Q_p = 0.0
        P_gen = P_pump = 0.0
        if P_req > 0 and V_u > self.V_min:                       # 発電: 要求をまかなえる最小の台数を動かし、その台数で等分する
            Qm = self.Q_turb_max
            P_max = min(RHO * G * Qm * (H - self.k_loss * Qm ** 2) * self.turb_eta(Qm) * self.eta_gen / 1e6, self.P_unit)   # この落差での1台の最大出力
            n_on = min(self.n, math.ceil(P_req / P_max - 1e-9))
            Q_t = n_on * self.turbine_flow(P_req / n_on, H)
            H_eff = H - self.k_loss * (Q_t / n_on) ** 2
            P_gen = RHO * G * Q_t * H_eff * self.turb_eta(Q_t / n_on) * self.eta_gen / 1e6
        elif P_req < 0 and V_u < self.V_spill and V_l > self.V_min and (self.pump_in_rain or self.rain(t) <= 0):  # 揚水
            n_pump = min(self.n, int(-P_req / self.P_pump_nominal + 1e-9))   # 余った電力でまかなえる台数（固定速なので入力は調整できない）
            Q1 = self.pump_operating_point(H)
            Q_p = n_pump * Q1
            H_p = self.pump_H(Q1)
            eta = max(self.pump_eta(Q1), 1e-3)
            P_pump = RHO * G * Q_p * H_p / eta / self.eta_motor / 1e6
        # 外乱
        Q_rain = self.rain(t) / 1000 / 3600 * self.catch_up * self.runoff
        Q_evap = self.evap / 1000 / DAY * self.A
        return dict(H=H, P_req=P_req, Q_t=Q_t, Q_p=Q_p, P_gen=P_gen, P_pump=P_pump, Q_rain=Q_rain, Q_evap=Q_evap)

    def derivatives(self, t, s):
        f = self.flows(t, s)
        dVu = f["Q_p"] - f["Q_t"] + f["Q_rain"] - f["Q_evap"]
        dVl = f["Q_t"] - f["Q_p"] - f["Q_evap"]
        spill = spill_l = 0.0
        if s["V_u"] >= self.V_spill and dVu > 0:                   # 上池満水 → 越流（満水では揚水しないので、あふれるのは雨の分だけ）
            spill, dVu = dVu, 0.0
        if s["V_l"] >= self.V_spill and dVl > 0:                   # 下池満水 → 発電した水を下流の川へ放流
            spill_l, dVl = dVl, 0.0
        return {"V_u": dVu, "V_l": dVl,
                "E_gen": f["P_gen"] / 3600, "E_pump": f["P_pump"] / 3600,
                "E_spill": RHO * G * spill * f["H"] * 0.9 / 3.6e9,        # 越流で失った発電機会 [MWh/s]
                "E_short": max(f["P_req"] - f["P_gen"], 0.0) / 3600 if f["P_req"] > 0 else 0.0,   # 供給不足 [MWh/s]
                "W_gen": f["Q_t"], "W_pump": f["Q_p"],                                           # 水車・ポンプを通った水量 [m3/s]
                "W_spill_u": spill, "W_spill_l": spill_l, "W_rain": f["Q_rain"], "W_evap": 2 * f["Q_evap"]}   # 水の収支の確認用 [m3/s]

    def clamp(self, s):
        s["V_u"] = min(max(s["V_u"], 0.0), self.V_max)
        s["V_l"] = min(max(s["V_l"], 0.0), self.V_max)
        return s

    def outputs(self, t, s):
        f = self.flows(t, s)
        E_stored = RHO * G * max(s["V_u"] - self.V_min, 0) * f["H"] * 0.9 / 3.6e9   # 発電可能エネルギー [MWh]（効率0.9）
        return dict(H=f["H"], P_req=f["P_req"], P_gen=f["P_gen"], P_pump=f["P_pump"], Q_t=f["Q_t"], Q_p=f["Q_p"],
                    Q_rain=f["Q_rain"], E_stored=E_stored, level_u=s["V_u"] / self.A, rain=self.rain(t))


# ----------------------------------------------------------------------------- 外乱プロファイル
def default_demand(t, peak_MW=900.0, pump_MW=-900.0):
    """1日の運用パターン [MW]。昼・夕方ピークに発電、深夜に揚水。"""
    h = (t % DAY) / 3600
    day = int(t // DAY)
    weekend = day % 7 in (5, 6)
    if h >= 22 or h < 6:                       # 深夜 8 時間：余剰電力で揚水
        return pump_MW * (0.7 if weekend else 1.0)
    if 13 <= h < 16:                           # 昼ピーク 3 時間
        return peak_MW * (0.5 if weekend else 1.0)
    if 18 <= h < 21:                           # 夕ピーク 3 時間
        return peak_MW * (0.6 if weekend else 1.0)
    return 0.0


def make_demand(peak_MW=900.0, pump_MW=-900.0, heatwave_day=None, heatwave_MW=None):
    def d(t):
        base = default_demand(t, peak_MW, pump_MW)
        if heatwave_day is not None and int(t // DAY) == heatwave_day and base > 0:
            return heatwave_MW if heatwave_MW is not None else base
        if heatwave_day is not None and int(t // DAY) == heatwave_day:
            h = (t % DAY) / 3600
            if 10 <= h < 13 or 16 <= h < 18:      # 猛暑日は昼間ずっと発電要求
                return peak_MW
        return base
    return d


def make_rain(storm_day=None, mm_per_h=30.0, hours=12.0):
    """通常は無降雨、storm_day の 6:00 から hours 時間 mm_per_h の豪雨。"""
    def r(t):
        if storm_day is None:
            return 0.0
        t0 = storm_day * DAY + 6 * 3600
        return mm_per_h if t0 <= t < t0 + hours * 3600 else 0.0
    return r


def roundtrip(res):
    """往復効率: 水 1 m3 あたりの発電量 ÷ 水 1 m3 あたりの揚水電力量。雨で増えた水や、週の初めと終わりの水位の差に左右されない。"""
    if res["W_gen"][-1] <= 0 or res["W_pump"][-1] <= 0:
        raise ValueError("往復効率は、発電と揚水の両方をした期間で計算してください")
    return (res["E_gen"][-1] / res["W_gen"][-1]) / (res["E_pump"][-1] / res["W_pump"][-1])


if __name__ == "__main__":
    from onedsim import simulate
    m = PumpedStorage(rain=make_rain(storm_day=5), demand=make_demand(heatwave_day=2, heatwave_MW=940.0))
    res = simulate(m, t_end=7 * DAY, dt=60.0, method="rk4", record_every=5)
    print(f"7日間: 発電 {res['E_gen'][-1]:,.0f} MWh / 揚水 {res['E_pump'][-1]:,.0f} MWh "
          f"→ 往復効率 {roundtrip(res):.3f}（同じ量の水を上げ下げしたときの 発電量 ÷ 揚水の電力量）/ 供給不足 {res['E_short'][-1]:,.0f} MWh / 越流損失 {res['E_spill'][-1]:,.0f} MWh")
    print(f"上池水位 min {res['level_u'].min():.1f} m  max {res['level_u'].max():.1f} m, 落差 {res['H'].min():.0f}–{res['H'].max():.0f} m")
