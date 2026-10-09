"""
ガスジェネレータ（GG）サイクル液体ロケットエンジンの1Dモデル（onedsim デモ②）
==========================================================================
Merlin 1D 級（LOX/RP-1、燃焼圧 9.7 MPa、海面推力 845 kN）を題材に、
タンク → ターボポンプ → 主燃焼室 / ガスジェネレータ → タービン を1本の軸でつないだ系を、起動から停止まで解きます。

前提（まきブログ「SpaceX特集 第3回」の検証と同じ値。DESIGN にまとめてあります）:
  燃焼圧 9.7 MPa・主燃焼室の混合比 2.36・ノズル開口比 16、ポンプ効率 75 %、GG タービン効率 45 %・入口温度 900 K、
  ポンプ吐出圧 = 燃焼圧 × 1.4、GG 圧 = 燃焼圧 × 0.8、タービン出口圧 0.2 MPa、タンク圧 0.3 MPa

部品（すべて「特性 + 保存則」だけ）:
  ポンプ         : 相似則で無次元化した揚程マップ・軸動力マップ（本来は単体試験で取るもの）
  弁・噴射器     : ṁ = CdA √(2ρΔP)
  ポンプの動作点 : ポンプ曲線と、下流（弁・噴射器）の抵抗曲線の交点を二分法で探す
  燃焼室 / GG    : 0D の容器。ガスの質量保存（圧力）と、燃焼温度への追従（温度）。GG はガスの組成（O/F）も混ざりながら追従する
  燃焼ガスの特性 : 酸燃比 O/F の表。化学平衡計算（RocketCEA）＋ 燃料過剰側は教科書（Huzel & Huang）の代表値
  タービン       : 速度比（周速 ÷ ガスの噴出速度）で決まる効率マップ
  ノズル         : F = η CF_vac Pc At − Pa Ae

状態量: m_f, m_o [kg], omega [rad/s], Pc, Pgg [Pa], Tc, Tgg [K], Ygg（GG の中のガスの酸化剤の質量分率。O/F = Ygg/(1 − Ygg)）
起動シーケンス: t = 0 スピンスタート（0.5 s の起動トルク）→ 0.15 s GG 弁 → 0.2–0.5 s 主弁ランプ開
               （燃料弁と LOX 弁は別々。gg_lox_lag / main_lox_lag で LOX 弁を遅らせる。負なら先行）
停止シーケンス: cutoff_t に停止の指令 → GG 弁（0.05 s）と主弁（0.15 s）を閉じる。gg_cut_t / main_cut_t は、指令から閉じ始め・閉じ切りまでの時間
               （LOX 弁を 20 ms 先に閉じて燃料過剰のまま止める。gg_lox_cut_lag / main_lox_cut_lag が負なら燃料弁が、正なら LOX 弁がその分遅れて閉じる）
"""
from __future__ import annotations
import math
from onedsim import Model, Table1D, bisect

G0, P_ATM = 9.80665, 101325.0
GAMMA_C = 1.15                          # 主燃焼室ガスの比熱比（CEA、チョーク判定にだけ使う）
RHO_F, RHO_O = 810.0, 1141.0            # RP-1 / LOX の密度 [kg/m3]


def crit_flow(gamma):
    """チョークしたノズルの流量係数 Γ: ṁ = Γ P A / √(RT)（c* = √(RT)/Γ）"""
    return math.sqrt(gamma) * (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))


def unchoke(P_back, P, gamma):
    """背圧が高くてチョークしないときの流量の落ち方（チョーク時を 1 とした比）"""
    r = P_back / P
    if r >= 1.0:
        return 0.0
    if r <= (2 / (gamma + 1)) ** (gamma / (gamma - 1)):
        return 1.0
    return math.sqrt(2 / (gamma - 1) * (r ** (2 / gamma) - r ** ((gamma + 1) / gamma))) / (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))


def crossing_time(t, y, level, rising=True):
    """y が初めて level 以上（rising=False なら level 以下）になる時刻。記録点のあいだは直線で補間する。達しなければ None"""
    past = (lambda v: v >= level) if rising else (lambda v: v <= level)
    for i in range(len(y)):
        if past(y[i]):
            if i == 0:                               # 最初の記録点ですでに条件を満たしている
                return float(t[0])
            return float(t[i - 1] + (level - y[i - 1]) / (y[i] - y[i - 1]) * (t[i] - t[i - 1]))
    return None


# ============================================================================= 部品の特性（表）
# --- 燃焼ガス（LOX/RP-1）。O/F ≥ 0.8 は化学平衡計算（NASA CEA、燃焼圧 9.7 MPa。cea_tables.py で作り直せます）
#     O/F 0.303〜0.516 は燃料過剰の GG 領域。平衡計算（O/F 0.32 で約 1,200 K）は教科書の代表値（約 900 K）と大きく違うので、教科書の値を使用
#     （Huzel & Huang, "Design of Liquid Propellant Rocket Engines", NASA SP-125, Table 6-4。それより下と、0.8 までのあいだは点をつないで補った値）
_OF_GG = [0.303, 0.320, 0.337, 0.354, 0.372, 0.390, 0.408, 0.425, 0.443, 0.460, 0.478, 0.497, 0.516]
_OF_CEA = [0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.2, 2.36, 2.5, 2.8, 3.0, 3.4, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 100.0]
T_AD = Table1D([0.0, 0.1, 0.2] + _OF_GG + _OF_CEA,
               [300, 550, 705] + [867, 894, 922, 950, 978, 1005, 1033, 1061, 1089, 1117, 1144, 1172, 1200]
               + [1488, 1610, 1912, 2527, 3021, 3376, 3560, 3652, 3703, 3752, 3758, 3742, 3683, 3554, 3414, 3121, 2825, 2181, 1739, 1223, 935, 616, 352],
               "燃焼温度 T_ad(O/F) [K]")
GG_CP = Table1D(_OF_GG, [2659, 2675, 2692, 2705, 2713, 2726, 2734, 2742, 2751, 2759, 2763, 2768, 2772], "GG ガスの定圧比熱 [J/kg/K]")
GG_GAMMA = Table1D(_OF_GG, [1.097, 1.100, 1.106, 1.111, 1.115, 1.119, 1.124, 1.128, 1.132, 1.137, 1.140, 1.144, 1.148], "GG ガスの比熱比")


def _cstar_fuel_rich(of):
    """燃料過剰側の c* は、上の GG ガスの性質から c* = √(RT)/Γ で計算する"""
    cp, gam = GG_CP(of), GG_GAMMA(of)
    return math.sqrt(cp * (gam - 1) / gam * T_AD(of)) / crit_flow(gam)


CSTAR = Table1D([0.0] + _OF_GG + _OF_CEA, [_cstar_fuel_rich(of) for of in [0.0] + _OF_GG]
                + [1331, 1398, 1497, 1650, 1745, 1797, 1813, 1814, 1808, 1786, 1767, 1730, 1679, 1605, 1543, 1438, 1344, 1154, 1022, 850, 738, 593, 446],
                "特性排気速度 c*(O/F) [m/s]")
CF_VAC = Table1D([1.5, 2.0, 2.2, 2.36, 2.5, 2.8, 3.4], [1.749, 1.783, 1.805, 1.823, 1.838, 1.860, 1.870], "真空推力係数 CF_vac(O/F)（開口比 16）")


def gg_gas(Y):
    """GG の中のガス（酸化剤の質量分率 Y）の定圧比熱 cp [J/kg/K]、比熱比、ガス定数 R [J/kg/K]"""
    of = Y / max(1.0 - Y, 1e-9)
    cp, gam = GG_CP(of), GG_GAMMA(of)
    return cp, gam, cp * (gam - 1) / gam

# --- ポンプ（相似則で無次元化。Qn = (Q/ω)/(Q_d/ω_d)）: 揚程 ΔP = ΔP_d (ω/ω_d)^2 g(Qn)、軸動力 W = W_d (ω/ω_d)^3 p(Qn)
_QN = [0.0, 0.3, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.75]
PUMP_HEAD = Table1D(_QN, [1.25, 1.24, 1.18, 1.10, 1.00, 0.85, 0.62, 0.30, 0.00], "ポンプ揚程マップ g(Qn)")
PUMP_POWER = Table1D(_QN, [0.45, 0.60, 0.78, 0.89, 1.00, 1.09, 1.15, 1.18, 1.19], "ポンプ軸動力マップ p(Qn)")   # 締切りでも約45%食う

# --- タービン（衝動タービン）: 効率は速度比 x = (U/C0)/(U/C0)_設計 の放物線 η/η_d = 2x − x²。設計点（x = 1）で最大
def turbine_eta_ratio(x):
    """タービン効率マップ η/η_d。粗い表にして直線でつなぐと設計点に角ができるので、式のまま計算する。x ≤ 0 と x ≥ 2 では 0"""
    return max(x * (2.0 - x), 0.0)


# ============================================================================= 設計点（静的なつり合い）
DESIGN = dict(F_sl=845e3, Pc=9.7e6, OF_c=2.36, eps=16.0,        # Merlin 1D の公表値（推力・燃焼圧・開口比）と主燃焼室の混合比
              eta_cstar=0.975, eta_cf=0.98,                     # 燃焼効率（c* 効率）とノズル効率（仮定）
              eta_pump=0.75, eta_turb=0.45, T_gg=900.0,         # ポンプ効率、GG タービンの効率と入口温度（仮定）
              k_discharge=1.4, k_gg=0.8, P_exh=2.0e5, P_tank=3.0e5,
              rpm=36000.0, L_star=1.0, V_gg=0.004, J=0.06)      # 回転数、燃焼室の特性長さ L*、GG 容積、軸の慣性（仮定）

# 教科書の例題エンジン（Huzel & Huang, NASA SP-125, Table 3-2 の A-1 エンジン。LOX/RP-1 の GG サイクル）のターボポンプの値。
# 吐出圧は燃焼圧（ノズル入口のよどみ圧 1,000 psia）に対する比: LOX 1,505 psia、燃料 1,720 psia。
# ポンプ効率 70.7 % / 65.8 %、タービン効率 58.2 %、タービン入口温度 1,400 °F（1,033 K）。design_point(**HH_A1) で置き換えて計算できる
HH_A1 = dict(k_discharge_o=1.505, k_discharge_f=1.72, eta_pump_o=0.707, eta_pump_f=0.658, eta_turb=0.582, T_gg=1033.0)


def design_point(**kw):
    """ポンプ動力 = タービン動力 になる GG 流量を求め、スロートやオリフィスの大きさを決める（時間変化なしの計算）。"""
    d = dict(DESIGN, **kw)
    assert d["eps"] == 16.0 and d["OF_c"] == 2.36, "CF_VAC の表は開口比 16、主燃焼室のガス定数（GGEngine.R_c）は O/F 2.36 の値です"
    Pc, Pgg = d["Pc"], d["k_gg"] * d["Pc"]
    # ポンプの吐出圧と効率は、LOX 側（_o）と燃料側（_f）を別々にも指定できる（省略すると共通の値）
    Pd_o, Pd_f = d.get("k_discharge_o", d["k_discharge"]) * Pc, d.get("k_discharge_f", d["k_discharge"]) * Pc
    eta_o, eta_f = d.get("eta_pump_o", d["eta_pump"]), d.get("eta_pump_f", d["eta_pump"])
    # GG: 入口温度 T_gg になる O/F（燃料過剰側）と、そのときのガスの性質
    assert 600.0 < d["T_gg"] < 1200.0, "T_gg は GG の表の範囲（600〜1200 K）で指定してください"
    OF_g = bisect(lambda of: T_AD(of) - d["T_gg"], 0.1, 0.516)
    cp, gam = GG_CP(OF_g), GG_GAMMA(OF_g)
    R_g = cp * (gam - 1) / gam
    dh = cp * d["T_gg"] * (1 - (d["P_exh"] / Pgg) ** ((gam - 1) / gam))      # タービンで使える熱落差 [J/kg]
    w_t = d["eta_turb"] * dh                                                 # GG ガス 1 kg あたりの軸仕事
    T_out = d["T_gg"] - w_t / cp
    v_exh = math.sqrt(2 * cp * T_out * (1 - (P_ATM / d["P_exh"]) ** ((gam - 1) / gam)))   # タービン排気を大気圧まで膨張
    # ポンプ仕事（推進剤 1 kg あたり）: 主燃焼室向けと GG 向け
    dP_o, dP_f = Pd_o - d["P_tank"], Pd_f - d["P_tank"]
    pump = lambda of: of / (1 + of) * dP_o / (RHO_O * eta_o) + 1 / (1 + of) * dP_f / (RHO_F * eta_f)
    r = pump(d["OF_c"]) / (w_t - pump(OF_g))                                 # GG 流量 ÷ 主燃焼室流量
    # 推力から流量とスロート面積を決める
    cstar = d["eta_cstar"] * CSTAR(d["OF_c"])
    cf_vac = d["eta_cf"] * CF_VAC(d["OF_c"])
    cf_sl = cf_vac - d["eps"] * P_ATM / Pc
    m_c = d["F_sl"] / (cstar * cf_sl + r * v_exh)
    m_g = r * m_c
    A_t = m_c * cstar / Pc
    mf_c, mo_c = m_c / (1 + d["OF_c"]), m_c * d["OF_c"] / (1 + d["OF_c"])
    mf_g, mo_g = m_g / (1 + OF_g), m_g * OF_g / (1 + OF_g)
    Qd_f, Qd_o = (mf_c + mf_g) / RHO_F, (mo_c + mo_g) / RHO_O
    T_e = T_out * (P_ATM / d["P_exh"]) ** ((gam - 1) / gam)                   # タービン排気ノズルの出口（大気圧まで膨張）の温度
    A_exh = m_g * R_g * T_e / (P_ATM * v_exh)                                # 排気ノズルの出口面積（ṁ = ρ v A、ρ = P/(R T)）
    F_vac = d["F_sl"] + P_ATM * (d["eps"] * A_t + A_exh)                     # 真空では、主ノズルと排気ノズルの出口で背圧の分だけ推力が増える
    Wd_f, Wd_o = Qd_f * dP_f / eta_f, Qd_o * dP_o / eta_o
    return dict(d, Pd=Pd_o, Pd_o=Pd_o, Pd_f=Pd_f, Pgg=Pgg, OF_g=OF_g, cp_g=cp, gam_g=gam, R_g=R_g, dh=dh, v_exh=v_exh,
                m_c=m_c, m_g=m_g, mf_c=mf_c, mo_c=mo_c, mf_g=mf_g, mo_g=mo_g, gg_frac=m_g / (m_c + m_g),
                OF_engine=(mo_c + mo_g) / (mf_c + mf_g),
                A_t=A_t, D_t=2 * math.sqrt(A_t / math.pi), A_e=d["eps"] * A_t, V_c=d["L_star"] * A_t, A_exh=A_exh,
                A_tn=m_g * math.sqrt(R_g * d["T_gg"]) / (crit_flow(gam) * Pgg),         # タービンノズルののど面積
                CdA_inj_f=mf_c / math.sqrt(2 * RHO_F * (Pd_f - Pc)), CdA_inj_o=mo_c / math.sqrt(2 * RHO_O * (Pd_o - Pc)),
                CdA_gg_f=mf_g / math.sqrt(2 * RHO_F * (Pd_f - Pgg)), CdA_gg_o=mo_g / math.sqrt(2 * RHO_O * (Pd_o - Pgg)),
                Qd_f=Qd_f, Qd_o=Qd_o, dPd_f=dP_f, dPd_o=dP_o, Wd_f=Wd_f, Wd_o=Wd_o,
                W_pump=Wd_f + Wd_o, omega_d=d["rpm"] * 2 * math.pi / 60,
                Isp_tc=cstar * cf_sl / G0, Isp_sl=d["F_sl"] / ((m_c + m_g) * G0), Isp_vac=F_vac / ((m_c + m_g) * G0), F_vac=F_vac)


# ============================================================================= 動的モデル
class GGEngine(Model):
    def __init__(self, gg_lox_scale=1.0, mcc_lox_scale=1.0, A_gg_orifice_scale=1.0, gg_lox_lag=0.0, main_lox_lag=0.0,
                 gg_valve_t=0.15, main_valve_t=(0.20, 0.50), spin_torque=250.0, spin_time=0.5,
                 cutoff_t=None, gg_cut_t=(0.0, 0.05), main_cut_t=(0.0, 0.15), gg_lox_cut_lag=-0.02, main_lox_cut_lag=-0.02,
                 eta_turb=None, burn_time=150.0, T_init=300.0, design=None):
        d = self.d = design or design_point()
        for k in ("A_t", "A_e", "V_c", "V_gg", "A_tn", "J", "P_exh", "P_tank", "omega_d", "dPd_f", "dPd_o", "Qd_f", "Qd_o", "Wd_f", "Wd_o", "dh"):
            setattr(self, k, d[k])
        # ハードウェア（設計点の寸法 × 製作誤差などを模擬する倍率）
        self.CdA_inj_f, self.CdA_inj_o = d["CdA_inj_f"], d["CdA_inj_o"] * mcc_lox_scale
        self.CdA_gg_f, self.CdA_gg_o = d["CdA_gg_f"] * A_gg_orifice_scale, d["CdA_gg_o"] * A_gg_orifice_scale * gg_lox_scale
        self.eta_turb = d["eta_turb"] if eta_turb is None else eta_turb
        self.R_c = 8314.46 / 22.66                                # 主燃焼室ガスのガス定数 [J/kg/K]（分子量 22.66、CEA）
        # タンク: 設計流量 × burn_time 秒ぶんを積む
        self.m_f0, self.m_o0 = (d["mf_c"] + d["mf_g"]) * burn_time, (d["mo_c"] + d["mo_g"]) * burn_time
        # 起動シーケンス
        self.spin_torque, self.spin_time = spin_torque, spin_time
        self.gg_valve_t, self.main_valve_t = gg_valve_t, main_valve_t
        self.gg_lox_lag, self.main_lox_lag = gg_lox_lag, main_lox_lag
        # 停止シーケンス（cutoff_t = None なら停止しない）。弁が動くのは必ず指令のあと
        assert 0.0 <= gg_cut_t[0] < gg_cut_t[1] and 0.0 <= main_cut_t[0] < main_cut_t[1], "gg_cut_t / main_cut_t は (閉じ始め, 閉じ切り) を 0 ≤ 閉じ始め < 閉じ切り で指定してください"
        self.cutoff_t, self.gg_cut_t, self.main_cut_t = cutoff_t, gg_cut_t, main_cut_t
        self.gg_lox_cut_lag, self.main_lox_cut_lag = gg_lox_cut_lag, main_lox_cut_lag
        self.T_init = T_init

    # ---------------- 弁の開度（0〜1） ----------------
    def closing(self, t, cut_t, lag):
        """停止の指令（cutoff_t）のあと、弁が閉じていく割合（1 = 開いたまま、0 = 全閉）。lag ≥ 0 はその弁だけ遅らせる時間"""
        if self.cutoff_t is None:
            return 1.0
        c0, c1 = cut_t
        return 1.0 - min(max((t - self.cutoff_t - lag - c0) / (c1 - c0), 0.0), 1.0)

    def valve_main(self, t, lox=False):
        t0, t1 = self.main_valve_t
        lag, cut_lag = (self.main_lox_lag, max(self.main_lox_cut_lag, 0.0)) if lox else (0.0, max(-self.main_lox_cut_lag, 0.0))
        return min(min(max((t - t0 - lag) / (t1 - t0), 0.0), 1.0), self.closing(t, self.main_cut_t, cut_lag))

    def valve_gg(self, t, lox=False):
        lag, cut_lag = (self.gg_lox_lag, max(self.gg_lox_cut_lag, 0.0)) if lox else (0.0, max(-self.gg_lox_cut_lag, 0.0))
        return min(min(max((t - self.gg_valve_t - lag) / 0.1, 0.0), 1.0), self.closing(t, self.gg_cut_t, cut_lag))

    # ---------------- 部品：ポンプの動作点 ----------------
    def pump_point(self, rho, Qd, dPd, w, cda_c, cda_g, Pc, Pgg, has_liquid):
        """ポンプ曲線（回転数で伸び縮みする）と、下流の弁・噴射器の抵抗曲線の交点。戻り値: 流量 Q、吐出圧 Pd"""
        if not has_liquid:
            return 0.0, self.P_tank
        Pd = lambda Q: self.P_tank + dPd * w * w * PUMP_HEAD(Q / (Qd * w))

        def excess(Q):                                       # ポンプが送り出す流量 − 下流が飲み込む流量
            P = Pd(Q)
            return rho * Q - cda_c * math.sqrt(2 * rho * max(P - Pc, 0.0)) - cda_g * math.sqrt(2 * rho * max(P - Pgg, 0.0))
        Q = bisect(excess, 0.0, Qd * max(1.75 * w, 0.3), 24)
        return Q, Pd(Q)

    @staticmethod
    def of_ratio(mo, mf):
        return min(mo / max(mf, 1e-9), 100.0)

    def flows(self, t, s):
        Pc, Pgg, Tc, Tgg = s["Pc"], s["Pgg"], s["Tc"], s["Tgg"]
        w = s["omega"] / self.omega_d
        xm_f, xm_o = self.valve_main(t), self.valve_main(t, lox=True)
        xg_f, xg_o = self.valve_gg(t), self.valve_gg(t, lox=True)
        has_f, has_o = s["m_f"] > 0, s["m_o"] > 0
        if not (has_f and has_o):                              # どちらかのタンクが空 → 全弁を閉じてエンジン停止
            xm_f = xm_o = xg_f = xg_o = 0.0
        # --- ポンプの動作点 → 各経路の流量
        Q_f, Pd_f = self.pump_point(RHO_F, self.Qd_f, self.dPd_f, w, xm_f * self.CdA_inj_f, xg_f * self.CdA_gg_f, Pc, Pgg, has_f)
        Q_o, Pd_o = self.pump_point(RHO_O, self.Qd_o, self.dPd_o, w, xm_o * self.CdA_inj_o, xg_o * self.CdA_gg_o, Pc, Pgg, has_o)
        orifice = lambda cda, rho, dP: cda * math.sqrt(2 * rho * max(dP, 0.0))
        mf_c, mf_g = (orifice(xm_f * self.CdA_inj_f, RHO_F, Pd_f - Pc), orifice(xg_f * self.CdA_gg_f, RHO_F, Pd_f - Pgg)) if has_f else (0.0, 0.0)
        mo_c, mo_g = (orifice(xm_o * self.CdA_inj_o, RHO_O, Pd_o - Pc), orifice(xg_o * self.CdA_gg_o, RHO_O, Pd_o - Pgg)) if has_o else (0.0, 0.0)
        W_f = self.Wd_f * w ** 3 * PUMP_POWER(Q_f / (self.Qd_f * w)) if has_f else 0.0
        W_o = self.Wd_o * w ** 3 * PUMP_POWER(Q_o / (self.Qd_o * w)) if has_o else 0.0
        # --- GG とタービン
        of_g = self.of_ratio(mo_g, mf_g)                       # GG に流れ込む推進剤の O/F（燃焼温度はこれで決まる）
        cp, gam, R_g = gg_gas(s["Ygg"])                        # タービンへ流れる、GG の中のガスの性質（組成は流入に遅れて追従）
        m_t = crit_flow(gam) * Pgg * self.A_tn / math.sqrt(R_g * Tgg) * unchoke(self.P_exh, Pgg, gam)
        dh = cp * Tgg * (1 - (self.P_exh / Pgg) ** ((gam - 1) / gam)) if Pgg > self.P_exh else 0.0
        x = w * math.sqrt(self.dh / dh) if dh > 0 else 2.0             # 速度比（設計点で 1）
        eta_t = self.eta_turb * turbine_eta_ratio(x)
        W_t = m_t * dh * eta_t
        T_out = Tgg - eta_t * dh / cp
        F_exh = m_t * math.sqrt(2 * cp * T_out * (1 - (P_ATM / self.P_exh) ** ((gam - 1) / gam)))
        # --- 主燃焼室とノズル
        of_c = self.of_ratio(mo_c, mf_c)
        cstar = CSTAR(of_c) * math.sqrt(Tc / T_AD(of_c))
        m_c_out = Pc * self.A_t / cstar * unchoke(P_ATM, Pc, GAMMA_C)
        F = max(self.d["eta_cf"] * CF_VAC(of_c) * Pc * self.A_t - P_ATM * self.A_e, 0.0) + F_exh
        return dict(mf_c=mf_c, mo_c=mo_c, mf_g=mf_g, mo_g=mo_g, m_c_out=m_c_out, m_t=m_t, of_c=of_c, of_g=of_g, R_g=R_g,
                    W_f=W_f, W_o=W_o, W_t=W_t, eta_t=eta_t, F=F, F_exh=F_exh, Pd_f=Pd_f, Pd_o=Pd_o)

    # ---------------- 状態方程式 ----------------
    def initial_state(self):
        return {"m_f": self.m_f0, "m_o": self.m_o0, "omega": 50.0, "Pc": P_ATM, "Pgg": P_ATM, "Tc": self.T_init, "Tgg": self.T_init, "Ygg": 0.0}

    @staticmethod
    def chamber(P, T, V, R, m_in, m_out, T_target, dlnR=0.0):
        """0D の容器: 温度は燃焼温度にガスの入れ替わり時間 τ で追従、圧力はガスの質量保存（P = mRT/V の時間微分）。
        dlnR はガス定数の変化率 d(ln R)/dt（ガスの組成が変わる GG だけ。主燃焼室は R 一定）"""
        tau = (P * V / (R * T)) / m_in if m_in > 1e-6 else math.inf
        dT = (T_target - T) / tau
        dP = R * T / V * (m_in - m_out) + P / T * dT + P * dlnR
        return dP, dT

    def derivatives(self, t, s):
        f = self.flows(t, s)            # ポンプの動作点・各流量・動力
        # 軸の運動方程式
        torque_spin = self.spin_torque if t < self.spin_time else 0.0
        torque = (f["W_t"] - f["W_f"] - f["W_o"]) / s["omega"] + torque_spin
        # 主燃焼室と GG（容器のモデル）
        T_c = self.d["eta_cstar"] ** 2 * T_AD(f["of_c"])
        dPc, dTc = self.chamber(s["Pc"], s["Tc"], self.V_c, self.R_c,
                                f["mf_c"] + f["mo_c"], f["m_c_out"], T_c)
        m_in_g = f["mf_g"] + f["mo_g"]
        m_gg = s["Pgg"] * self.V_gg / (f["R_g"] * s["Tgg"])                  # GG の中のガスの質量
        dY = m_in_g * (f["mo_g"] / m_in_g - s["Ygg"]) / m_gg if m_in_g > 1e-6 else 0.0   # 流れ込んだ推進剤が混ざって組成が変わる
        dlnR = (gg_gas(s["Ygg"] + 1e-6)[2] / f["R_g"] - 1.0) / 1e-6 * dY        # 組成が変わるとガス定数も変わる
        dPgg, dTgg = self.chamber(s["Pgg"], s["Tgg"], self.V_gg, f["R_g"], m_in_g, f["m_t"], T_AD(f["of_g"]), dlnR)
        return {"m_f": -(f["mf_c"] + f["mf_g"]),        # タンクの質量保存
                "m_o": -(f["mo_c"] + f["mo_g"]),
                "omega": torque / self.J,
                "Pc": dPc, "Pgg": dPgg, "Tc": dTc, "Tgg": dTgg, "Ygg": dY}

    def clamp(self, s):
        s["m_f"], s["m_o"] = max(s["m_f"], 0.0), max(s["m_o"], 0.0)
        s["Pc"], s["Pgg"] = max(s["Pc"], P_ATM), max(s["Pgg"], P_ATM)
        s["omega"] = max(s["omega"], 1.0)
        s["Tc"], s["Tgg"] = min(max(s["Tc"], 250.0), 4000.0), min(max(s["Tgg"], 250.0), 4000.0)
        s["Ygg"] = min(max(s["Ygg"], 0.0), 0.999)
        return s

    def outputs(self, t, s):
        f = self.flows(t, s)
        m_c, m_g = f["mf_c"] + f["mo_c"], f["mf_g"] + f["mo_g"]
        return dict(F_kN=f["F"] / 1e3, Isp=f["F"] / (max(m_c + m_g, 1e-6) * G0), rpm=s["omega"] * 60 / (2 * math.pi),
                    Pc_MPa=s["Pc"] / 1e6, Pgg_MPa=s["Pgg"] / 1e6, Pd_f_MPa=f["Pd_f"] / 1e6, Pd_o_MPa=f["Pd_o"] / 1e6,
                    mdot_c=m_c, mdot_gg=m_g, gg_pct=100 * m_g / max(m_c + m_g, 1e-6),
                    OF_mcc=f["of_c"], OF_gg=f["of_g"], OF_gg_gas=s["Ygg"] / (1.0 - s["Ygg"]), OF_engine=self.of_ratio(f["mo_c"] + f["mo_g"], f["mf_c"] + f["mf_g"]),
                    T_c_K=s["Tc"], T_gg_K=s["Tgg"], W_t_MW=f["W_t"] / 1e6, W_p_MW=(f["W_f"] + f["W_o"]) / 1e6, eta_t=f["eta_t"],
                    mdot_c_out=f["m_c_out"], mdot_t=f["m_t"],                                    # 容器から出ていく流量（ノズル・タービン）
                    m_gas_c=s["Pc"] * self.V_c / (self.R_c * s["Tc"]), m_gas_gg=s["Pgg"] * self.V_gg / (f["R_g"] * s["Tgg"]),   # 容器の中のガスの質量
                    valve_main=self.valve_main(t), valve_main_lox=self.valve_main(t, lox=True), valve_gg=self.valve_gg(t), valve_gg_lox=self.valve_gg(t, lox=True))


if __name__ == "__main__":
    from onedsim import simulate
    d = design_point()
    print(f"設計計算: スロート径 {d['D_t']*100:.1f} cm, 主燃焼室 {d['m_c']:.1f} kg/s, GG {d['m_g']:.2f} kg/s（全体の {d['gg_frac']*100:.1f} %, O/F {d['OF_g']:.3f}）, "
          f"ポンプ動力 {d['W_pump']/1e6:.2f} MW, 比推力 海面 {d['Isp_sl']:.1f} s / 真空 {d['Isp_vac']:.1f} s")
    h = design_point(**HH_A1)
    print(f"教科書の例題エンジンのターボポンプの値に置き換えた設計計算: GG {h['gg_frac']*100:.1f} %, ポンプ動力 {h['W_pump']/1e6:.2f} MW, "
          f"比推力 海面 {h['Isp_sl']:.1f} s / 真空 {h['Isp_vac']:.1f} s（公表値 282 s / 311 s）")
    T_CUT = 3.0                                             # 3 秒で停止の指令を出す
    res = simulate(GGEngine(cutoff_t=T_CUT), t_end=T_CUT + 1.0, dt=2e-4, method="rk4", record_every=10)
    run, cut = res["t"] < T_CUT, res["t"] >= T_CUT
    i = int(run.sum()) - 1                                  # 停止の指令の直前
    print(f"t = 3 s: 推力 {res['F_kN'][i]:.0f} kN, Pc {res['Pc_MPa'][i]:.2f} MPa, {res['rpm'][i]:.0f} rpm（起動中のピーク {res['rpm'][run].max():.0f}）, 比推力 {res['Isp'][i]:.1f} s, "
          f"GG {res['gg_pct'][i]:.2f} %, O/F 主燃焼室 {res['OF_mcc'][i]:.3f} / GG {res['OF_gg'][i]:.3f}, Tc {res['T_c_K'][i]:.0f} K, Tgg {res['T_gg_K'][i]:.0f} K, "
          f"タービン {res['W_t_MW'][i]:.2f} MW / ポンプ {res['W_p_MW'][i]:.2f} MW")
    F, tc = res["F_kN"][cut], res["t"][cut] - T_CUT
    print(f"停止（3 s に指令）: {crossing_time(tc, F, 0.01 * F[0], rising=False):.2f} s で推力が 1 % 以下, 指令のあとの回転数のピーク {res['rpm'][cut].max():.0f} rpm, "
          f"1 s 後 {res['rpm'][-1]:.0f} rpm, GG 出口温度のピーク {res['T_gg_K'][cut].max():.0f} K")
