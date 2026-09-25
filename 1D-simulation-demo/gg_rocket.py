"""
ガスジェネレータサイクル液体ロケットエンジンの1Dモデル（onedsim デモ②）
====================================================================
Merlin 1D 級（海面推力 845 kN、燃焼室圧 9.7 MPa、LOX/RP-1）を目標に、
タンク → ターボポンプ → 燃焼室 / ガスジェネレータ（GG）→ タービン を1軸でつないだ系を解きます。

部品（すべて「特性 + 保存則」だけ）:
  タンク       : 圧力一定（調圧式）。質量が状態量。枯渇で停止
  ポンプ       : 相似則で無次元化した揚程マップ g(Qn)・効率 η(Qn)（試験で取るもの）。ΔP = ΔP_d (ω/ω_d)^2 g(Qn)
  ターボポンプ軸: 慣性 J。 J dω/dt = (P_turbine − P_pumps) / ω
  噴射器・オリフィス : ṁ = Cd A √(2ρΔP)
  燃焼室 / GG  : 圧力が状態量。 dP/dt = (RT/V)(ṁ_in − ṁ_out)、ṁ_out = P A_t / c*
  タービン     : P = ṁ_gg cp T_gg (1 − PR^(−(γ−1)/γ)) η_t
  推力         : F = Cf Pc A_t

状態量: m_f, m_o [kg], omega [rad/s], Pc [Pa], Pgg [Pa]
起動シーケンス: t=0 スピンスタート（0.5 s の起動トルク）→ 0.15 s GG弁 → 0.3–0.9 s 主弁ランプ開
"""
from __future__ import annotations
import numpy as np
from onedsim import Model, Table1D

G0 = 9.80665


class GGEngine(Model):
    def __init__(self, eta_turb=0.58, A_gg_orifice_scale=1.0, A_throat=0.0544, J_shaft=0.06,
                 P_tank_f=3.0e5, P_tank_o=3.0e5, m_f0=3.0e3, m_o0=7.0e3,
                 spin_torque=250.0, spin_time=0.5, gg_valve_t=0.15, main_valve_t=(0.30, 0.90),
                 c_star=1730.0, Cf=1.60, T_gg=1000.0, cp_gg=2000.0, gamma_gg=1.20, R_gg=330.0,
                 R_c=340.0, T_c=3500.0, V_c=0.06, V_gg=0.004, A_gg_throat=0.0018):
        # 推進剤
        self.rho_f, self.rho_o = 810.0, 1140.0
        self.P_tank_f, self.P_tank_o = P_tank_f, P_tank_o
        self.m_f0, self.m_o0 = m_f0, m_o0
        # ポンプマップ（相似則で無次元化した試験曲線）:
        #   Qn = (Q/ω)/(Q_d/ω_d),  ΔP = ΔP_d (ω/ω_d)^2 g(Qn),  η = η(Qn)
        self.g_map = Table1D([0.0, 0.3, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6], [1.25, 1.24, 1.18, 1.10, 1.00, 0.85, 0.62, 0.30], "pump head map")
        self.eta_p = Table1D([0.0, 0.3, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6], [0.05, 0.35, 0.60, 0.72, 0.76, 0.72, 0.60, 0.40], "pump eta map")
        self.omega_design = 36000 * 2 * np.pi / 60
        # 設計点（Merlin 1D 級）: ṁ_f=91, ṁ_o=214 kg/s, 吐出圧 12.7 MPa, Pc 9.7 MPa
        self.Qd_f, self.Qd_o = 91.0 / self.rho_f, 214.0 / self.rho_o
        self.dPd_f, self.dPd_o = 12.4e6, 12.4e6
        self.J = J_shaft
        # 噴射器（主室）: 設計 ΔP 3.0 MPa
        self.CdA_inj_f = 91.0 / np.sqrt(2 * self.rho_f * 3.0e6)
        self.CdA_inj_o = 214.0 / np.sqrt(2 * self.rho_o * 3.0e6)
        # GG 供給オリフィス: 設計 ṁ_gg≈11 kg/s（O/F_gg≈0.36, 燃料過剰で約 1000 K）, Pgg 5 MPa
        self.CdA_gg_f = A_gg_orifice_scale * 8.1 / np.sqrt(2 * self.rho_f * 7.7e6)
        self.CdA_gg_o = A_gg_orifice_scale * 2.9 / np.sqrt(2 * self.rho_o * 7.7e6)
        # 燃焼室 / GG
        self.c_star, self.Cf, self.A_t, self.R_c, self.T_c, self.V_c = c_star, Cf, A_throat, R_c, T_c, V_c
        self.c_star_gg = np.sqrt(R_gg * T_gg) * 1.4    # 簡易
        self.A_gg_t, self.V_gg, self.T_gg, self.cp_gg, self.gamma_gg, self.R_gg = A_gg_throat, V_gg, T_gg, cp_gg, gamma_gg, R_gg
        self.eta_turb = eta_turb
        self.P_exhaust = 1.0e5 * 2.0            # タービン出口圧（ダンプ）
        # 起動シーケンス
        self.spin_torque, self.spin_time = spin_torque, spin_time
        self.gg_valve_t, self.main_valve_t = gg_valve_t, main_valve_t
        self.P_amb = 1.013e5

    # ---------------- バルブ開度（0-1） ----------------
    def valve_main(self, t):
        t0, t1 = self.main_valve_t
        return float(np.clip((t - t0) / (t1 - t0), 0.0, 1.0))

    def valve_gg(self, t):
        return float(np.clip((t - self.gg_valve_t) / 0.1, 0.0, 1.0))

    # ---------------- 部品：ポンプ ----------------
    def pump(self, Qd, dPd, omega, Q):
        """相似則: 回転数比の2乗で揚程、流量係数でマップを引く。"""
        w = omega / self.omega_design
        Qn = (Q / max(omega, 1.0)) / (Qd / self.omega_design)
        dP = dPd * w ** 2 * self.g_map(Qn)
        eta = max(self.eta_p(Qn), 0.05)
        return dP, Q * dP / eta

    def flows(self, t, s, omega=None):
        """代数ループ（ポンプ吐出圧 ⇄ 流量）を数回の反復で解く。"""
        omega = s["omega"] if omega is None else omega
        Pc, Pgg = s["Pc"], s["Pgg"]
        vm, vg = self.valve_main(t), self.valve_gg(t)
        Q_f, Q_o = 0.05, 0.10
        for _ in range(6):
            dP_f, W_f = self.pump(self.Qd_f, self.dPd_f, omega, Q_f)
            dP_o, W_o = self.pump(self.Qd_o, self.dPd_o, omega, Q_o)
            Pd_f, Pd_o = self.P_tank_f + dP_f, self.P_tank_o + dP_o
            mf_c = vm * self.CdA_inj_f * np.sqrt(2 * self.rho_f * max(Pd_f - Pc, 0.0))
            mo_c = vm * self.CdA_inj_o * np.sqrt(2 * self.rho_o * max(Pd_o - Pc, 0.0))
            mf_g = vg * self.CdA_gg_f * np.sqrt(2 * self.rho_f * max(Pd_f - Pgg, 0.0))
            mo_g = vg * self.CdA_gg_o * np.sqrt(2 * self.rho_o * max(Pd_o - Pgg, 0.0))
            Q_f = 0.5 * Q_f + 0.5 * (mf_c + mf_g) / self.rho_f
            Q_o = 0.5 * Q_o + 0.5 * (mo_c + mo_g) / self.rho_o
        # タービン
        m_gg_out = Pgg * self.A_gg_t / self.c_star_gg
        PR = max(Pgg / self.P_exhaust, 1.0)
        W_t = m_gg_out * self.cp_gg * self.T_gg * (1 - PR ** (-(self.gamma_gg - 1) / self.gamma_gg)) * self.eta_turb
        m_c_out = Pc * self.A_t / self.c_star
        F = self.Cf * Pc * self.A_t if Pc > 1.5 * self.P_amb else 0.0
        return dict(mf_c=mf_c, mo_c=mo_c, mf_g=mf_g, mo_g=mo_g, m_c_out=m_c_out, m_gg_out=m_gg_out,
                    W_f=W_f, W_o=W_o, W_t=W_t, F=F, Pd_f=Pd_f, Pd_o=Pd_o, Q_f=Q_f, Q_o=Q_o)

    # ---------------- 状態方程式 ----------------
    def initial_state(self):
        return {"m_f": self.m_f0, "m_o": self.m_o0, "omega": 50.0, "Pc": self.P_amb, "Pgg": self.P_amb}

    def derivatives(self, t, s):
        f = self.flows(t, s)
        omega = max(s["omega"], 1.0)
        tau_spin = self.spin_torque if t < self.spin_time else 0.0
        domega = (f["W_t"] - f["W_f"] - f["W_o"]) / omega / self.J + tau_spin / self.J
        dPc = self.R_c * self.T_c / self.V_c * (f["mf_c"] + f["mo_c"] - f["m_c_out"])
        dPgg = self.R_gg * self.T_gg / self.V_gg * (f["mf_g"] + f["mo_g"] - f["m_gg_out"])
        empty = s["m_f"] <= 0 or s["m_o"] <= 0
        if empty:                                   # タンク枯渇：燃焼停止
            dPc = -self.R_c * self.T_c / self.V_c * f["m_c_out"]
            dPgg = -self.R_gg * self.T_gg / self.V_gg * f["m_gg_out"]
            domega = -(f["W_f"] + f["W_o"]) / omega / self.J
        return {"m_f": -(f["mf_c"] + f["mf_g"]) if not empty else 0.0,
                "m_o": -(f["mo_c"] + f["mo_g"]) if not empty else 0.0,
                "omega": domega, "Pc": dPc, "Pgg": dPgg}

    def clamp(self, s):
        s["m_f"] = max(s["m_f"], 0.0); s["m_o"] = max(s["m_o"], 0.0)
        s["Pc"] = max(s["Pc"], self.P_amb); s["Pgg"] = max(s["Pgg"], self.P_amb)
        s["omega"] = max(s["omega"], 1.0)
        return s

    def outputs(self, t, s):
        f = self.flows(t, s)
        m_tot = f["mf_c"] + f["mo_c"]
        Isp = f["F"] / (max(m_tot + f["mf_g"] + f["mo_g"], 1e-6) * G0)
        return dict(F_kN=f["F"] / 1e3, Isp=Isp, rpm=s["omega"] * 60 / 2 / np.pi, Pc_MPa=s["Pc"] / 1e6, Pgg_MPa=s["Pgg"] / 1e6,
                    mdot_c=m_tot, mdot_gg=f["mf_g"] + f["mo_g"], OF=f["mo_c"] / max(f["mf_c"], 1e-6),
                    W_t_MW=f["W_t"] / 1e6, W_p_MW=(f["W_f"] + f["W_o"]) / 1e6, Pd_o_MPa=f["Pd_o"] / 1e6,
                    valve_main=self.valve_main(t), valve_gg=self.valve_gg(t))


if __name__ == "__main__":
    from onedsim import simulate
    eng = GGEngine()
    res = simulate(eng, t_end=3.0, dt=2e-4, method="rk4", record_every=10)
    i = -1
    print(f"t=3 s: F={res['F_kN'][i]:.0f} kN, Pc={res['Pc_MPa'][i]:.2f} MPa, rpm={res['rpm'][i]:.0f}, "
          f"Isp={res['Isp'][i]:.0f} s, mdot={res['mdot_c'][i]:.0f} kg/s, O/F={res['OF'][i]:.2f}, "
          f"GG={res['mdot_gg'][i]:.1f} kg/s, Wt={res['W_t_MW'][i]:.2f} MW, Wp={res['W_p_MW'][i]:.2f} MW")
