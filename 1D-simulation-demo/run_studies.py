"""
run_studies.py — 記事用のパラメトリックスタディを一括実行し、図（PNG）と数値（summary.json）を出力
python run_studies.py            # 全部（数分）
python run_studies.py quick      # 揚水のみ
"""
import json, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from onedsim import simulate, sweep
from pumped_storage import PumpedStorage, make_demand, make_rain, DAY
from gg_rocket import GGEngine

from matplotlib import font_manager as _fm
_have = {f.name for f in _fm.fontManager.ttflist}   # 日本語フォントを OS ごとに自動選択（Windows: Yu Gothic / Meiryo）
plt.rcParams["font.family"] = next((f for f in ["Noto Sans CJK JP", "Yu Gothic", "Meiryo", "MS Gothic", "Hiragino Sans", "IPAexGothic"] if f in _have), "sans-serif")
INK, MUTED, LINK, BLUSH, SURF = "#232a42", "#888888", "#008db7", "#fcb8b8", "#fbfbff"
RED, AMBER, GREEN = "#e07a7a", "#e0a850", "#6cb59a"
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig"); os.makedirs(OUT, exist_ok=True)
S = {}


def style(ax, title=None, xl=None, yl=None):
    ax.set_facecolor(SURF); ax.grid(color="#e3e3ea", lw=0.8); ax.tick_params(colors=INK, labelsize=10)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    if title: ax.set_title(title, fontsize=13, fontweight="bold", color=INK, loc="left")
    if xl: ax.set_xlabel(xl, fontsize=11, color=INK)
    if yl: ax.set_ylabel(yl, fontsize=11, color=INK)


def fig(n=1, h=4.0, sharex=True):
    f, ax = plt.subplots(n, 1, figsize=(10.24, h * n), dpi=100, sharex=sharex)
    f.patch.set_facecolor(SURF)
    return f, (ax if n > 1 else [ax])


def save(f, name):
    f.tight_layout(); f.savefig(f"{OUT}/{name}", facecolor=SURF, bbox_inches="tight", pad_inches=0.15); plt.close(f); print("saved", name)


# =============================================================================== 揚水
def study_pumped():
    t0 = time.time()
    m = PumpedStorage(rain=make_rain(storm_day=5), demand=make_demand(heatwave_day=2))
    r = simulate(m, 7 * DAY, 60.0, record_every=5)
    d = r["t"] / DAY
    dE = r["E_stored"][-1] - r["E_stored"][0]
    S["ps_base"] = dict(E_gen=r["E_gen"][-1], E_pump=r["E_pump"][-1], E_spill=r["E_spill"][-1], E_short=r["E_short"][-1],
                        roundtrip=(r["E_gen"][-1] + dE) / r["E_pump"][-1], level_min=r["level_u"].min(), level_max=r["level_u"].max(),
                        H_min=r["H"].min(), H_max=r["H"].max(), Qp=r["Q_p"].max(), Qt=r["Q_t"].max(), P_pump_max=r["P_pump"].max(),
                        run_s=time.time() - t0)
    # --- 図1: 7日間の運転 ---
    f, ax = fig(4, 2.6)
    ax[0].fill_between(d, 0, r["P_gen"], color=LINK, alpha=0.85, label="発電出力")
    ax[0].fill_between(d, 0, -r["P_pump"], color=RED, alpha=0.75, label="揚水入力")
    ax[0].plot(d, np.where(r["P_req"] > 0, r["P_req"], np.nan), color=INK, lw=1, ls=":", label="発電要求")
    ax[0].legend(loc="upper right", fontsize=9, frameon=False, ncol=3); style(ax[0], "電力 [MW]（上：発電、下：揚水）")
    ax[0].set_ylim(-1200, 1200)
    ax[1].plot(d, r["level_u"], color=INK, lw=2); ax[1].axhline(m.V_max / m.A, color=RED, ls="--", lw=1); ax[1].axhline(m.V_min / m.A, color=RED, ls="--", lw=1)
    ax[1].text(0.05, m.V_max / m.A - 1.2, "満水", color=RED, fontsize=9, va="top"); ax[1].text(0.05, m.V_min / m.A + 0.6, "下限（最低水位）", color=RED, fontsize=9)
    style(ax[1], "上池水位 [m]"); ax[1].set_ylim(0, 37)
    ax[2].plot(d, r["E_stored"] / 1e3, color=GREEN, lw=2); style(ax[2], "発電可能な貯蔵エネルギー [GWh]")
    ax[3].fill_between(d, 0, r["rain"], color=LINK, alpha=0.6); style(ax[3], "降雨 [mm/h]", "経過日数（0=月曜）")
    for a in ax:
        a.axvspan(2, 3, color=AMBER, alpha=0.12); a.axvspan(5, 7, color="#dddddd", alpha=0.35)
    ax[1].text(2.5, 34.5, "猛暑日", ha="center", fontsize=9, color=INK); ax[1].text(6.0, 34.5, "週末（低需要）", ha="center", fontsize=9, color=INK)
    ax[3].text(5.15, 25, "豪雨 30 mm/h × 12 h", fontsize=9, color=INK, ha="right")
    save(f, "1dcae2_11_ps_week.png")

    # --- 図2: ポンプ動作点（曲線の交点） ---
    f, ax = fig(1, 4.6, sharex=False)
    Q = np.linspace(0, 80, 200)
    ax[0].plot(Q, m.pump_H(Q), color=INK, lw=2.5, label="ポンプ Q-H 曲線（試験値）")
    for H, c, lab, tx in [(S["ps_base"]["H_min"], LINK, f"系の要求（落差 {S['ps_base']['H_min']:.0f} m）", (58, 635)),
                          (S["ps_base"]["H_max"], RED, f"系の要求（落差 {S['ps_base']['H_max']:.0f} m）", (36, 728))]:
        ax[0].plot(Q, H + m.k_loss * Q ** 2, color=c, lw=2, ls="--", label=lab)
        Qop = m.pump_operating_point(H); ax[0].scatter([Qop], [m.pump_H(Qop)], s=90, color=c, edgecolor=INK, zorder=5)
        ax[0].annotate(f"動作点 Q = {Qop:.1f} m³/s", (Qop, m.pump_H(Qop)), xytext=tx, fontsize=10, color=c, arrowprops=dict(arrowstyle="->", color=c, lw=1))
    ax[0].set_ylim(600, 830); ax[0].set_xlim(0, 80)
    style(ax[0], "動作点は「ポンプ」ではなく「系」が決める", "流量 Q [m³/s]（1台あたり）", "揚程 [m]")
    ax[0].legend(loc="upper right", fontsize=9, frameon=False)
    save(f, "1dcae2_12_ps_operating_point.png")

    # --- 図3: 豪雨強度スイープ → 越流損失・最高水位 ---
    rains = np.arange(0, 101, 10)
    spill, lvmax = [], []
    for mm in rains:
        mm_ = float(mm)
        rr = simulate(PumpedStorage(rain=make_rain(storm_day=5, mm_per_h=mm_, hours=24.0), demand=make_demand(heatwave_day=2)), 7 * DAY, 60.0, record_every=20)
        spill.append(rr["E_spill"][-1]); lvmax.append(rr["level_u"].max())
    S["ps_rain"] = dict(rains=rains.tolist(), spill=spill, lvmax=lvmax)
    f, ax = fig(1, 4.4, sharex=False)
    ax[0].plot(rains, np.array(spill) / 1e3, "o-", color=RED, lw=2, label="越流で失った発電機会 [GWh]")
    style(ax[0], "パラメトリックスタディ①：豪雨強度 vs 越流損失（週末に24時間の豪雨）", "降雨強度 [mm/h]", "損失 [GWh]")
    ax2 = ax[0].twinx(); ax2.plot(rains, lvmax, "s--", color=LINK, lw=1.5, label="上池最高水位 [m]"); ax2.axhline(m.V_max / m.A, color=LINK, ls=":", lw=1)
    ax2.set_ylabel("上池最高水位 [m]", color=LINK); ax2.tick_params(colors=LINK); ax2.spines["top"].set_visible(False)
    h1, l1 = ax[0].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax[0].legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9, frameon=False)
    save(f, "1dcae2_13_ps_sweep_rain.png")

    # --- 図4: 需要ピークスイープ → 供給不足 ---
    peaks = np.arange(600, 1501, 100)
    short, lvmin = [], []
    for pk in peaks:
        rr = simulate(PumpedStorage(rain=make_rain(storm_day=5), demand=make_demand(peak_MW=float(pk), heatwave_day=2)), 7 * DAY, 60.0, record_every=20)
        short.append(rr["E_short"][-1]); lvmin.append(rr["level_u"].min())
    S["ps_peak"] = dict(peaks=peaks.tolist(), short=short, lvmin=lvmin)
    f, ax = fig(1, 4.4, sharex=False)
    ax[0].bar(peaks, np.array(short) / 1e3, width=60, color=RED, alpha=0.85, label="供給不足 [GWh/週]")
    style(ax[0], "パラメトリックスタディ②：ピーク需要 vs 供給不足（2台運転）", "ピーク発電要求 [MW]", "供給不足 [GWh/週]")
    ax2 = ax[0].twinx(); ax2.plot(peaks, lvmin, "o-", color=INK, lw=1.5, label="上池最低水位 [m]"); ax2.axhline(m.V_min / m.A, color=INK, ls=":", lw=1)
    ax2.set_ylabel("上池最低水位 [m]", color=INK); ax2.spines["top"].set_visible(False); ax2.set_ylim(0, 20)
    h1, l1 = ax[0].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax[0].legend(h1 + h2, l1 + l2, loc="upper center", fontsize=9, frameon=False)
    save(f, "1dcae2_14_ps_sweep_demand.png")


# =============================================================================== ロケット
def study_rocket():
    t0 = time.time()
    r = simulate(GGEngine(), 3.0, 2e-4, record_every=5)
    S["gg_base"] = {k: float(r[k][-1]) for k in ["F_kN", "Pc_MPa", "Pgg_MPa", "rpm", "Isp", "mdot_c", "mdot_gg", "OF", "W_t_MW", "W_p_MW", "Pd_o_MPa"]}
    S["gg_base"]["rpm_peak"] = float(r["rpm"].max()); S["gg_base"]["t_rpm_peak"] = float(r["t"][r["rpm"].argmax()])
    S["gg_base"]["t_90F"] = float(r["t"][np.argmax(r["F_kN"] > 0.9 * r["F_kN"][-1])]); S["gg_base"]["run_s"] = time.time() - t0
    S["gg_base"]["gg_pct"] = 100 * S["gg_base"]["mdot_gg"] / (S["gg_base"]["mdot_gg"] + S["gg_base"]["mdot_c"])
    # --- 図5: 起動過渡 ---
    f, ax = fig(4, 2.5)
    t = r["t"]
    ax[0].plot(t, r["valve_gg"], color=RED, lw=2, label="GG弁"); ax[0].plot(t, r["valve_main"], color=LINK, lw=2, label="主弁"); ax[0].axvspan(0, 0.5, color=AMBER, alpha=0.2)
    ax[0].text(0.25, 0.5, "スピン\nスタート", ha="center", va="center", fontsize=9, color=INK); ax[0].legend(loc="lower right", fontsize=9, frameon=False); style(ax[0], "バルブ開度 / 起動シーケンス")
    ax[1].plot(t, r["rpm"] / 1e3, color=INK, lw=2); ax[1].axhline(36, color=MUTED, ls=":", lw=1); ax[1].text(2.95, 31.5, "設計回転数 36,000 rpm", ha="right", fontsize=9, color=MUTED)
    ax[1].annotate(f"オーバーシュート {S['gg_base']['rpm_peak']/1e3:.1f} krpm", (S["gg_base"]["t_rpm_peak"], S["gg_base"]["rpm_peak"] / 1e3), xytext=(1.2, 44), fontsize=10, color=RED, arrowprops=dict(arrowstyle="->", color=RED))
    style(ax[1], "ターボポンプ回転数 [×1000 rpm]"); ax[1].set_ylim(0, 50)
    ax[2].plot(t, r["Pc_MPa"], color=INK, lw=2, label="燃焼室 Pc"); ax[2].plot(t, r["Pgg_MPa"], color=RED, lw=2, label="GG Pgg"); ax[2].plot(t, r["Pd_o_MPa"], color=LINK, lw=1.5, ls="--", label="酸化剤ポンプ吐出圧")
    ax[2].legend(loc="lower right", fontsize=9, frameon=False, ncol=3); style(ax[2], "圧力 [MPa]")
    ax[3].plot(t, r["F_kN"], color=GREEN, lw=2.5); ax[3].axhline(845, color=MUTED, ls=":", lw=1); ax[3].text(2.95, 740, "Merlin 1D 海面推力 845 kN", ha="right", fontsize=9, color=MUTED)
    style(ax[3], "推力 [kN]", "時間 [s]"); ax[3].set_ylim(0, 1000)
    save(f, "1dcae2_21_gg_start.png")

    # --- 図6: タービン効率スイープ ---
    etas = np.arange(0.44, 0.71, 0.02)
    F, Pc, rpm = [], [], []
    for e in etas:
        rr = simulate(GGEngine(eta_turb=float(e)), 2.5, 5e-4, record_every=50)
        F.append(rr["F_kN"][-1]); Pc.append(rr["Pc_MPa"][-1]); rpm.append(rr["rpm"][-1])
    S["gg_eta"] = dict(etas=etas.tolist(), F=F, Pc=Pc, rpm=rpm)
    f, ax = fig(1, 4.4, sharex=False)
    ax[0].plot(etas * 100, F, "o-", color=GREEN, lw=2, label="推力 [kN]"); ax[0].axhline(845, color=MUTED, ls=":", lw=1)
    style(ax[0], "パラメトリックスタディ③：タービン効率 vs 推力・回転数", "タービン断熱効率 [%]", "推力 [kN]")
    for e, Fv, rv in zip(etas[::3], F[::3], rpm[::3]):
        ax[0].annotate(f"{rv/1e3:.1f} krpm", (e * 100, Fv), xytext=(6, -14), textcoords="offset points", fontsize=8, color=MUTED)
    ax[0].text(70, 720, f"効率 {etas[0]*100:.0f}→{etas[-1]*100:.0f}% で推力 {F[0]:.0f}→{F[-1]:.0f} kN、Pc {Pc[0]:.1f}→{Pc[-1]:.1f} MPa\n（推力 ∝ Pc）。点の脇はターボポンプ回転数", ha="right", fontsize=10, color=INK,
               bbox=dict(boxstyle="round,pad=0.4", fc="#ffe8ee", ec="none"))
    ax[0].text(44, 845 + 8, "Merlin 1D 845 kN", fontsize=9, color=MUTED); ax[0].legend(loc="upper left", fontsize=9, frameon=False); ax[0].set_ylim(660, 1010)
    save(f, "1dcae2_22_gg_sweep_eta.png")

    # --- 図7: 主弁開タイミング → 回転数オーバーシュート ---
    t1s = np.arange(0.5, 1.61, 0.1)
    peak, ss = [], []
    for t1 in t1s:
        rr = simulate(GGEngine(main_valve_t=(0.30, float(t1))), 2.5, 5e-4, record_every=10)
        peak.append(rr["rpm"].max()); ss.append(rr["rpm"][-1])
    S["gg_valve"] = dict(t1=t1s.tolist(), peak=peak, ss=ss)
    f, ax = fig(1, 4.4, sharex=False)
    ax[0].plot(t1s, np.array(peak) / 1e3, "o-", color=RED, lw=2, label="回転数ピーク [krpm]"); ax[0].plot(t1s, np.array(ss) / 1e3, "s--", color=INK, lw=1.5, label="定常回転数 [krpm]")
    ax[0].axhline(36 * 1.15, color=RED, ls=":", lw=1); ax[0].text(1.58, 36 * 1.15 - 0.55, "許容上限の例（設計 +15%）", fontsize=9, color=RED, ha="right")
    style(ax[0], "パラメトリックスタディ④：主弁の全開時刻 vs ターボポンプ回転数", "主弁が全開になる時刻 [s]（開き始めは 0.30 s 固定）", "回転数 [×1000 rpm]"); ax[0].legend(loc="upper left", fontsize=9, frameon=False)
    save(f, "1dcae2_23_gg_sweep_valve.png")

    # --- 図8: GG分岐量スイープ → 推力 vs Isp ---
    scales = np.arange(0.7, 1.31, 0.1)
    F2, Isp2, gg = [], [], []
    for sc in scales:
        rr = simulate(GGEngine(A_gg_orifice_scale=float(sc)), 2.5, 5e-4, record_every=50)
        F2.append(rr["F_kN"][-1]); Isp2.append(rr["Isp"][-1]); gg.append(100 * rr["mdot_gg"][-1] / (rr["mdot_gg"][-1] + rr["mdot_c"][-1]))
    S["gg_bleed"] = dict(scales=scales.tolist(), F=F2, Isp=Isp2, gg_pct=gg)
    f, ax = fig(1, 4.4, sharex=False)
    ax[0].plot(gg, F2, "o-", color=GREEN, lw=2, label="推力 [kN]")
    style(ax[0], "パラメトリックスタディ⑤：GG分岐流量 vs 推力・比推力", "GG に分岐する推進剤の割合 [%]", "推力 [kN]")
    ax2 = ax[0].twinx(); ax2.plot(gg, Isp2, "s--", color=INK, lw=1.5, label="比推力 Isp [s]"); ax2.set_ylabel("比推力 [s]"); ax2.spines["top"].set_visible(False)
    h1, l1 = ax[0].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax[0].legend(h1 + h2, l1 + l2, loc="center left", fontsize=9, frameon=False)
    save(f, "1dcae2_24_gg_sweep_bleed.png")

    # --- 図9: Euler vs RK4（裏技） ---
    f, ax = fig(1, 4.2, sharex=False)
    for meth, dt, c, lab in [("rk4", 2e-3, INK, "RK4  dt = 2 ms"), ("euler", 1e-3, LINK, "Euler dt = 1 ms"), ("euler", 2e-3, RED, "Euler dt = 2 ms")]:
        try:
            rr = simulate(GGEngine(), 1.4, dt, method=meth, record_every=1)
            ax[0].plot(rr["t"], rr["Pc_MPa"], color=c, lw=1.8 if meth == "rk4" else 1.4, label=lab)
        except FloatingPointError:
            ax[0].plot([], [], color=c, label=lab + "（発散）")
    ax[0].set_ylim(0, 16); ax[0].legend(loc="upper left", fontsize=10, frameon=False)
    ax[0].annotate("物理ではなく“数値”の振動\n（燃焼室の時定数 ≈ 数 ms より dt が大きい）", (1.27, 12.5), xytext=(0.62, 13.2), fontsize=10, color=RED, arrowprops=dict(arrowstyle="->", color=RED))
    style(ax[0], "裏技：同じモデル・同じ dt でも積分法で結果が変わる（燃焼室圧）", "時間 [s]", "Pc [MPa]")
    save(f, "1dcae2_25_gg_euler_vs_rk4.png")

    # --- 図10: タンク枯渇までの長時間運転 ---
    rr = simulate(GGEngine(m_f0=1200.0, m_o0=2800.0), 20.0, 5e-4, record_every=40)
    S["gg_deplete"] = dict(t_end=float(rr["t"][np.argmax(rr["F_kN"] < 10) if (rr["F_kN"] < 10).any() else -1]))
    f, ax = fig(2, 3.0)
    ax[0].plot(rr["t"], rr["m_f"], color=AMBER, lw=2, label="燃料 [kg]"); ax[0].plot(rr["t"], rr["m_o"], color=LINK, lw=2, label="酸化剤 [kg]"); ax[0].legend(fontsize=9, frameon=False); style(ax[0], "タンク残量")
    ax[1].plot(rr["t"], rr["F_kN"], color=GREEN, lw=2); style(ax[1], "推力 [kN]", "時間 [s]")
    t_off = float(rr["t"][np.argmax((rr["F_kN"] < 10) & (rr["t"] > 1))]); S["gg_deplete"] = dict(t_off=t_off, m_o_left=float(rr["m_o"][-1]))
    ax[0].annotate(f"燃料が先に枯渇 → 酸化剤 {rr['m_o'][-1]:.0f} kg が残る（残推進剤）", (t_off, rr["m_o"][-1]), xytext=(11.5, 1400), fontsize=10, color=INK, ha="center", arrowprops=dict(arrowstyle="->", color=INK))
    ax[1].annotate(f"燃焼停止 t = {t_off:.1f} s", (t_off, 400), xytext=(15.0, 500), fontsize=10, color=INK, arrowprops=dict(arrowstyle="->", color=INK))
    save(f, "1dcae2_26_gg_depletion.png")


if __name__ == "__main__":
    study_pumped()
    if "quick" not in sys.argv:
        study_rocket()
    json.dump(S, open(f"{OUT}/summary.json", "w"), indent=2, ensure_ascii=False, default=float)
    print(json.dumps({k: v for k, v in S.items() if k.endswith("base")}, indent=1, ensure_ascii=False, default=float))
