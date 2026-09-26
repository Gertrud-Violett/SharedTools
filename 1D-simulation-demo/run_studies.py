"""
run_studies.py — 記事用のパラメトリックスタディを一括実行し、図（PNG）と数値（summary.json）を出力
python run_studies.py            # 全部（数分）
python run_studies.py quick      # 揚水のみ
python run_studies.py rocket     # ロケットのみ
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
    S["gg_base"].update(OF_gg=float(r["OF_gg"][-1]), T_gg=float(r["T_gg_K"][-1]), T_c=float(r["T_c_K"][-1]))
    OFD_GG, OFD_C, T_LIM = S["gg_base"]["OF_gg"], S["gg_base"]["OF"], 1100.0   # 設計 O/F、タービン入口温度の上限（目安）

    # --- 図5: 起動過渡 ---
    f, ax = fig(5, 2.4)
    t = r["t"]
    ax[0].plot(t, r["valve_gg"], color=RED, lw=2, label="GG弁（燃料・LOX）"); ax[0].plot(t, r["valve_main"], color=LINK, lw=2, label="主弁（燃料・LOX）"); ax[0].axvspan(0, 0.5, color=AMBER, alpha=0.2)
    ax[0].text(0.40, 0.62, "スピン\nスタート", ha="center", va="center", fontsize=9, color=INK); ax[0].legend(loc="lower right", fontsize=9, frameon=False); style(ax[0], "バルブ開度 / 起動シーケンス")
    ax[1].plot(t, r["rpm"] / 1e3, color=INK, lw=2); ax[1].axhline(36, color=MUTED, ls=":", lw=1); ax[1].text(2.95, 31.5, "設計回転数 36,000 rpm", ha="right", fontsize=9, color=MUTED)
    ax[1].annotate(f"オーバーシュート {S['gg_base']['rpm_peak']/1e3:.1f} krpm", (S["gg_base"]["t_rpm_peak"], S["gg_base"]["rpm_peak"] / 1e3), xytext=(1.2, 44), fontsize=10, color=RED, arrowprops=dict(arrowstyle="->", color=RED))
    style(ax[1], "ターボポンプ回転数 [×1000 rpm]"); ax[1].set_ylim(0, 50)
    ax[2].plot(t, r["Pc_MPa"], color=INK, lw=2, label="燃焼室 Pc"); ax[2].plot(t, r["Pgg_MPa"], color=RED, lw=2, label="GG Pgg"); ax[2].plot(t, r["Pd_o_MPa"], color=LINK, lw=1.5, ls="--", label="酸化剤ポンプ吐出圧")
    ax[2].legend(loc="lower right", fontsize=9, frameon=False, ncol=3); style(ax[2], "圧力 [MPa]")
    mg, mc = r["mdot_gg"] > 1.0, r["mdot_c"] > 30.0
    ax[3].plot(t, np.where(mg, r["OF_gg"] / OFD_GG, np.nan), color=RED, lw=2, label=f"GG（設計 {OFD_GG:.2f}）")
    ax[3].plot(t, np.where(mc, r["OF_mcc"] / OFD_C, np.nan), color=LINK, lw=2, label=f"主燃焼室 MCC（設計 {OFD_C:.2f}）")
    ax[3].set_ylim(0.8, 1.2); ax[3].axhline(1, color=MUTED, ls=":", lw=1); ax[3].legend(loc="lower right", fontsize=9, frameon=False, ncol=2); style(ax[3], "酸燃比 O/F（設計値で割った値）")
    ax[4].plot(t, r["F_kN"], color=GREEN, lw=2.5); ax[4].axhline(845, color=MUTED, ls=":", lw=1); ax[4].text(2.95, 740, "Merlin 1D 海面推力 845 kN", ha="right", fontsize=9, color=MUTED)
    style(ax[4], "推力 [kN]", "時間 [s]"); ax[4].set_ylim(0, 1000)
    save(f, "1dcae2_21_gg_start_v2.png")

    # --- 図: 燃焼温度 vs O/F（モデルに入れた表） ---
    from gg_rocket import T_COMB
    of = np.logspace(np.log10(0.1), np.log10(40), 400)
    f, ax = fig(1, 4.2, sharex=False)
    ax[0].plot(of, T_COMB(of), color=INK, lw=2.2)
    ax[0].axhline(T_LIM, color=RED, ls=":", lw=1.2); ax[0].text(0.55, T_LIM - 170, f"タービン入口温度の上限（目安 {T_LIM:.0f} K）", fontsize=9, color=RED)
    ax[0].axvspan(2.6, 3.0, color=AMBER, alpha=0.25); ax[0].text(2.8, 3750, "量論比付近\n（最高温度）", ha="center", fontsize=9, color=INK)
    for x, y, lab, c, dx, dy in [(OFD_GG, S["gg_base"]["T_gg"], f"GG 設計点\nO/F {OFD_GG:.2f}・{S['gg_base']['T_gg']:.0f} K", RED, 0.22, 1500),
                                 (OFD_C, S["gg_base"]["T_c"], f"主燃焼室 設計点\nO/F {OFD_C:.2f}・{S['gg_base']['T_c']:.0f} K", LINK, 0.55, 2900)]:
        ax[0].plot([x], [y], "o", color=c, ms=9, zorder=5); ax[0].annotate(lab, (x, y), xytext=(dx, dy), fontsize=10, color=c, arrowprops=dict(arrowstyle="->", color=c))
    ax[0].text(20, 2500, "酸素過剰側\n（LOXだけなら燃えない）", ha="center", fontsize=9, color=MUTED); ax[0].text(0.13, 2500, "燃料過剰側", fontsize=9, color=MUTED)
    ax[0].set_xscale("log"); ax[0].set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 40]); ax[0].set_xticklabels(["0.1", "0.2", "0.5", "1", "2", "5", "10", "20", "40"]); ax[0].set_ylim(0, 4100)
    style(ax[0], "LOX/RP-1 の燃焼温度と酸燃比 O/F（モデルに入れた表・目安）", "酸燃比 O/F（対数目盛）", "燃焼温度 [K]")
    save(f, "1dcae2_27_of_temperature.png")

    # --- 図6: 定常の O/F ずれ（GG と MCC の LOX 側オリフィスを ±20%） ---
    scs = np.arange(0.8, 1.201, 0.05)
    def ss_run(**kw):
        rr = simulate(GGEngine(**kw), 2.5, 5e-4, record_every=50)
        return {k: float(rr[k][-1]) for k in ["OF_gg", "OF_mcc", "T_gg_K", "T_c_K", "F_kN", "rpm", "Isp"]}
    G = [ss_run(gg_lox_scale=float(x)) for x in scs]; M = [ss_run(mcc_lox_scale=float(x)) for x in scs]
    S["gg_ofmis"] = dict(scales=scs.tolist(), gg=G, mcc=M)
    f, ax = plt.subplots(2, 2, figsize=(10.24, 7.0), dpi=100); f.patch.set_facecolor(SURF)
    for j, (D, key, tk, name, col) in enumerate([(G, "OF_gg", "T_gg_K", "GG の LOX オリフィス ±20%", RED), (M, "OF_mcc", "T_c_K", "主燃焼室の LOX 噴射器 ±20%", LINK)]):
        x = [d[key] for d in D]
        a0, a1 = ax[0, j], ax[1, j]
        a0.plot(x, [d[tk] for d in D], "o-", color=col, lw=2)
        if j == 0:
            a0.axhline(T_LIM, color=RED, ls=":", lw=1.2); a0.set_ylim(860, 1140); a0.text(x[0], T_LIM + 8, f"上限の目安 {T_LIM:.0f} K", fontsize=9, color=RED)
        style(a0, name, None, "GG 出口温度 [K]" if j == 0 else "燃焼温度 [K]")
        a1.plot(x, [d["F_kN"] for d in D], "o-", color=GREEN, lw=2, label="推力 [kN]")
        b1 = a1.twinx(); b1.plot(x, [d["rpm"] / 1e3 for d in D], "s--", color=INK, lw=1.4, label="回転数 [krpm]"); b1.spines["top"].set_visible(False); b1.tick_params(labelsize=10)
        style(a1, None, "GG の O/F" if j == 0 else "主燃焼室の O/F", "推力 [kN]"); b1.set_ylabel("回転数 [×1000 rpm]", fontsize=11, color=INK)
        a1.set_ylim(700, 1000); b1.set_ylim(32, 42)
        h1, l1 = a1.get_legend_handles_labels(); h2, l2 = b1.get_legend_handles_labels(); a1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9, frameon=False)
        for a in (a0, a1): a.axvline(OFD_GG if j == 0 else OFD_C, color=MUTED, ls=":", lw=1)
    f.suptitle("パラメトリックスタディ③：酸燃比 O/F が設計からずれると", fontsize=13, fontweight="bold", color=INK, x=0.02, ha="left")
    save(f, "1dcae2_22_gg_sweep_of.png")

    # --- 図: GG の LOX 弁 先行／遅れ ---
    lags = np.array([-0.04, -0.03, -0.02, -0.015, -0.01, -0.005, 0.0, 0.01, 0.02, 0.03, 0.04])
    Tpk, rpk, hist = [], [], {}
    for lg in lags:
        rr = simulate(GGEngine(gg_lox_lag=float(lg)), 1.2, 2e-4, record_every=2)
        Tpk.append(float(rr["T_gg_K"].max())); rpk.append(float(rr["rpm"].max()))
        if round(lg * 1e3) in (-20, -10, 0, 20): hist[round(lg * 1e3)] = rr
    S["gg_gglag"] = dict(lags_ms=(lags * 1e3).tolist(), T_peak=Tpk, rpm_peak=rpk)
    f, ax = fig(2, 3.6, sharex=False)
    for (k, rr), c in zip(sorted(hist.items()), [RED, AMBER, INK, LINK]):
        lab = f"LOX {abs(k)} ms 先行" if k < 0 else (f"LOX {k} ms 遅れ" if k > 0 else "同時")
        ax[0].plot(rr["t"], rr["T_gg_K"], color=c, lw=2, label=lab)
    ax[0].axhline(T_LIM, color=RED, ls=":", lw=1.2); ax[0].set_xlim(0.12, 0.40); ax[0].legend(loc="upper right", fontsize=9, frameon=False)
    style(ax[0], "パラメトリックスタディ④：GG の燃料弁と LOX 弁の開くタイミング差 → GG 出口温度", "時間 [s]", "GG 出口温度 [K]")
    ax[1].plot(lags * 1e3, Tpk, "o-", color=RED, lw=2); ax[1].axhline(T_LIM, color=RED, ls=":", lw=1.2); ax[1].text(38, T_LIM + 80, f"上限の目安 {T_LIM:.0f} K", ha="right", fontsize=9, color=RED)
    ax[1].axvspan(-45, 0, color=BLUSH, alpha=0.3); ax[1].text(-43, 2900, "LOX 先行", fontsize=10, color=INK); ax[1].text(43, 2900, "燃料 先行", fontsize=10, color=INK, ha="right")
    style(ax[1], None, "LOX 弁の遅れ [ms]（マイナス ＝ LOX が先に開く）", "GG 出口温度のピーク [K]"); ax[1].set_xlim(-45, 45)
    save(f, "1dcae2_28_gg_lox_lead.png")

    # --- 図: 主弁の LOX 先行／遅れ ---
    mlags = np.arange(-0.15, 0.151, 0.025)
    rpk2, ofmax, ofmin, mh = [], [], [], {}
    for lg in mlags:
        rr = simulate(GGEngine(main_lox_lag=float(lg)), 2.0, 2e-4, record_every=2)
        m = rr["mdot_c"] > 30.0
        rpk2.append(float(rr["rpm"].max())); ofmax.append(float(rr["OF_mcc"][m].max())); ofmin.append(float(rr["OF_mcc"][m].min()))
        if round(lg * 1e3) in (-100, 0, 100): mh[round(lg * 1e3)] = rr
    S["gg_mainlag"] = dict(lags_ms=(mlags * 1e3).tolist(), rpm_peak=rpk2, OF_max=ofmax, OF_min=ofmin)
    f, ax = fig(2, 3.6, sharex=False)
    for (k, rr), c in zip(sorted(mh.items()), [RED, INK, LINK]):
        m = rr["mdot_c"] > 30.0
        lab = f"LOX {abs(k)} ms 先行" if k < 0 else (f"LOX {k} ms 遅れ" if k > 0 else "同時")
        ax[0].plot(rr["t"], np.where(m, np.clip(rr["OF_mcc"], 0.1, 60), np.nan), color=c, lw=2, label=lab)
    ax[0].set_yscale("log"); ax[0].set_ylim(0.1, 80); ax[0].axhspan(2.6, 3.0, color=AMBER, alpha=0.3); ax[0].text(1.95, 3.3, "量論比付近", ha="right", fontsize=9, color=INK)
    ax[0].axhline(OFD_C, color=MUTED, ls=":", lw=1); ax[0].set_xlim(0.25, 2.0); ax[0].legend(loc="upper right", fontsize=9, frameon=False)
    style(ax[0], "パラメトリックスタディ⑤：主弁の燃料弁と LOX 弁のタイミング差 → 主燃焼室の O/F", "時間 [s]", "主燃焼室の O/F（対数）")
    ax[1].plot(mlags * 1e3, np.array(rpk2) / 1e3, "o-", color=INK, lw=2); ax[1].axhline(36 * 1.15, color=RED, ls=":", lw=1); ax[1].text(145, 36 * 1.15 + 0.3, "許容上限の例（設計 +15%）", ha="right", fontsize=9, color=RED)
    ax[1].axvspan(-160, 0, color=BLUSH, alpha=0.3); ax[1].text(-150, 49.2, "LOX 先行：MCC が酸素過剰を通過", fontsize=9, color=INK); ax[1].text(150, 38.7, "LOX 遅れ：LOX ポンプが空回り", ha="right", fontsize=9, color=INK)
    style(ax[1], None, "主 LOX 弁の遅れ [ms]（マイナス ＝ LOX が先に開く）", "回転数ピーク [×1000 rpm]"); ax[1].set_xlim(-160, 160); ax[1].set_ylim(38, 51)
    save(f, "1dcae2_29_main_lox_lead.png")

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
    style(ax[0], "パラメトリックスタディ⑥：主弁の全開時刻 vs ターボポンプ回転数", "主弁が全開になる時刻 [s]（開き始めは 0.30 s 固定）", "回転数 [×1000 rpm]"); ax[0].legend(loc="upper left", fontsize=9, frameon=False)
    save(f, "1dcae2_23_gg_sweep_valve_v2.png")

    # --- 図8: GG分岐量スイープ → 推力 vs Isp ---
    scales = np.arange(0.7, 1.31, 0.1)
    F2, Isp2, gg, ofg = [], [], [], []
    for sc in scales:
        rr = simulate(GGEngine(A_gg_orifice_scale=float(sc)), 2.5, 5e-4, record_every=50)
        F2.append(rr["F_kN"][-1]); Isp2.append(rr["Isp"][-1]); gg.append(100 * rr["mdot_gg"][-1] / (rr["mdot_gg"][-1] + rr["mdot_c"][-1])); ofg.append(rr["OF_gg"][-1])
    S["gg_bleed"] = dict(scales=scales.tolist(), F=F2, Isp=Isp2, gg_pct=gg, OF_gg=ofg)
    f, ax = fig(1, 4.4, sharex=False)
    ax[0].plot(gg, F2, "o-", color=GREEN, lw=2, label="推力 [kN]")
    style(ax[0], "パラメトリックスタディ⑦：GG分岐流量 vs 推力・比推力（GG の O/F は一定）", "GG に分岐する推進剤の割合 [%]", "推力 [kN]")
    ax2 = ax[0].twinx(); ax2.plot(gg, Isp2, "s--", color=INK, lw=1.5, label="比推力 Isp [s]"); ax2.set_ylabel("比推力 [s]"); ax2.spines["top"].set_visible(False)
    h1, l1 = ax[0].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax[0].legend(h1 + h2, l1 + l2, loc="center left", fontsize=9, frameon=False)
    save(f, "1dcae2_24_gg_sweep_bleed_v2.png")

    # --- 図9: オイラー法 / 2段・4段ルンゲ＝クッタ法 ---
    ref = simulate(GGEngine(), 1.5, 2e-5, method="rk4", record_every=1)
    dts = [2e-4, 5e-4, 1e-3, 1.5e-3, 1.8e-3, 2e-3, 2.5e-3, 3e-3]
    err = {m: [] for m in ("euler", "rk2", "rk4")}
    for m in err:
        for dt in dts:
            try:
                rr = simulate(GGEngine(), 1.5, dt, method=m, record_every=1)
                e = float(np.max(np.abs(rr["Pc_MPa"] - np.interp(rr["t"], ref["t"], ref["Pc_MPa"]))))
            except FloatingPointError:
                e = np.inf
            err[m].append(e)
    S["gg_int"] = dict(dts_ms=[d * 1e3 for d in dts], err=err)
    f, ax = fig(2, 3.8, sharex=False)
    names = {"euler": "オイラー法", "rk2": "2段ルンゲ＝クッタ法", "rk4": "4段ルンゲ＝クッタ法"}
    cols = {"euler": RED, "rk2": AMBER, "rk4": INK}
    for m in ("rk4", "rk2", "euler"):
        try:
            rr = simulate(GGEngine(), 1.4, 2e-3, method=m, record_every=1)
            bad2 = not np.isfinite(err[m][dts.index(2e-3)]) or err[m][dts.index(2e-3)] > 1.0
            ax[0].plot(rr["t"], rr["Pc_MPa"], color=cols[m], lw=1.8 if m == "rk4" else 1.3, label=names[m] + ("（dt = 2 ms、発散）" if bad2 else "（dt = 2 ms）"), zorder=3 if m == "rk4" else 2)
        except FloatingPointError:
            ax[0].plot([], [], color=cols[m], label=names[m] + "（dt = 2 ms、発散）")
    ax[0].set_ylim(0, 16); ax[0].legend(loc="upper left", fontsize=9, frameon=False)
    style(ax[0], "同じモデル・同じ dt = 2 ms でも積分法で結果が変わる（燃焼室圧）", "時間 [s]", "Pc [MPa]")
    for m, yx in (("euler", 0.6), ("rk2", 0.3), ("rk4", 0.15)):
        e = np.array(err[m]); ok = np.isfinite(e) & (e < 1.0)
        ax[1].plot(np.array(dts)[ok] * 1e3, np.maximum(e[ok], 1e-5), "o-", color=cols[m], lw=2, label=names[m])
        bad = ~ok
        if bad.any():
            ax[1].plot(np.array(dts)[bad] * 1e3, np.full(bad.sum(), yx), "x", color=cols[m], ms=10, mew=2.5)
    ax[1].set_yscale("log"); ax[1].set_ylim(1e-5, 1.0); ax[1].text(0.2, 0.3, "× = 発散（誤差 1 MPa 以上）", ha="left", va="center", fontsize=9, color=INK)
    ax[1].legend(loc="lower right", fontsize=9, frameon=False)
    style(ax[1], None, "時間刻み dt [ms]", "Pc の最大誤差 [MPa]")
    save(f, "1dcae2_25_gg_integrators.png")

    # --- 図10: タンク枯渇までの長時間運転 ---
    rr = simulate(GGEngine(m_f0=1200.0, m_o0=2800.0), 20.0, 5e-4, record_every=40)
    S["gg_deplete"] = dict(t_end=float(rr["t"][np.argmax(rr["F_kN"] < 10) if (rr["F_kN"] < 10).any() else -1]))
    f, ax = fig(2, 3.0)
    ax[0].plot(rr["t"], rr["m_f"], color=AMBER, lw=2, label="燃料 [kg]"); ax[0].plot(rr["t"], rr["m_o"], color=LINK, lw=2, label="酸化剤 [kg]"); ax[0].legend(fontsize=9, frameon=False); style(ax[0], "タンク残量")
    ax[1].plot(rr["t"], rr["F_kN"], color=GREEN, lw=2); style(ax[1], "推力 [kN]", "時間 [s]")
    t_off = float(rr["t"][np.argmax((rr["F_kN"] < 10) & (rr["t"] > 1))]); S["gg_deplete"] = dict(t_off=t_off, m_o_left=float(rr["m_o"][-1]))
    ax[0].annotate(f"燃料が先に枯渇 → 酸化剤 {rr['m_o'][-1]:.0f} kg が残る（残推進剤）", (t_off, rr["m_o"][-1]), xytext=(11.5, 1400), fontsize=10, color=INK, ha="center", arrowprops=dict(arrowstyle="->", color=INK))
    ax[1].annotate(f"燃焼停止 t = {t_off:.1f} s", (t_off, 400), xytext=(15.0, 500), fontsize=10, color=INK, arrowprops=dict(arrowstyle="->", color=INK))
    save(f, "1dcae2_26_gg_depletion_v2.png")


if __name__ == "__main__":
    if "rocket" not in sys.argv:
        study_pumped()
    if "quick" not in sys.argv:
        study_rocket()
    old = json.load(open(f"{OUT}/summary.json")) if os.path.exists(f"{OUT}/summary.json") else {}
    old.update(S); S = old
    json.dump(S, open(f"{OUT}/summary.json", "w"), indent=2, ensure_ascii=False, default=float)
    print(json.dumps({k: v for k, v in S.items() if k.endswith("base")}, indent=1, ensure_ascii=False, default=float))
