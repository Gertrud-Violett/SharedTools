"""
run_studies.py — 記事の全スタディを実行し、図（./fig/*.png）と数値（./fig/summary.json）を出力
python run_studies.py            # 全部（10〜15分）
python run_studies.py pumped     # 揚水発電所だけ（2分）
python run_studies.py rocket     # ロケットエンジンだけ
"""
import json, math, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib import font_manager as _fm
from onedsim import simulate, bisect
from pumped_storage import PumpedStorage, make_demand, make_rain, roundtrip, DAY
from gg_rocket import GGEngine, design_point, T_AD, PUMP_HEAD, PUMP_POWER, HH_A1

# ----------------------------------------------------------------------------- 図の体裁（まきブログの図版スタイル）
_have = {f.name for f in _fm.fontManager.ttflist}      # 日本語フォントを OS ごとに自動選択
plt.rcParams["font.family"] = next((f for f in ["Noto Sans CJK JP", "Yu Gothic", "Meiryo", "Hiragino Sans", "MS Gothic", "IPAexGothic"] if f in _have), "sans-serif")
plt.rcParams["axes.unicode_minus"] = False
INK, MUTED, LINK, PINK, SKY, GREEN, STAR, GRID = "#232a42", "#888888", "#008db7", "#e896c7", "#89c2f4", "#73bc9b", "#f5b301", "#fdcdcd"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig")
os.makedirs(OUT, exist_ok=True)
S = {}                                                  # 記事に書く数値はすべてここに入れて summary.json に保存


def chart(title, xlabel, ylabel, note=None, h=6.0, top=0.78):
    """1枚1テーマの図。左と下だけの軸、薄いピンクの目盛線、太字のタイトル、下に注記。"""
    f, ax = plt.subplots(figsize=(10.24, h), dpi=100)
    f.patch.set_facecolor("white")
    f.subplots_adjust(left=0.125, right=0.955, top=1 - top / h, bottom=(1.55 if note else 1.05) / h)
    ax.set_facecolor("white")
    ax.grid(color=GRID, lw=0.9)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK)
        ax.spines[sp].set_linewidth(1.6)
    ax.plot(1, 0, ">", color=INK, ms=8, transform=ax.transAxes, clip_on=False)
    ax.plot(0, 1, "^", color=INK, ms=8, transform=ax.transAxes, clip_on=False)
    ax.tick_params(colors=INK, labelsize=15, width=1.2, length=5)
    ax.set_title(title, fontsize=19, fontweight="bold", color=INK, pad=16)
    ax.set_xlabel(xlabel, fontsize=17, color=INK, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=17, color=INK, labelpad=8)
    if note:
        f.text(0.03, 0.035, note, fontsize=12.5, color=MUTED, ha="left", va="bottom", linespacing=1.5)
    return f, ax


def commas(ax, axis="y"):
    """目盛りの数字に 3 桁区切りを入れる"""
    getattr(ax, axis + "axis").set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))


def legend(ax, **kw):
    kw.setdefault("fontsize", 15)
    return ax.legend(frameon=False, labelcolor=INK, handlelength=2.2, **kw)


def label(ax, x, y, text, color=INK, size=15, **kw):
    ax.text(x, y, text, color=color, fontsize=size, **kw)


def save(f, name):
    f.savefig(os.path.join(OUT, name), facecolor="white")
    plt.close(f)
    print("saved", name)


# =============================================================================== 揚水発電所
def study_pumped():
    HOT_MW = 940.0                                       # 猛暑日の要求 = 2台の定格出力（94万kW）
    t0 = time.time()
    m = PumpedStorage(rain=make_rain(storm_day=5), demand=make_demand(heatwave_day=2, heatwave_MW=HOT_MW))
    r = simulate(m, 7 * DAY, 60.0, record_every=5)
    run_s = time.time() - t0
    d = r["t"] / DAY
    gen = r["P_gen"] > 1
    two = r["P_pump"] > 1.5 * m.P_pump_nominal            # 2台で揚水している時間帯（平日の夜）
    hot = gen & (d >= 2) & (d < 3)
    S["ps_base"] = dict(E_gen=r["E_gen"][-1], E_pump=r["E_pump"][-1], E_spill=r["E_spill"][-1], E_short=r["E_short"][-1],
                        roundtrip=roundtrip(r), level_min=r["level_u"].min(), level_max=r["level_u"].max(),
                        level_full=m.V_max / m.A, level_low=m.V_min / m.A,
                        H_min=r["H"].min(), H_max=r["H"].max(), Qp_max=r["Q_p"].max(), Qp_min=r["Q_p"][r["Q_p"] > 1].min(), Qt_max=r["Q_t"].max(),
                        P_pump_max=r["P_pump"][two].max(), P_pump_min=r["P_pump"][two].min(), P_pump_1unit_max=r["P_pump"][(r["P_pump"] > 1) & ~two].max(),
                        P_gen_max=r["P_gen"].max(), P_gen_hot_min=r["P_gen"][hot].min(), P_gen_hot_max=r["P_gen"][hot].max(),
                        Q_rain_max=r["Q_rain"].max(), run_s=run_s, hot_MW=HOT_MW, hot_hours=float(hot.sum() * 300 / 3600),
                        H_hot_min=r["H"][hot].min(), catch_km2=m.catch_up / 1e6, P_unit=m.P_unit, n_units=m.n)
    b = S["ps_base"]
    at_req = hot & (r["P_req"] > HOT_MW - 1)                # 猛暑日に定格いっぱいを要求している時間帯
    b["P_gen_hotreq_min"], b["H_hotreq_min"] = r["P_gen"][at_req].min(), r["H"][at_req].min()
    b["Heff_hotreq_min"] = b["H_hotreq_min"] - m.k_loss * m.Q_turb_max ** 2          # そのときの有効落差（流量が上限のとき）
    b["level_hot_start"] = float(r["level_u"][np.argmax(hot)])                       # 猛暑日の発電を始めるときの上池水位
    b["surplus_MW"], b["surplus_weekend_MW"], b["pump_nominal_MW"], b["runoff"] = 900.0, 630.0, m.P_pump_nominal, m.runoff
    b["level_thu_min"], b["level_fri_min"] = float(r["level_u"][(d >= 3) & (d < 4)].min()), float(r["level_u"][(d >= 4) & (d < 5)].min())
    b["level_end"], b["level_start"] = float(r["level_u"][-1]), float(r["level_u"][0])
    b["Q_op_Hmin"], b["Q_op_Hmax"], b["Q_op_Hbase"] = (m.pump_operating_point(float(round(b["H_min"]))), m.pump_operating_point(float(round(b["H_max"]))),
                                                     m.pump_operating_point(m.H_base))      # 図と本文に書く動作点（落差は m 単位に丸めた値で計算）
    shade = lambda ax: (ax.axvspan(2, 3, color=STAR, alpha=0.13, lw=0), ax.axvspan(5, 7, color="#dcdfe6", alpha=0.45, lw=0))
    days = ["月", "火", "水", "木", "金", "土", "日"]

    # --- 図: 1週間の電力（発電 / 揚水）
    f, ax = chart("1週間の運転：発電と揚水の電力", "曜日", "電力 [MW]",
                  "プラス側が発電、マイナス側が揚水。黄色は猛暑日（水曜）、灰色は週末（揚水は1台）。点線は発電の要求値")
    shade(ax)
    ax.fill_between(d, 0, r["P_gen"], color=LINK, alpha=0.9, lw=0, label="発電出力")
    ax.fill_between(d, 0, -r["P_pump"], color=PINK, alpha=0.9, lw=0, label="揚水入力")
    ax.plot(d, np.where(r["P_req"] > 0, r["P_req"], np.nan), color=INK, lw=1.4, ls=":")
    ax.axhline(0, color=INK, lw=1.0)
    ax.set_xlim(0, 7); ax.set_ylim(-1250, 1450)
    ax.set_xticks(np.arange(7) + 0.5); ax.set_xticklabels(days); ax.tick_params(axis="x", length=0)
    ax.set_xticks(np.arange(8), minor=True); ax.grid(False, axis="x"); ax.grid(True, axis="x", which="minor", color=GRID, lw=0.9)
    legend(ax, loc="upper right")
    label(ax, 2.5, 1210, f"猛暑日\n要求 {HOT_MW:.0f} MW", ha="center", va="center", size=14)
    commas(ax)
    save(f, "1dcae2_04_ps_week_power_v3.png")

    # --- 図: 1週間の上池水位
    f, ax = chart("1週間の運転：上池の水位", "曜日", "上池の水位 [m]",
                  "黄色は猛暑日（水曜）、灰色は週末。土曜の朝6時から12時間、30 mm/h の豪雨を入れている")
    shade(ax)
    ax.plot(d, r["level_u"], color=LINK, lw=2.6)
    ax.axhline(b["level_full"], color=PINK, ls="--", lw=1.8); ax.axhline(b["level_low"], color=PINK, ls="--", lw=1.8)
    label(ax, 0.1, b["level_full"] - 0.6, f"満水 {b['level_full']:.1f} m", color="#c0569a", va="top")
    label(ax, 6.9, b["level_low"] + 0.5, f"下限 {b['level_low']:.1f} m", color="#c0569a", va="bottom", ha="right")
    i = int(np.argmin(r["level_u"]))
    ax.annotate(f"猛暑日の夜に {b['level_min']:.1f} m まで低下", (d[i], r["level_u"][i]), xytext=(2.72, 3.4), fontsize=15, color=INK, ha="right", va="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.3, shrinkB=5))
    ax.axvspan(5.25, 5.75, color=SKY, alpha=0.45, lw=0); label(ax, 5.5, 29.5, "豪雨", ha="center", va="center", size=14)
    ax.set_xlim(0, 7); ax.set_ylim(0, 34)
    ax.set_xticks(np.arange(7) + 0.5); ax.set_xticklabels(days); ax.tick_params(axis="x", length=0)
    ax.set_xticks(np.arange(8), minor=True); ax.grid(False, axis="x"); ax.grid(True, axis="x", which="minor", color=GRID, lw=0.9)
    save(f, "1dcae2_05_ps_week_level_v3.png")

    # --- 図: ポンプの動作点（ポンプ曲線と系の要求曲線の交点）
    f, ax = chart("ポンプの動作点：2本の曲線の交点", "流量 [m³/s]（1台あたり）", "揚程 [m]",
                  "ポンプ曲線は固定。システムの要求（＝落差＋水路の損失）は、上池と下池の水位で上下する")
    Q = np.linspace(0, 80, 200)
    ax.plot(Q, m.pump_H(Q), color=INK, lw=3, label="ポンプのQ-H曲線（モデルに入れた表）")
    for H, c, tx, ha in [(float(round(b["H_max"])), PINK, (30.0, 668.0), "center"), (float(round(b["H_min"])), LINK, (58.0, 632.0), "center")]:
        ax.plot(Q, H + m.k_loss * Q ** 2, color=c, lw=2.6, ls="--", label=f"システムの要求（落差 {H:.0f} m）")
        Qop = m.pump_operating_point(H)
        ax.plot([Qop], [m.pump_H(Qop)], "o", color=c, ms=12, mec=INK, mew=1.5, zorder=5)
        ax.annotate(f"動作点 {Qop:.1f} m³/s", (Qop, m.pump_H(Qop)), xytext=tx, fontsize=15, color=INK, ha=ha,
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.3, shrinkB=8))
    ax.set_xlim(0, 80); ax.set_ylim(580, 800)
    legend(ax, loc="upper right")
    save(f, "1dcae2_03_ps_operating_point_v3.png")

    # --- 図: 豪雨の強さ → 越流で失う電力量（猛暑日のない通常の週。揚水を予定どおり続ける場合と、雨のあいだは止める場合）
    rains = np.arange(0, 101, 5)
    REC_MM = 922.5                                       # 日降水量の国内記録（気象庁：箱根、2019年10月12日）
    def spill_at(mm, rule):
        rr = simulate(PumpedStorage(rain=make_rain(storm_day=5, mm_per_h=float(mm), hours=24.0), pump_in_rain=rule), 7 * DAY, 60.0, record_every=20)
        return float(rr["E_spill"][-1])
    sp = {rule: [spill_at(mm, rule) for mm in rains] for rule in (True, False)}
    ok = {rule: int(rains[np.argmax(np.array(sp[rule]) > 1.0)]) - 5 for rule in sp}      # 5 mm/h 刻みで見た、あふれない最大の強さ
    thr = {rule: bisect(lambda mm: spill_at(mm, rule) - 1.0, float(ok[rule]), ok[rule] + 5.0, 8) for rule in sp}   # あふれ始める強さ（二分法で細かく探す）
    S["ps_rain"] = dict(rains=rains.tolist(), spill=sp[True], spill_hold=sp[False], ok=ok[True], ok_hold=ok[False], thr=thr[True], thr_hold=thr[False],
                        record_mm_day=REC_MM, record_mm_h=REC_MM / 24)
    f, ax = chart("豪雨の強さと、越流で失う電力量", "降雨の強さ [mm/h]（土曜の朝から24時間降り続く）", "失った発電の機会 [GWh]",
                  "猛暑日のない通常の週に、土曜の朝6時から24時間の雨を入れた計算。上池が満水になると、それ以上の雨は越流として\n捨てるしかない。捨てた水で発電できたはずの電力量を縦軸にとった")
    top = max(sp[True]) / 1e3 * 1.16
    ax.plot(rains, np.array(sp[True]) / 1e3, "o-", color=PINK, lw=2.8, ms=8, label="予定どおり夜に揚水する")
    ax.plot(rains, np.array(sp[False]) / 1e3, "s-", color=LINK, lw=2.8, ms=8, label="雨のあいだは揚水しない")
    ax.axvline(REC_MM / 24, color=INK, ls=":", lw=1.4)
    label(ax, REC_MM / 24 - 1.5, top * 0.62, "日降水量の国内記録\n（箱根 922.5 mm）\nに相当", ha="right", va="center", size=14, color=MUTED)
    ax.annotate(f"約 {thr[True]:.0f} mm/h までは\nあふれない", (thr[True], 0), xytext=(40.8, top * 0.31), fontsize=15, color=INK, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.3, shrinkB=4))
    ax.annotate(f"揚水を止めれば\n約 {thr[False]:.0f} mm/h まで", (thr[False], 0), xytext=(74.0, top * 0.075), fontsize=15, color=INK, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.3, shrinkB=4))
    ax.set_xlim(-4, 104); ax.set_ylim(-top * 0.04, top); legend(ax, loc="upper left")
    save(f, "1dcae2_06_ps_rain_v3.png")

    # --- 図: ピーク需要 → 供給不足 / 上池の最低水位
    peaks = np.arange(600, 1201, 10)
    short, lvmin = [], []
    for pk in peaks:
        rr = simulate(PumpedStorage(rain=make_rain(storm_day=5), demand=make_demand(peak_MW=float(pk), heatwave_day=2)), 7 * DAY, 60.0, record_every=20)
        short.append(rr["E_short"][-1]); lvmin.append(rr["level_u"].min())
    S["ps_peak"] = dict(peaks=peaks.tolist(), short=short, lvmin=lvmin)
    f, ax = chart("ピーク時の発電要求と供給不足（2台運転）", "ピーク時の発電要求 [MW]", "供給不足 [GWh/週]",
                  "水曜の猛暑日は11時間、ほかの平日は昼と夕方に3時間ずつ、この値を要求する（週末はその5〜6割）。\n雨は基準の週と同じ。設備の出せる出力を超えた分を、1週間の供給不足として積算した")
    ax.bar(peaks, np.array(short) / 1e3, width=7.0, color=LINK)
    ok = int(peaks[np.argmax(np.array(short) > 1.0)]) - 10
    top = max(short) / 1e3 * 1.12
    label(ax, ok - 60, top * 0.30, f"{ok:,} MW までは\n不足なし", ha="center", va="center")
    ax.annotate("", (ok, top * 0.02), xytext=(ok - 30, top * 0.20), arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    ax.axvline(HOT_MW, color=INK, ls=":", lw=1.4); label(ax, HOT_MW + 8, top * 0.93, f"2台の定格 {HOT_MW:.0f} MW", ha="left", va="center", color=MUTED, size=14)
    ax.set_xlim(570, 1230); ax.set_ylim(0, top); commas(ax, "x")
    save(f, "1dcae2_07_ps_demand_short_v3.png")

    f, ax = chart("ピーク時の発電要求と上池の最低水位（2台運転）", "ピーク時の発電要求 [MW]", "上池の最低水位 [m]",
                  "要求を上げても、ある点から水位は下がらなくなる。\n設備の上限（出力・流量）で運転していて、それ以上は水を使えないため")
    ax.plot(peaks, lvmin, "o-", color=LINK, lw=2.8, ms=5)
    ax.axhline(b["level_low"], color=PINK, ls="--", lw=1.8); label(ax, 1222, b["level_low"] + 0.4, f"下限 {b['level_low']:.1f} m", color="#c0569a", ha="right", va="bottom")
    label(ax, 1090, lvmin[-1] + 1.3, f"{lvmin[-1]:.1f} m で下げ止まり", ha="center", va="bottom")
    ax.set_xlim(570, 1230); ax.set_ylim(0, 18); commas(ax, "x")
    save(f, "1dcae2_08_ps_demand_level_v3.png")


# =============================================================================== ロケットエンジン
def last(r, keys):
    return {k: float(r[k][-1]) for k in keys}


def study_rocket():
    KEYS = ["F_kN", "Pc_MPa", "Pgg_MPa", "rpm", "Isp", "mdot_c", "mdot_gg", "gg_pct", "OF_mcc", "OF_gg", "OF_engine", "T_c_K", "T_gg_K", "W_t_MW", "W_p_MW", "Pd_o_MPa", "Pd_f_MPa"]
    d = design_point()
    S["gg_design"] = {k: float(v) for k, v in d.items()}
    T_LIM = 1200.0          # タービン入口温度の実用上の目安（約 1,700 °F。Huzel & Huang）
    RPM_D = d["rpm"]

    # ----- 基準ケース（起動 3 秒）
    t0 = time.time()
    r = simulate(GGEngine(), 3.0, 2e-4, record_every=5)
    t = r["t"]
    S["gg_base"] = last(r, KEYS)
    ip = int(r["rpm"].argmax())
    S["gg_base"].update(run_s=time.time() - t0, rpm_peak=float(r["rpm"][ip]), t_rpm_peak=float(t[ip]), Pd_peak=float(r["Pd_o_MPa"].max()),
                        Pgg_peak=float(r["Pgg_MPa"].max()), Pc_peak=float(r["Pc_MPa"].max()), F_peak=float(r["F_kN"].max()),
                        t_90F=float(t[np.argmax(r["F_kN"] > 0.9 * r["F_kN"][-1])]), W_t_peak=float(r["W_t_MW"].max()), W_p_peak=float(r["W_p_MW"].max()),
                        OF_mcc_start_max=float(r["OF_mcc"][r["mdot_c"] > 30].max()), T_gg_start_max=float(r["T_gg_K"].max()),
                        OF_mcc_start_min=float(r["OF_mcc"][r["mdot_c"] > 30].min()),
                        t_settle=float(t[np.where(np.abs(r["F_kN"] - r["F_kN"][-1]) > 0.01 * r["F_kN"][-1])[0][-1]]))   # 以後、推力が定常値の ±1 % に収まる時刻
    b = S["gg_base"]
    seq = "スピンスタート 0〜0.5 s、GG弁 0.15〜0.25 s、主弁 0.3〜0.9 s（燃料弁とLOX弁は同時に開く）"
    TMAX = 1.5

    def seq_marks(ax, y):
        ax.axvspan(0.15, 0.25, color=STAR, alpha=0.22, lw=0); ax.axvspan(0.30, 0.90, color=SKY, alpha=0.22, lw=0)
        label(ax, 0.20, y, "GG弁\n開", ha="center", va="top", size=14); label(ax, 0.60, y, "主弁を開いていく", ha="center", va="top", size=14)

    f, ax = chart("起動過渡：ターボポンプの回転数", "時間 [s]", "回転数 [×1,000 rpm]", seq)
    seq_marks(ax, 51.5)
    ax.plot(t, r["rpm"] / 1e3, color=LINK, lw=2.8)
    ax.axhline(RPM_D / 1e3, color=INK, ls=":", lw=1.4); label(ax, 1.48, RPM_D / 1e3 - 1.3, f"設計回転数 {RPM_D:,.0f} rpm", ha="right", va="top", color=MUTED, size=14)
    ax.annotate(f"ピーク {b['rpm_peak']:,.0f} rpm\n（設計の {b['rpm_peak'] / RPM_D * 100 - 100:.0f} % 増し）", (b["t_rpm_peak"], b["rpm_peak"] / 1e3), xytext=(1.13, 47.0), fontsize=15,
                color=INK, va="center", ha="center", arrowprops=dict(arrowstyle="->", color=INK, lw=1.3, shrinkB=4))
    ax.set_xlim(0, TMAX); ax.set_ylim(0, 53)
    save(f, "1dcae2_23_gg_start_rpm_v3.png")

    f, ax = chart("起動過渡：タービンの出力とポンプの動力", "時間 [s]", "動力 [MW]", "タービンの出力がポンプの動力を上回っているあいだ（水色の部分）、軸は加速する。\n" + seq)
    seq_marks(ax, 9.75)
    ax.plot(t, r["W_t_MW"], color=LINK, lw=2.8, label="タービンの出力")
    ax.plot(t, r["W_p_MW"], color=PINK, lw=2.8, ls="--", label="ポンプの動力（2台合計）")
    ax.fill_between(t, r["W_t_MW"], r["W_p_MW"], where=r["W_t_MW"] > r["W_p_MW"], color=LINK, alpha=0.15, lw=0)
    ax.set_xlim(0, TMAX); ax.set_ylim(0, 10); legend(ax, loc="upper right")
    save(f, "1dcae2_24_gg_start_power_v3.png")

    f, ax = chart("起動過渡：各部の圧力", "時間 [s]", "圧力 [MPa]", seq)
    seq_marks(ax, 27.3)
    ax.plot(t, r["Pd_o_MPa"], color=INK, lw=2.4, ls="-.", label="ポンプ吐出圧（LOX側）")
    ax.plot(t, r["Pgg_MPa"], color=PINK, lw=2.8, ls="--", label="ガスジェネレータ")
    ax.plot(t, r["Pc_MPa"], color=LINK, lw=2.8, label="主燃焼室")
    ax.annotate(f"吐出圧のピーク {b['Pd_peak']:.1f} MPa", (float(t[r['Pd_o_MPa'].argmax()]), b["Pd_peak"]), xytext=(0.66, 24.7), fontsize=15, color=INK, va="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.3, shrinkB=6))
    ax.set_xlim(0, TMAX); ax.set_ylim(0, 28); legend(ax, loc="center right", bbox_to_anchor=(1.0, 0.62))
    save(f, "1dcae2_25_gg_start_pressure_v3.png")

    f, ax = chart("起動過渡：推力", "時間 [s]", "推力 [kN]", seq)
    seq_marks(ax, 975)
    ax.plot(t, r["F_kN"], color=LINK, lw=2.8)
    ax.axhline(845, color=INK, ls=":", lw=1.4); label(ax, 1.48, 815, "Merlin 1D の海面推力 845 kN", ha="right", va="top", color=MUTED, size=14)
    ax.set_xlim(0, TMAX); ax.set_ylim(0, 1000); commas(ax)
    save(f, "1dcae2_26_gg_start_thrust_v3.png")

    # ----- 部品の特性の図: 燃焼温度の表、ポンプマップ
    of = np.logspace(np.log10(0.1), np.log10(60), 500)
    f, ax = chart("LOX/ケロシンの燃焼温度と酸燃比（モデルに入れた表）", "酸燃比 O/F（対数目盛）", "燃焼温度 [K]",
                  "O/F 0.8以上は化学平衡計算（RocketCEA、9.7 MPa）、0.30〜0.52は教科書の代表値（Huzel & Huang, NASA SP-125）。\nそのあいだと0.30より下は、点をつないで補った値。主燃焼室はこの温度に燃焼効率の2乗（約0.95）を掛けて使う")
    ax.plot(of, T_AD(of), color=LINK, lw=2.8)
    ax.axhline(T_LIM, color=PINK, ls="--", lw=1.8); label(ax, 4.0, T_LIM - 70, f"タービン入口温度の目安 {T_LIM:,.0f} K", color="#c0569a", ha="center", va="top", size=14)
    ax.plot([d["OF_g"]], [d["T_gg"]], "o", color=STAR, ms=12, mec=INK, mew=1.5, zorder=5)
    ax.annotate(f"GG の設計点\nO/F {d['OF_g']:.2f}・{d['T_gg']:.0f} K", (d["OF_g"], d["T_gg"]), xytext=(0.135, 2250), fontsize=15, color=INK, va="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.3, shrinkB=7))
    ax.plot([d["OF_c"]], [float(T_AD(d["OF_c"]))], "o", color=STAR, ms=12, mec=INK, mew=1.5, zorder=5)
    ax.annotate(f"主燃焼室の設計点\nO/F {d['OF_c']:.2f}・{float(T_AD(d['OF_c'])):,.0f} K", (d["OF_c"], float(T_AD(d["OF_c"]))), xytext=(0.62, 3250), fontsize=15, color=INK, va="center", ha="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.3, shrinkB=7))
    label(ax, 24, 3550, "酸素過剰側\nLOXが多いほど冷える", ha="center", va="center", color=MUTED, size=14)
    ax.set_xscale("log"); ax.set_xlim(0.1, 60); ax.set_ylim(0, 4200); commas(ax)
    ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]); ax.set_xticklabels(["0.1", "0.2", "0.5", "1", "2", "5", "10", "20", "50"]); ax.minorticks_off()
    save(f, "1dcae2_21_gg_gas_table_v3.png")

    qn = np.linspace(0, 1.75, 200)
    f, ax = chart("ポンプの特性マップ（設計点を1とした比）", "流量（設計点 = 1）", "揚程・軸動力（設計点 = 1）",
                  "回転数が変わっても、流量を回転数で、揚程を回転数の2乗で、動力を3乗で割ればこの1組の曲線に乗る（相似則）。\n曲線の形は今回の仮定で、実際は単体試験で決める")
    ax.plot(qn, PUMP_HEAD(qn), color=LINK, lw=2.8, label="揚程")
    ax.plot(qn, PUMP_POWER(qn), color=PINK, lw=2.8, ls="--", label="軸動力")
    ax.plot([1], [1], "o", color=STAR, ms=12, mec=INK, mew=1.5, zorder=5); label(ax, 1.04, 1.07, "設計点", ha="left", va="bottom")
    ax.annotate("締切り（流量ゼロ）でも\n設計点の45 %の動力を食う", (0, 0.45), xytext=(0.30, 0.25), fontsize=15, color=INK, va="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.3, shrinkB=4))
    ax.set_xlim(0, 1.8); ax.set_ylim(0, 1.45); legend(ax, loc="upper center", ncol=2)
    save(f, "1dcae2_22_gg_pump_map_v3.png")

    # ----- 定常の O/F ずれ: GG の LOX オリフィス / 主燃焼室の LOX 噴射器 を ±20 %
    scs = np.round(np.arange(0.80, 1.201, 0.05), 2)
    ss_run = lambda **kw: last(simulate(GGEngine(**kw), 2.5, 5e-4, record_every=50), KEYS)
    G = [ss_run(gg_lox_scale=float(x)) for x in scs]
    M = [ss_run(mcc_lox_scale=float(x)) for x in scs]
    S["gg_ofmis"] = dict(scales=scs.tolist(), gg=G, mcc=M)
    pct = (scs - 1) * 100
    f, ax = chart("LOX側の流れやすさが ±20 % ずれたときの推力", "LOXオリフィス（噴射器）の流れやすさのずれ [%]", "推力 [kN]",
                  "流れやすさ＝流量係数×面積（CdA）。ずれの原因は製作公差、詰まり、部品の付け間違いなど。値は起動後の定常値")
    ax.plot(pct, [g["F_kN"] for g in G], "o-", color=LINK, lw=2.8, ms=9, label="GG の LOX オリフィスがずれた場合")
    ax.plot(pct, [g["F_kN"] for g in M], "s--", color=PINK, lw=2.8, ms=9, label="主燃焼室の LOX 噴射器がずれた場合")
    ax.axhline(845, color=INK, ls=":", lw=1.4)
    ax.set_xlim(-22, 22); ax.set_ylim(650, 1100); legend(ax, loc="upper left"); commas(ax)
    save(f, "1dcae2_27_gg_of_thrust_v3.png")

    f, ax = chart("GGのLOXオリフィスが ±20 % ずれたときのGG出口温度", "GG の LOX オリフィスの流れやすさのずれ [%]", "GG出口（タービン入口）温度 [K]",
                  "下の軸はオリフィスのずれ、上の軸はそのときの GG の酸燃比。値は起動後の定常値", h=6.5, top=1.3)
    ax.plot(pct, [g["T_gg_K"] for g in G], "o-", color=LINK, lw=2.8, ms=9)
    ax.axhline(T_LIM, color=PINK, ls="--", lw=1.8); label(ax, -21, T_LIM - 12, f"タービン入口温度の目安 {T_LIM:,.0f} K", color="#c0569a", va="top", size=14)
    ax.axhline(d["T_gg"], color=INK, ls=":", lw=1.4); label(ax, 21, d["T_gg"] - 12, f"設計 {d['T_gg']:.0f} K", color=MUTED, ha="right", va="top", size=14)
    ax2 = ax.twiny(); ax2.set_xlim(-22, 22); ax2.set_xticks(pct[::2]); ax2.set_xticklabels([f"{g['OF_gg']:.2f}" for g in G[::2]])
    ax2.tick_params(colors=MUTED, labelsize=14, length=0); [sp.set_visible(False) for sp in ax2.spines.values()]
    ax2.set_xlabel("そのときの GG の酸燃比 O/F", fontsize=14, color=MUTED, labelpad=6)
    ax.set_title(ax.get_title(), fontsize=19, fontweight="bold", color=INK, pad=62)
    ax.set_xlim(-22, 22); ax.set_ylim(750, 1260); commas(ax)
    save(f, "1dcae2_28_gg_of_temp_v3.png")

    # ----- GG の LOX 弁 先行／遅れ
    lags = np.array([-0.04, -0.03, -0.02, -0.015, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.03, 0.04])
    Tpk, rpk, hist = [], [], {}
    for lg in lags:
        rr = simulate(GGEngine(gg_lox_lag=float(lg)), 1.2, 1e-4, record_every=2)
        Tpk.append(float(rr["T_gg_K"].max())); rpk.append(float(rr["rpm"].max()))
        if round(lg * 1e3) in (-20, -10, 0, 20):
            hist[round(lg * 1e3)] = rr
    S["gg_gglag"] = dict(lags_ms=(lags * 1e3).tolist(), T_peak=Tpk, rpm_peak=rpk)
    f, ax = chart("GGの弁を開く順番と、GG出口温度の時間変化", "時間 [s]", "GG出口温度 [K]",
                  "GGの燃料弁は 0.15 s から 0.1 s かけて開く。LOX弁をそれより早く／遅く開けた場合の比較")
    for k, c, ls, lab in [(-20, PINK, "-", "LOX弁が 20 ms 先"), (-10, STAR, "-", "LOX弁が 10 ms 先"), (0, LINK, "-", "同時"), (20, INK, "--", "LOX弁が 20 ms 後（燃料が先）")]:
        ax.plot(hist[k]["t"], hist[k]["T_gg_K"], color=c, lw=2.8, ls=ls, label=lab)
    ax.axhline(T_LIM, color=PINK, ls="--", lw=1.8); label(ax, 0.398, T_LIM + 25, f"タービン入口温度の目安 {T_LIM:,.0f} K", color="#c0569a", ha="right", va="bottom", size=14)
    ax.set_xlim(0.12, 0.40); ax.set_ylim(200, 1700); legend(ax, loc="lower right"); commas(ax)
    save(f, "1dcae2_29_gg_lead_time_v3.png")

    f, ax = chart("GGのLOX弁のタイミングと、GG出口温度のピーク", "LOX弁のずれ [ms]（マイナス＝LOXが先、プラス＝燃料が先）", "GG出口温度のピーク [K]",
                  "燃料が先なら温度は設計値のまま。LOXが先だと、燃料が届いた直後は極端な酸素過剰で、\nそこから燃料が増えていく途中で、燃焼温度が最高になる O/F 3 付近を通り抜ける")
    ax.axvspan(-45, 0, color=PINK, alpha=0.14, lw=0)
    ax.plot(lags * 1e3, Tpk, "o-", color=LINK, lw=2.8, ms=9)
    ax.axhline(T_LIM, color=PINK, ls="--", lw=1.8); label(ax, 43, T_LIM + 30, f"タービン入口温度の目安 {T_LIM:,.0f} K", color="#c0569a", ha="right", va="bottom", size=14)
    label(ax, -42, 2440, "LOX が先", va="center"); label(ax, 42, 2440, "燃料が先", ha="right", va="center")
    ax.set_xlim(-45, 45); ax.set_ylim(600, 2600); commas(ax)
    save(f, "1dcae2_30_gg_lead_peak_v3.png")

    # ----- 主弁の LOX 先行／遅れ
    mlags = np.round(np.arange(-0.15, 0.151, 0.025), 3)
    rpk2, ofmax, ofmin, tcmax, mh = [], [], [], [], {}
    for lg in mlags:
        rr = simulate(GGEngine(main_lox_lag=float(lg)), 2.0, 2e-4, record_every=2)
        msk = rr["mdot_c"] > 30.0
        rpk2.append(float(rr["rpm"].max())); ofmax.append(float(rr["OF_mcc"][msk].max())); ofmin.append(float(rr["OF_mcc"][msk].min())); tcmax.append(float(rr["T_c_K"].max()))
        if round(lg * 1e3) in (-50, 0, 50):
            mh[round(lg * 1e3)] = rr
    S["gg_mainlag"] = dict(lags_ms=(mlags * 1e3).tolist(), rpm_peak=rpk2, OF_max=ofmax, OF_min=ofmin, Tc_max=tcmax)
    f, ax = chart("主弁を開く順番と、主燃焼室の酸燃比の時間変化", "時間 [s]", "主燃焼室の O/F（対数目盛）",
                  "燃料の主弁は 0.3 s から 0.6 s かけて開く。LOXの主弁を 50 ms 早く／遅く開けた場合の比較。流量 30 kg/s 以上の区間を表示")
    ax.axhspan(2.6, 3.4, color=STAR, alpha=0.25, lw=0); label(ax, 1.18, 2.98, "燃焼温度が最高になる範囲", ha="right", va="center", size=13)
    for k, c, ls, lab in [(-50, PINK, "-", "LOX弁が 50 ms 先"), (0, LINK, "-", "同時"), (50, INK, "--", "LOX弁が 50 ms 後（燃料が先）")]:
        msk = mh[k]["mdot_c"] > 30.0
        ax.plot(mh[k]["t"], np.where(msk, np.clip(mh[k]["OF_mcc"], 0.05, 100), np.nan), color=c, lw=2.8, ls=ls, label=lab)
    ax.axhline(d["OF_c"], color=INK, ls=":", lw=1.4); label(ax, 1.18, d["OF_c"] * 0.86, f"設計 {d['OF_c']:.2f}", color=MUTED, ha="right", va="top", size=14)
    ax.set_yscale("log"); ax.set_ylim(0.5, 12); ax.set_xlim(0.28, 1.2)
    ax.set_yticks([0.5, 1, 2, 5, 10]); ax.set_yticklabels(["0.5", "1", "2", "5", "10"]); ax.minorticks_off(); legend(ax, loc="upper right")
    save(f, "1dcae2_31_main_lead_of_v3.png")

    f, ax = chart("主LOX弁のタイミングと、回転数のピーク", "主LOX弁のずれ [ms]（マイナス＝LOXが先、プラス＝燃料が先）", "回転数のピーク [×1,000 rpm]",
                  "LOXの主弁が遅れるほど、LOXポンプに負荷がかかるのが遅れて回転数のピークが上がる")
    ax.axvspan(-160, 0, color=PINK, alpha=0.14, lw=0)
    ax.plot(mlags * 1e3, np.array(rpk2) / 1e3, "o-", color=LINK, lw=2.8, ms=9)
    ax.axhline(RPM_D / 1e3, color=INK, ls=":", lw=1.4); label(ax, 155, RPM_D / 1e3 + 0.3, f"設計回転数 {RPM_D:,.0f} rpm", color=MUTED, ha="right", va="bottom", size=14)
    label(ax, -152, 51.3, "LOX が先：酸素過剰を通過", va="center"); label(ax, 152, 39.0, "燃料が先：回転数が上がる", ha="right", va="center")
    ax.set_xlim(-160, 160); ax.set_ylim(34, 53); ax.set_yticks(range(34, 53, 2))
    save(f, "1dcae2_32_main_lead_rpm_v3.png")

    # ----- 主弁の全開時刻 → 回転数のピーク
    t1s = np.round(np.arange(0.4, 1.61, 0.1), 2)
    peak, pdmax = [], []
    for t1 in t1s:
        rr = simulate(GGEngine(main_valve_t=(0.30, float(t1))), 2.5, 2e-4, record_every=5)
        peak.append(float(rr["rpm"].max())); pdmax.append(float(rr["Pd_o_MPa"].max()))
    S["gg_valve"] = dict(t1=t1s.tolist(), peak=peak, Pd_max=pdmax)
    f, ax = chart("主弁を開き切る時刻と、回転数のピーク", "主弁が全開になる時刻 [s]（開き始めは 0.3 s で固定）", "回転数のピーク [×1,000 rpm]",
                  "主弁をゆっくり開くほど、ポンプに負荷がかかるのが遅れ、そのあいだにタービンが軸を加速してしまう")
    ax.plot(t1s, np.array(peak) / 1e3, "o-", color=LINK, lw=2.8, ms=9)
    ax.axhline(RPM_D / 1e3, color=INK, ls=":", lw=1.4); label(ax, 1.62, RPM_D / 1e3 + 0.3, f"設計回転数 {RPM_D:,.0f} rpm", color=MUTED, ha="right", va="bottom", size=14)
    ib = int(np.argmin(abs(t1s - 0.9)))
    ax.annotate("基準ケース", (t1s[ib], peak[ib] / 1e3), xytext=(1.05, 41.5), fontsize=15, color=INK, va="center", arrowprops=dict(arrowstyle="->", color=INK, lw=1.3, shrinkB=7))
    ax.set_xlim(0.34, 1.66); ax.set_ylim(34, 52); ax.set_yticks(range(34, 52, 2))
    save(f, "1dcae2_33_valve_rpm_v3.png")

    # ----- 起動トルク（スピンスタートの強さ）→ 回転数のピーク
    tqs = np.arange(100.0, 401.0, 50.0)
    tpk, t90s = [], []
    for tq in tqs:
        rr = simulate(GGEngine(spin_torque=float(tq)), 3.0, 2e-4, record_every=5)
        tpk.append(float(rr["rpm"].max())); t90s.append(float(rr["t"][np.argmax(rr["F_kN"] > 0.9 * rr["F_kN"][-1])]))
    S["gg_spin"] = dict(torque=tqs.tolist(), peak=tpk, t90=t90s, base_torque=GGEngine().spin_torque, base_time=GGEngine().spin_time)
    f, ax = chart("起動トルクの強さと、回転数のピーク", "起動トルク [N·m]（0.5秒間かける）", "回転数のピーク [×1,000 rpm]",
                  "起動トルクが強いほど立ち上がりは速いが、回転数の行き過ぎも大きくなる。\n点の横の数字は、推力が定常値の90 %に達するまでの時間")
    ax.plot(tqs, np.array(tpk) / 1e3, "o-", color=LINK, lw=2.8, ms=9)
    for x, y, tt in zip(tqs, tpk, t90s):
        low = y / 1e3 < RPM_D / 1e3 + 2.5                          # 設計回転数の点線に近い点は、数字を左側に置く
        if low:
            label(ax, x - 8, y / 1e3, f"{tt:.2f} s", color=MUTED, ha="right", va="center", size=14)
        else:
            label(ax, x + 6, y / 1e3 - 0.55, f"{tt:.2f} s", color=MUTED, ha="left", va="top", size=14)
    ax.axhline(RPM_D / 1e3, color=INK, ls=":", lw=1.4); label(ax, 418, RPM_D / 1e3 + 0.3, f"設計回転数 {RPM_D:,.0f} rpm", color=MUTED, ha="right", va="bottom", size=14)
    ib = int(np.argmin(abs(tqs - 250)))
    ax.annotate("基準ケース", (tqs[ib], tpk[ib] / 1e3), xytext=(185, 48.5), fontsize=15, color=INK, va="center", ha="center", arrowprops=dict(arrowstyle="->", color=INK, lw=1.3, shrinkB=7))
    ax.set_xlim(55, 420); ax.set_ylim(34, 53); ax.set_yticks(range(34, 53, 2)); ax.set_xticks(range(100, 401, 50))
    save(f, "1dcae2_41_spin_torque_v3.png")

    # ----- GG の流量を絞る（スロットリング）
    ths = np.round(np.arange(0.4, 1.01, 0.1), 2)
    TH = [ss_run(A_gg_orifice_scale=float(x)) for x in ths]
    S["gg_throttle"] = dict(scales=ths.tolist(), runs=TH)
    f, ax = chart("GGへの流量を絞ると、エンジン全体が絞られる", "GGオリフィスの開き具合 [%]（燃料・LOXとも同じ割合で絞る）", "推力 [設計値に対する %]",
                  "GGの酸燃比は変えずに流量だけを絞った定常値。右の数字はそのときの比推力（海面）")
    Fp = [g["F_kN"] / 845 * 100 for g in TH]
    ax.plot(ths * 100, Fp, "o-", color=LINK, lw=2.8, ms=9)
    for x, y, g in zip(ths * 100, Fp, TH):
        label(ax, x + 1.6, y - 2.0, f"{g['Isp']:.0f} s", color=MUTED, ha="left", va="top", size=14)
    ax.set_xlim(35, 108); ax.set_ylim(30, 108)
    save(f, "1dcae2_34_throttle_v3.png")

    # ----- 設計の仮定を振る（時間変化なしの計算）: タービン効率 → GG 流量と比推力
    ets = np.round(np.arange(0.35, 0.701, 0.05), 2)
    DS = [design_point(eta_turb=float(e)) for e in ets]
    rng = [design_point(eta_turb=e, T_gg=T, eta_pump=p) for e in (0.40, 0.45, 0.50) for T in (880.0, 900.0, 920.0) for p in (0.72, 0.75, 0.78)]
    hh = design_point(eta_turb=0.582, T_gg=1033.0)                       # タービンだけ教科書の例題エンジンの値に
    h2 = design_point(**HH_A1)                                           # ポンプも含めて置き換え
    PUB = dict(Isp_sl=282.0, Isp_vac=311.0, W_MW=10000 * 0.7457 / 1e3)   # 公表値: 比推力 282 s / 311 s、ターボポンプ 10,000 馬力
    S["gg_assume"] = dict(eta_t=ets.tolist(), Isp_sl=[x["Isp_sl"] for x in DS], gg_pct=[x["gg_frac"] * 100 for x in DS],
                          Isp_min=min(x["Isp_sl"] for x in rng), Isp_max=max(x["Isp_sl"] for x in rng), Isp_vac_min=min(x["Isp_vac"] for x in rng), Isp_vac_max=max(x["Isp_vac"] for x in rng),
                          gg_min=min(x["gg_frac"] for x in rng) * 100, gg_max=max(x["gg_frac"] for x in rng) * 100,
                          hh_Isp=hh["Isp_sl"], hh_Isp_vac=hh["Isp_vac"], hh_gg=hh["gg_frac"] * 100, hh_OF_g=hh["OF_g"],
                          a1=dict(Isp_sl=h2["Isp_sl"], Isp_vac=h2["Isp_vac"], W_MW=h2["W_pump"] / 1e6, gg=h2["gg_frac"] * 100, m_tot=h2["m_c"] + h2["m_g"],
                                  D_t=h2["D_t"], OF_engine=h2["OF_engine"], OF_g=h2["OF_g"], Pd_o=h2["Pd_o"], Pd_f=h2["Pd_f"], **HH_A1),
                          base=dict(Isp_sl=d["Isp_sl"], Isp_vac=d["Isp_vac"], W_MW=d["W_pump"] / 1e6, gg=d["gg_frac"] * 100, m_tot=d["m_c"] + d["m_g"]), pub=PUB)
    f, ax = chart("タービン効率の仮定と、エンジンの比推力", "GGタービンの効率 [%]", "海面比推力 [s]",
                  "推力 845 kN・燃焼圧 9.7 MPa になるように、効率ごとにGGの大きさを決め直した計算（時間変化なし）。\n下に並べた数字は、そのときGGに回す推進剤の割合")
    ax.plot(ets * 100, [x["Isp_sl"] for x in DS], "o-", color=LINK, lw=2.8, ms=9)
    for e, x in zip(ets * 100, DS):
        label(ax, e, 272.45, f"{x['gg_frac'] * 100:.1f} %", color=MUTED, ha="center", va="bottom", size=14)
    ax.axhline(282, color=PINK, ls="--", lw=1.8); label(ax, 35.5, 282.25, "Merlin 1D の公表値 282 s", color="#c0569a", va="bottom", size=14)
    ax.plot([45], [d["Isp_sl"]], "o", color=STAR, ms=13, mec=INK, mew=1.5, zorder=5)
    ax.annotate("今回の基準\n（第3回と同じ 45 %）", (45, d["Isp_sl"]), xytext=(40.5, 280.4), fontsize=15, color=INK, va="center", ha="center", arrowprops=dict(arrowstyle="->", color=INK, lw=1.3, shrinkB=8))
    ax.set_xlim(33, 74); ax.set_ylim(272, 285)
    save(f, "1dcae2_35_turbine_eta_v3.png")

    # ----- ターボポンプの動力と燃焼圧: 海面推力 845 kN・真空推力 914 kN の組から、現行型の燃焼圧を逆算する
    F_VAC_PUB, D_ENG_PUB = 914e3, 0.92                                   # 公表値: 真空推力、エンジンの直径（Wikipedia）
    Pc_inf = bisect(lambda pc: design_point(Pc=pc)["F_vac"] - F_VAC_PUB, 9e6, 16e6)   # 真空推力が 914 kN になる燃焼圧（開口比 16 のまま）
    di, da = design_point(Pc=Pc_inf), design_point(Pc=Pc_inf, **HH_A1)
    pcs = np.arange(9.0, 13.01, 0.25)
    Wb = [design_point(Pc=x * 1e6)["W_pump"] / 1e6 for x in pcs]
    Wa = [design_point(Pc=x * 1e6, **HH_A1)["W_pump"] / 1e6 for x in pcs]
    S["gg_assume"]["pc_inferred"] = dict(Pc=Pc_inf, D_t=di["D_t"], D_e=2 * math.sqrt(di["A_e"] / math.pi), dF=F_VAC_PUB - d["F_sl"], F_vac_base=d["F_vac"],
                                         D_e_base=2 * math.sqrt(d["A_e"] / math.pi), W_base=di["W_pump"] / 1e6, W_a1=da["W_pump"] / 1e6, gg_base=di["gg_frac"] * 100, gg_a1=da["gg_frac"] * 100,
                                         Isp_base=[di["Isp_sl"], di["Isp_vac"]], Isp_a1=[da["Isp_sl"], da["Isp_vac"]], F_vac_pub=F_VAC_PUB, D_eng_pub=D_ENG_PUB)
    f, ax = chart("燃焼圧とターボポンプの動力（海面推力 845 kN）", "燃焼圧 [MPa]", "ターボポンプの動力 [MW]",
                  "推力 845 kN になるように、燃焼圧ごとにスロートとGGの大きさを決め直した計算（時間変化なし）。\n教科書の値＝Huzel & Huang の例題エンジンのタービン効率・入口温度、ポンプの吐出圧・効率")
    ax.axhline(PUB["W_MW"], color=PINK, ls="--", lw=1.8); label(ax, 9.05, PUB["W_MW"] + 0.1, "紹介されている値 10,000 馬力（約 7.5 MW）", color="#c0569a", va="bottom", size=14)
    ax.plot(pcs, Wb, "o-", color=LINK, lw=2.8, ms=8, label="今回の基準（第3回と同じ仮定）")
    ax.plot(pcs, Wa, "s-", color=STAR, lw=2.8, ms=8, mec=INK, mew=0.8, label="教科書の値に置き換え")
    ax.axvline(9.7, color=INK, ls=":", lw=1.4); label(ax, 9.75, 4.25, "9.7 MPa\n（2011年の設計目標）", va="bottom", size=14, color=MUTED)
    ax.axvline(Pc_inf / 1e6, color=INK, ls=":", lw=1.4); label(ax, Pc_inf / 1e6 - 0.05, 4.25, f"約 {Pc_inf / 1e6:.0f} MPa\n（推力から逆算）", va="bottom", ha="right", size=14, color=MUTED)
    ax.set_xlim(8.9, 13.1); ax.set_ylim(4, 10); legend(ax, loc="upper left", bbox_to_anchor=(0.0, 0.93))
    save(f, "1dcae2_40_pump_power_pc_v3.png")

    # ----- 起動の 1 秒と燃焼の 150 秒を 1 本の計算で（主燃焼室の LOX 噴射器が 2 % 渋いとして）
    t0 = time.time()
    eng = GGEngine(mcc_lox_scale=0.98, burn_time=150.0)
    rr = simulate(eng, 170.0, 5e-4, record_every=20, stop_when=lambda t, s: t > 5 and s["Pc"] < 1.5e5)
    tb = rr["t"]
    i_off = int(np.argmax((rr["F_kN"] < 0.5 * 845) & (tb > 5)))
    first_empty = "fuel" if rr["m_f"][-1] <= 0 else "lox"
    i_mid = int(np.searchsorted(tb, 75.0))
    S["gg_burn"] = dict(run_s=time.time() - t0, steps=int(round(tb[-1] / 5e-4)), t_off=float(tb[i_off]), m_f0=eng.m_f0, m_o0=eng.m_o0, first_empty=first_empty,
                        m_f_left=float(rr["m_f"][-1]), m_o_left=float(rr["m_o"][-1]), F_mid=float(rr["F_kN"][i_mid]), OF_engine=float(rr["OF_engine"][i_mid]),
                        Isp_mid=float(rr["Isp"][i_mid]), OF_mcc=float(rr["OF_mcc"][i_mid]), rpm_mid=float(rr["rpm"][i_mid]))
    f, ax = chart("起動の1秒と燃焼の150秒を、1本の計算で", "時間 [s]（対数目盛）", "推力 [kN]",
                  "同じモデルを、起動から推進剤が尽きるまで時間刻み 0.5 ms で計算した結果。\n横軸を対数にとると、起動（1秒以内）と燃焼全体が1枚に収まる")
    ax.plot(tb[1:], rr["F_kN"][1:], color=LINK, lw=2.8)
    ax.axvspan(0.1, 1.0, color=SKY, alpha=0.22, lw=0); label(ax, 0.32, 975, "起動", ha="center", va="top", size=14)
    label(ax, 12, 975, "定常燃焼", ha="center", va="top", size=14)
    ax.annotate(f"{S['gg_burn']['t_off']:.0f} s で\n燃焼終了", (S["gg_burn"]["t_off"], 420), xytext=(38, 420), fontsize=15, color=INK, va="center", ha="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.3, shrinkB=4))
    ax.set_xscale("log"); ax.set_xlim(0.1, 250); ax.set_ylim(0, 1000); commas(ax)
    ax.set_xticks([0.1, 1, 10, 100]); ax.set_xticklabels(["0.1", "1", "10", "100"])
    save(f, "1dcae2_36_burn_thrust_v3.png")

    f, ax = chart("タンクの残量：どちらが先に尽きるか", "時間 [s]", "タンクの推進剤 [t]",
                  "150秒分の推進剤を、設計どおりの比率で積んだ場合。主燃焼室のLOX噴射器が設計より 2 % 流れにくいと仮定")
    ax.plot(tb, rr["m_o"] / 1e3, color=SKY, lw=3.0, label="液体酸素（LOX）")
    ax.plot(tb, rr["m_f"] / 1e3, color=PINK, lw=3.0, ls="--", label="燃料（ケロシン）")
    left = S["gg_burn"]["m_o_left"] if first_empty == "fuel" else S["gg_burn"]["m_f_left"]
    who = "燃料が先に尽きる" if first_empty == "fuel" else "LOXが先に尽きる"
    rest = "LOX" if first_empty == "fuel" else "燃料"
    ax.annotate(f"{who}\n{rest}が約 {left:,.0f} kg 残る", (tb[-1], left / 1e3), xytext=(118, 12.5), fontsize=15, color=INK, va="center", ha="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.3, shrinkB=5))
    ax.set_xlim(0, 160); ax.set_ylim(0, 34); legend(ax, loc="upper right")
    save(f, "1dcae2_37_burn_tank_v3.png")

    # ----- 時間刻みと積分法（うまくいかない話）
    eng = GGEngine()
    ref = simulate(eng, 1.5, 2e-5, method="rk4", record_every=1)
    dts = [2e-4, 5e-4, 1e-3, 1.5e-3, 1.7e-3, 2e-3, 2.3e-3, 2.5e-3, 3e-3]
    err = {m: [] for m in ("euler", "rk2", "rk4")}
    for m in err:
        for dt in dts:
            try:
                q = simulate(GGEngine(), 1.5, dt, method=m, record_every=1)
                e = float(np.max(np.abs(q["Pc_MPa"] - np.interp(q["t"], ref["t"], ref["Pc_MPa"]))))
            except FloatingPointError:
                e = float("inf")
            err[m].append(e)
    # 固有値（定常点まわりで線形化）
    s_ss = {k: float(r[k][-1]) for k in ("m_f", "m_o", "omega", "Pc", "Pgg", "Tc", "Tgg")}
    keys = ["omega", "Pc", "Pgg", "Tc", "Tgg"]

    def rhs(v):
        s = dict(s_ss, **dict(zip(keys, v)))
        dd = eng.derivatives(3.0, s)
        return np.array([dd[k] for k in keys])
    x0 = np.array([s_ss[k] for k in keys]); J = np.zeros((5, 5))
    for j in range(5):
        h = x0[j] * 1e-5; xp, xm = x0.copy(), x0.copy(); xp[j] += h; xm[j] -= h
        J[:, j] = (rhs(xp) - rhs(xm)) / (2 * h)
    ev = np.linalg.eigvals(J)
    lam = float(max(abs(ev)))
    m_gas = s_ss["Pc"] * eng.V_c / (eng.R_c * s_ss["Tc"])
    S["gg_int"] = dict(dts_ms=[x * 1e3 for x in dts], err=err, eig_real=sorted(float(x.real) for x in ev), eig_imag_max=float(max(abs(ev.imag))), lam_max=lam,
                       tau_fast_ms=1e3 / lam, dt_lim_euler_ms=2e3 / lam, dt_lim_rk4_ms=2.785e3 / lam, m_gas_c=m_gas, tau_res_ms=1e3 * m_gas / b["mdot_c"],
                       m_gas_gg=s_ss["Pgg"] * eng.V_gg / (d["R_g"] * s_ss["Tgg"]), tau_res_gg_ms=1e3 * s_ss["Pgg"] * eng.V_gg / (d["R_g"] * s_ss["Tgg"]) / b["mdot_gg"])
    # 精度の次数: 時間刻みを 0.5 ms → 0.2 ms にしたときの誤差の減り方から求める。起動トルクを途中で切らない場合（段差なし）も計算して比べる
    def orders(**kw):
        rf = ref if not kw else simulate(GGEngine(**kw), 1.5, 2e-5, method="rk4", record_every=1)
        out = {}
        for mth in ("euler", "rk2", "rk4"):
            e = []
            for dt in (2e-4, 5e-4):
                q = simulate(GGEngine(**kw), 1.5, dt, method=mth, record_every=1)
                e.append(float(np.max(np.abs(q["Pc_MPa"] - np.interp(q["t"], rf["t"], rf["Pc_MPa"])))))
            out[mth] = float(np.log(e[1] / e[0]) / np.log(2.5))
        return out
    S["gg_int"]["order"] = orders()
    S["gg_int"]["order_no_step"] = orders(spin_time=10.0)
    names = {"euler": "オイラー法", "rk2": "2段ルンゲ＝クッタ法", "rk4": "4段ルンゲ＝クッタ法"}
    cols = {"euler": PINK, "rk2": STAR, "rk4": LINK}
    f, ax = chart("同じモデル・同じ時間刻み 2 ms でも、積分法で結果が変わる", "時間 [s]", "主燃焼室の圧力 [MPa]",
                  "0.8 s 近くまでは3本とも重なっている。燃焼室圧が上がりきったところで、オイラー法は振動しながら、\n2段ルンゲ＝クッタ法は振動せずに発散する（×は計算が止まった点）。4段ルンゲ＝クッタ法は安定")
    blow = {}
    for m, lw, z in (("euler", 1.8, 2), ("rk2", 1.8, 3), ("rk4", 3.0, 4)):
        try:
            q = simulate(GGEngine(), 1.4, 2e-3, method=m, record_every=1); stopped = None
        except FloatingPointError as ex:
            q = ex.result; stopped = float(q["t"][-1])
        bad = np.abs(q["Pc_MPa"] - np.interp(q["t"], ref["t"], ref["Pc_MPa"])) > 1.0
        blow[m] = dict(first_bad=float(q["t"][np.argmax(bad)]) if bad.any() else None, stopped=stopped, Pc_max=float(q["Pc_MPa"].max()))
        ax.plot(q["t"], q["Pc_MPa"], color=cols[m], lw=lw, zorder=z, label=names[m] + ("（発散）" if bad.any() else ""))
        if stopped is not None:
            ax.plot([q["t"][-1]], [min(max(q["Pc_MPa"][-1], 0.4), 19.5)], "x", color=cols[m], ms=14, mew=3.5, zorder=6, clip_on=False)
    S["gg_int"]["blowup_t_2ms"] = blow
    ax.set_xlim(0.6, 1.0); ax.set_ylim(0, 20); ax.set_yticks(range(0, 21, 5)); legend(ax, loc="upper left")
    save(f, "1dcae2_38_int_time_v3.png")

    f, ax = chart("時間刻みと計算誤差", "時間刻み Δt [ms]", "燃焼室圧の最大誤差 [MPa]（対数目盛）",
                  "基準は時間刻み 0.02 ms の4段ルンゲ＝クッタ法。×は発散（誤差 1 MPa 以上）。\n縦の点線は、固有値から見積もった安定限界")
    for m, mk in (("euler", "o"), ("rk2", "s"), ("rk4", "D")):
        e = np.array(err[m]); ok = np.isfinite(e) & (e < 1.0)
        ax.plot(np.array(dts)[ok] * 1e3, np.maximum(e[ok], 1e-6), mk + "-", color=cols[m], lw=2.8, ms=9, label=names[m])
        if (~ok).any():
            ax.plot(np.array(dts)[~ok] * 1e3, np.full((~ok).sum(), {"euler": 4.2, "rk2": 2.0, "rk4": 0.95}[m]), "x", color=cols[m], ms=12, mew=3)
    for x, lab in ((S["gg_int"]["dt_lim_euler_ms"], "オイラー法・2段の限界"), (S["gg_int"]["dt_lim_rk4_ms"], "4段の限界")):
        ax.axvline(x, color=INK, ls=":", lw=1.4)
    label(ax, S["gg_int"]["dt_lim_euler_ms"] - 0.04, 2.2e-5, f"オイラー法・2段の限界\n{S['gg_int']['dt_lim_euler_ms']:.2f} ms", ha="right", va="center", size=13, color=MUTED)
    label(ax, S["gg_int"]["dt_lim_rk4_ms"] + 0.04, 2.2e-5, f"4段の限界\n{S['gg_int']['dt_lim_rk4_ms']:.2f} ms", ha="left", va="center", size=13, color=MUTED)
    label(ax, 3.15, 0.33, "× ＝ 発散", ha="right", va="center", size=14)
    ax.set_yscale("log"); ax.set_ylim(1e-5, 10.0); ax.set_xlim(0, 3.2); legend(ax, loc="upper left")
    save(f, "1dcae2_39_int_error_v3.png")


if __name__ == "__main__":
    summary = os.path.join(OUT, "summary.json")
    if "rocket" not in sys.argv:
        study_pumped()
    if "pumped" not in sys.argv:
        study_rocket()
    old = json.load(open(summary, encoding="utf-8")) if os.path.exists(summary) else {}
    old.update(S)
    json.dump(old, open(summary, "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=float)
    print("summary ->", summary)
