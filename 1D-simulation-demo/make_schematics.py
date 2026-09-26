"""記事用の系統図（揚水発電所・ソルバー構造）。ロケットの系統図は前編の 1dcae_08_gg_cycle.png を再利用。"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon, Rectangle
import numpy as np, os
from matplotlib import font_manager as _fm
_have = {f.name for f in _fm.fontManager.ttflist}   # 日本語フォントを OS ごとに自動選択（Windows: Yu Gothic / Meiryo）
plt.rcParams["font.family"] = next((f for f in ["Noto Sans CJK JP", "Yu Gothic", "Meiryo", "MS Gothic", "Hiragino Sans", "IPAexGothic"] if f in _have), "sans-serif")
INK, MUTED, LINK, BLUSH, SURF, CARD = "#232a42", "#888888", "#008db7", "#fcb8b8", "#fbfbff", "#ffffff"
TRUDE, READER = "#ffe8ee", "#e0fff3"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig"); os.makedirs(OUT, exist_ok=True)

def new(h): 
    f, ax = plt.subplots(figsize=(10.24, h), dpi=100); f.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]; return f, ax
def box(ax, x, y, w, h, t, fc=CARD, fs=11, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=fc, ec=INK, lw=1.5))
    ax.text(x + w/2, y + h/2, t, ha="center", va="center", fontsize=fs, color=INK, fontweight="bold" if bold else "normal", linespacing=1.4)
def arrow(ax, p, q, color=INK, lw=1.8, style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=15, color=color, lw=lw))
def save(f, n): f.savefig(f"{OUT}/{n}", facecolor=SURF, bbox_inches="tight", pad_inches=0.15); plt.close(f); print("saved", n)

# ---------- 揚水発電所の系統図 ----------
f, ax = new(5.8); ax.set_xlim(0, 12); ax.set_ylim(0, 6.9)
ax.text(6, 6.6, "揚水発電所の1Dモデル（神流川発電所を参考にした数値）", ha="center", fontsize=14, fontweight="bold", color=INK)
# upper reservoir (left, high)
ax.add_patch(Polygon([[0.5, 3.4], [0.5, 5.0], [3.6, 5.0], [3.6, 3.4]], closed=True, fc=CARD, ec=INK, lw=1.6))
ax.add_patch(Rectangle((0.55, 3.45), 3.0, 1.0, fc="#dff1ff", ec="none"))
ax.text(2.05, 4.72, "上池（南相木ダム相当）", ha="center", va="center", fontsize=10, color=INK, fontweight="bold")
ax.text(2.05, 3.95, "V_u（状態量）　面積 A = 40万 m²\n有効貯水量 1,267万 m³", ha="center", va="center", fontsize=9, color=INK)
# rain / evap (above the pond)
for xx in (2.9, 3.2): arrow(ax, (xx, 5.95), (xx, 5.15), color=LINK, lw=1.5)
ax.text(3.4, 5.6, "降雨（集水域 5 km²・流出率 0.6）", fontsize=9, color=LINK, ha="left", va="center")
arrow(ax, (0.9, 5.15), (0.9, 5.95), color=MUTED, lw=1.2); ax.text(1.05, 5.6, "蒸発 4 mm/日", fontsize=9, color=MUTED, va="center")
# lower reservoir (right, low)
ax.add_patch(Polygon([[8.4, 0.5], [8.4, 2.1], [11.5, 2.1], [11.5, 0.5]], closed=True, fc=CARD, ec=INK, lw=1.6))
ax.add_patch(Rectangle((8.45, 0.55), 3.0, 1.0, fc="#dff1ff", ec="none"))
ax.text(9.95, 1.85, "下池（上野ダム相当）", ha="center", va="center", fontsize=10, color=INK, fontweight="bold")
ax.text(9.95, 1.0, "V_l（状態量）", ha="center", va="center", fontsize=9, color=INK)
# penstock
ax.plot([3.6, 5.2, 5.2], [3.8, 3.8, 2.45], color=INK, lw=4); ax.plot([5.2, 5.2, 8.4], [1.35, 1.0, 1.0], color=INK, lw=4)
ax.text(4.4, 3.45, "水路（損失 kQ²）", fontsize=9, color=MUTED, ha="center", va="top")
# pump-turbine
ax.add_patch(Circle((5.2, 1.9), 0.55, fc=TRUDE, ec=INK, lw=1.6)); ax.text(5.2, 1.9, "ポンプ\n水車", ha="center", va="center", fontsize=9, color=INK)
ax.text(5.2, 0.55, "×2台（Q-H 曲線・効率は試験値）", ha="center", fontsize=9, color=INK)
# head
ax.annotate("", (7.0, 3.8), (7.0, 1.0), arrowprops=dict(arrowstyle="<->", color=LINK, lw=1.5)); ax.text(7.15, 2.6, "落差 H ≈ 653 m\n（水位で変化）", fontsize=9, color=LINK, va="center")
# grid / demand
box(ax, 0.5, 1.2, 2.6, 1.3, "電力系統\n（需要スケジュール）", fc=READER, fs=10)
arrow(ax, (3.1, 2.15), (4.65, 2.1), color=LINK); arrow(ax, (4.65, 1.7), (3.1, 1.6), color="#e07a7a")
ax.text(3.85, 2.3, "揚水（夜）", fontsize=8, color=LINK, ha="center"); ax.text(3.85, 1.35, "発電（昼）", fontsize=8, color="#e07a7a", ha="center")
save(f, "1dcae2_01_ps_schematic.png")

# ---------- ソルバー構造 ----------
f, ax = new(4.8); ax.set_xlim(0, 12); ax.set_ylim(0, 5.4)
ax.text(6, 5.1, "onedsim の構造：1Dツールの中身は 3 つだけ", ha="center", fontsize=14, fontweight="bold", color=INK)
box(ax, 0.4, 2.5, 2.8, 1.7, "① Table1D\n試験曲線をテーブル引き\n（Q-H, 効率, ポンプマップ）", fc=READER, fs=10)
box(ax, 4.6, 2.5, 2.8, 1.7, "② Model\n状態量 s と ds/dt\n（質量・エネルギー保存）", fc=TRUDE, fs=10)
box(ax, 8.8, 2.5, 2.8, 1.7, "③ simulate\n時間積分（オイラー法、\n2段・4段ルンゲ＝クッタ法）", fc="#dff1ff", fs=10)
arrow(ax, (3.2, 3.35), (4.6, 3.35)); arrow(ax, (7.4, 3.35), (8.8, 3.35))
ax.text(3.9, 3.6, "部品特性", ha="center", fontsize=9, color=INK); ax.text(8.1, 3.6, "微分方程式", ha="center", fontsize=9, color=INK)
ax.text(6.0, 1.15, "pumped_storage.py / gg_rocket.py\n＝ ② の具体例（部品をつないで系にする）", ha="center", va="center", fontsize=9, color=INK)
arrow(ax, (6.0, 1.6), (6.0, 2.5), color=MUTED, lw=1.2)
box(ax, 8.8, 0.6, 2.8, 1.1, "sweep：パラメータを振って\n③ を何百回も回す", fc=CARD, fs=10)
arrow(ax, (10.2, 2.5), (10.2, 1.7)); ax.text(10.35, 2.1, "結果 res[\"t\"], res[\"Pc\"]…", fontsize=9, color=MUTED, va="center")
ax.text(1.8, 1.15, "numpy だけで動く\n約 130 行", ha="center", va="center", fontsize=9, color=MUTED)
save(f, "1dcae2_02_solver_structure_v2.png")
