# 1D-simulation-demo (onedsim)

Minimal 1D (lumped-parameter) system simulator in pure Python + numpy, with two demo models: a pumped-storage hydro plant and a gas-generator-cycle rocket engine (start-up, shutdown and burn-out). Companion code for the makkiblog.com article series 【1Dシミュレーション】.

## onedsim — まきブログ版・最小1Dシミュレーションエンジン

【1Dシミュレーション②】（makkiblog.com）で使った、揚水発電所とガスジェネレータサイクルロケットエンジンの
1Dモデル一式です。依存は **numpy と matplotlib だけ**。

```
onedsim.py          ソルバー本体（Table1D / bisect / Model / simulate / sweep）約170行
pumped_storage.py   揚水発電所のモデル（神流川発電所の公表値を参考にした数値）
gg_rocket.py        ガスジェネレータサイクルのロケットエンジン（Merlin 1D 級）。設計計算 + 起動・停止・燃焼終了までの動的モデル
cea_tables.py       gg_rocket.py に埋め込んだ燃焼ガスの表を RocketCEA で作り直すスクリプト（任意）
run_studies.py      記事のグラフと数値をまとめて再計算（./fig に PNG と summary.json を出力）
test_gg_rocket.py   ロケットエンジンのモデルの確認（定常値、エネルギーと質量の保存、手順を変えたときの向き）。pytest で実行
```

## 動かし方

```
pip install -r requirements.txt
python pumped_storage.py       # 7日間の運転を計算して要約を表示（約1秒）
python gg_rocket.py            # 設計計算（教科書の値に置き換えた場合も）と、起動から停止までの結果を表示（数秒）
python run_studies.py          # 記事の全グラフを再生成（10〜15分）
python run_studies.py pumped   # 揚水発電所だけ
python run_studies.py rocket   # ロケットエンジンだけ
python run_studies.py solver   # 時間積分の方法の図だけ
```

uv を使う場合は、次のどちらでも動きます。

```
uv run --with-requirements requirements.txt pumped_storage.py   # ライブラリの用意から実行まで 1 行（仮想環境を残さない）

uv venv                                  # 仮想環境を作って使う場合
uv pip install -r requirements.txt
uv run python run_studies.py
```

モデルの確認は `pip install pytest` のあと `python -m pytest -q` で実行できます（1分ほど）。uv なら `uv run --with pytest --with-requirements requirements.txt python -m pytest -q` です。

## 自分のモデルを作る

```python
from onedsim import Model, Table1D, simulate

class Tank(Model):
    def initial_state(self):          return {"V": 100.0}                 # 状態量
    def derivatives(self, t, s):      return {"V": 2.0 - 0.05 * s["V"]}   # dV/dt
    def outputs(self, t, s):          return {"level": s["V"] / 10}       # 記録したい量

res = simulate(Tank(), t_end=100, dt=0.1)      # res["t"], res["V"], res["level"] は numpy 配列
```

- 時間積分は `method="rk4"`（4段ルンゲ＝クッタ法、既定）/ `"rk2"`（2段ルンゲ＝クッタ法）/ `"euler"`（オイラー法）から選べます。
- ポンプの Q-H 曲線のような特性データは `Table1D(x, y)` に入れて `table(q)` で引きます。範囲の外は端の値で止まります。
- 部品どうしのつり合い点（ポンプ曲線と配管抵抗の交点など）は `bisect(f, a, b)` で探します。
- 状態量が無限大や NaN になったとき（または途中の計算で負の数の平方根などが出たとき）は `FloatingPointError` が出ます。そこまでの記録は例外の `.result` に入っています。値が有限のまま発散する場合（振動し続ける、ありえない値に張り付く）は例外になりません。時間刻みを半分にして結果が変わらないかを確認してください。

## 揚水発電所のモデル

神流川発電所の公表値を参考にした 2 台構成です。実機の特性曲線は公表されていないので、公表されている仕様に合うように曲線の形を仮定しています
（水車: 有効落差 653 m・流量 85 m³/s で出力 482 MW、ポンプ: 揚水量 60 m³/s・最大軸入力 464 MW、発電機の出力は 470 MW/台。モデルの落差 680 m のとき、85 m³/s での有効落差は約 651 m）。

```python
from pumped_storage import PumpedStorage, make_demand, make_rain

PumpedStorage(n_units=6)                                  # 6 台構成（完成時の計画）
PumpedStorage(demand=make_demand(peak_MW=800.0))          # ピーク時の発電要求を変える
PumpedStorage(demand=make_demand(heatwave_day=2, heatwave_MW=940.0), rain=make_rain(storm_day=5, mm_per_h=30.0, hours=12.0))   # 記事の基準ケース
PumpedStorage(rain=make_rain(storm_day=5, mm_per_h=50.0, hours=24.0), pump_in_rain=False)   # 雨のあいだは揚水しない運用
```

- 揚水は、余った電力（需要スケジュールのマイナスの値）でまかなえる台数だけ動かします（1 台あたり約 450 MW。固定速なので入力は落差で決まります）。
- 上池が満水のあいだは揚水しません。満水を超えた雨は越流として捨て、発電できたはずの電力量を `E_spill` に積算します。
- 往復効率は `roundtrip(res)`（水 1 m³ あたりの発電量 ÷ 揚水電力量）で求めます。雨で増えた水や、週の初めと終わりの水位の差に左右されません。

## ロケットエンジンのモデル

前提は `gg_rocket.DESIGN` にまとめてあります（Wikipedia に載っている Merlin 1D の値: 海面推力 845 kN・燃焼圧 9.7 MPa・ノズル開口比 16。
ポンプ効率 75 %、GG タービン効率 45 %・入口温度 900 K などは仮定）。燃焼圧 9.7 MPa は 2011 年に発表された設計目標の値で、
推力を 845 kN に上げた現行型の燃焼圧は公表されていません（推力の公表値から逆算すると約 12 MPa。記事を参照）。

```python
from gg_rocket import GGEngine, design_point

design_point()                          # 時間変化なしの設計計算（スロート径、GG 流量、比推力など）
design_point(eta_turb=0.6, T_gg=1000)   # 仮定を変えて設計し直す
design_point(**HH_A1)                   # ターボポンプの値を教科書の例題エンジン（Huzel & Huang の A-1）に置き換える（from gg_rocket import HH_A1）
design_point(k_discharge_f=1.7)         # ポンプの吐出圧（燃焼圧に対する比）や効率は、LOX 側（_o）と燃料側（_f）を別々にも指定できる
design_point(Pc=12e6)                   # 燃焼圧を変える（推力 845 kN になるようにスロートを決め直す）

GGEngine(gg_lox_scale=1.1)              # GG の LOX オリフィスを +10 %（GG の O/F が上がる）
GGEngine(mcc_lox_scale=0.9)             # 主燃焼室の LOX 噴射器を -10 %
GGEngine(A_gg_orifice_scale=0.5)        # GG のオリフィスを両方とも 50 % に絞る（推力を絞る）
GGEngine(gg_lox_lag=-0.01)              # GG の LOX 弁を燃料弁より 10 ms 先に開く（マイナス = LOX が先）
GGEngine(main_lox_lag=0.05)             # 主 LOX 弁を燃料弁より 50 ms 遅らせる
GGEngine(main_valve_t=(0.2, 0.9))       # 主弁の開き始めと全開の時刻（既定は 0.2〜0.5 秒）
GGEngine(gg_valve_t=0.25)               # GG 弁を開く時刻（既定は 0.15 秒）
GGEngine(spin_torque=150.0)             # 起動トルク [N·m]（スピンスタートの強さ。既定は 250 N·m を 0.5 秒間）
GGEngine(burn_time=150.0)               # タンクに積む推進剤（設計流量 × 秒数）

GGEngine(cutoff_t=1.5)                             # 1.5 秒で停止の指令を出す（GG 弁を 0.05 秒、主弁を 0.15 秒で閉じる。どちらも LOX 弁が 20 ms 先）
GGEngine(cutoff_t=1.5, main_cut_t=(0.0, 0.05))     # 主弁を 0.05 秒で閉じる（指令から閉じ始め・閉じ切りまでの時間）
GGEngine(cutoff_t=1.5, gg_cut_t=(0.15, 0.20))      # GG 弁を閉じ始めるのを 0.15 秒遅らせる（回転数が跳ね上がる）
GGEngine(cutoff_t=1.5, gg_lox_cut_lag=0.01)        # GG の燃料弁を LOX 弁より 10 ms 先に閉じる（プラス = 燃料弁が先、マイナス = LOX 弁が先）
```

既定の起動手順は、回転数と圧力の行き過ぎが 1〜2 % に収まるように決めたものです。主弁を開くのを遅らせたり、起動トルクを強くしたりすると、回転数が大きく行き過ぎます。

燃焼ガスの表（燃焼温度 `T_AD`、特性排気速度 `CSTAR`、真空推力係数 `CF_VAC`）は、O/F 0.8 以上が化学平衡計算（RocketCEA、9.7 MPa）、
燃料過剰の GG の領域（O/F 0.30〜0.52）が Huzel & Huang "Design of Liquid Propellant Rocket Engines"（NASA SP-125）Table 6-4 の代表値です（そのあいだと 0.30 より下は、点をつないで補っています）。
`cea_tables.py` を実行すると、平衡計算の部分を作り直せます（`pip install rocketcea` が必要）。

このモデルに入っていないもの: 着火の遅れ、配管の慣性と圧力波（水撃）、燃焼の遅れ（チャギング）、再生冷却の熱、弁より下流に残る推進剤、
タンク圧の変化、機体の加速度やタンクの液面によるポンプ入口圧の変化とキャビテーション、ノズル内の流れの剥離。
ポンプとタービンのマップの形は典型的なものを仮定しています（タービンは設計点で効率が最大）。

## ライセンス
MIT。改造・再配布自由。記事へのリンクを添えてもらえると嬉しいです。
