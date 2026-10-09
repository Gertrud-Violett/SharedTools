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
run_studies.py      記事の計算結果のグラフと主な数値をまとめて再計算（./fig に PNG と summary.json を出力）
test_onedsim.py     ソルバーの確認（積分法の精度の次数、発散やモデルの誤りの検出）。pytest で実行
test_pumped_storage.py  揚水発電所のモデルの確認（水の収支、台数を変えたときの向き、記事の数値）
test_gg_rocket.py   ロケットエンジンのモデルの確認（定常値、ガスとエネルギーの保存、手順を変えたときの向き）
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
- ポンプの Q-H 曲線のような特性データは `Table1D(x, y)` に入れて `table(q)` で引きます。範囲の外は端の値で止まります（直線で外に延ばしません）。
- 部品どうしのつり合い点（ポンプ曲線と配管抵抗の交点など）は `bisect(f, a, b)` で探します。f(a) と f(b) の符号が同じ（範囲が答えをはさんでいない）ときは `ValueError` になります。
- `t_end` は `dt` の整数倍で指定します（そうでなければ `ValueError`）。最後の状態は、`record_every` が割り切れなくても必ず記録されます。
- 状態量が無限大・NaN・複素数になったとき、またはモデルの計算の途中で例外（負の数の平方根など）が出たときは `FloatingPointError` が出ます。元の例外は `__cause__`（トレースバックの前半）でたどれ、そこまでの記録は例外の `.result` に入っています。式の書き間違いでも同じ例外になるので、元の例外も確かめてください。
- 値が有限のまま発散する場合（振動し続ける、`clamp()` で決めた上限や下限に張り付く）は例外になりません。時間刻みを半分にして結果が変わらないかを確認してください。`clamp()` は各ステップの終わりにだけ呼ばれ、ルンゲ＝クッタ法の途中の段には効きません。

## 揚水発電所のモデル

神流川発電所の公表値を参考にした 2 台構成です。

- 池の形と落差は、公表されている水位から決めています（上池: 満水位 1,527.0 m・最低水位 1,500.0 m、下池: 843.4 m・814.0 m、有効貯水量はどちらも 1,267 万 m³）。
  どちらの池も面積が一定の柱とみなし（面積 = 有効貯水量 ÷ 利用水深。上池 約 46.9 ha、下池 約 43.1 ha）、落差 = 上池の水位 − 下池の水位 です
  （両池が半分のとき 684.8 m、とりうる範囲は 656.6〜713.0 m）。水位の出力 `level_u`、`level_l` は最低水位からの高さです（上池 0〜27.0 m、下池 0〜29.4 m）。
- 水路の損失は公表されていません。損失 = `k_loss` × (1 台あたりの流量)² とし、「両池が半分で 85 m³/s を流したとき、有効落差が公表の 653 m になる」ように
  係数を決めています（損失 31.8 m、`k_loss` ≈ 0.0044）。この決め方は仮定です。`PumpedStorage(k_loss=...)` で変えられます。
  公表の有効落差の範囲は 最高 675 m / 最低 617 m（東芝レビュー 2006）ですが、この損失ではモデルの範囲は 85 m³/s のとき 624.8〜681.2 m で、6〜8 m 高くなります。
  同じ水位の範囲で公表の両端に合うのは損失 38〜40 m 程度です（公表資料は、その範囲が何 m³/s・何台のときの値かを書いていないので、これは推定です）。
  そうすると 1 週間の供給不足は 82 MWh でなく 約 144〜163 MWh、往復効率は 約 70 % になります（`run_studies.py` の `ps_loss_pub`）。
- 実機の特性曲線は公表されていないので、曲線の形は仮定です。公表値に合わせたのは定格点だけです（有効落差 653 m・流量 85 m³/s で水車出力 482 MW、発電機の出力 470 MW/台）。
  ポンプの曲線は公表仕様（全揚程 677〜728 m、揚水量 60 m³/s、最大軸入力 464 MW）に合わせ込んでいません。両池が半分のときの揚水量は約 58 m³/s で、
  上池の水位が低いときは軸入力が 464 MW を 1.4 MW ほど上回ります。

```python
from pumped_storage import PumpedStorage, make_demand, make_rain

PumpedStorage(n_units=6)                                  # 6 台構成（完成時の計画）
PumpedStorage(demand=make_demand(peak_MW=800.0))          # ピーク時の発電要求を変える
PumpedStorage(demand=make_demand(heatwave_day=2, heatwave_MW=940.0), rain=make_rain(storm_day=5, mm_per_h=30.0, hours=12.0))   # 記事の基準ケース
PumpedStorage(rain=make_rain(storm_day=5, mm_per_h=50.0, hours=24.0), pump_in_rain=False)   # 雨のあいだは揚水しない運用
```

- 発電は、要求をまかなえる最小の台数で行います（その落差で 1 台が出せる最大の出力から台数を決め、その台数で等分）。
- 揚水は、余った電力（需要スケジュールのマイナスの値）で動かせる台数だけ動かします（1 台あたり約 450 MW として判定。固定速なので、実際の入力は落差で決まり、余った電力をいくらか上回ることがあります）。
- 上池が満水のあいだは揚水しません。満水を超えた雨は越流として捨て、その水を最大流量（1 台 85 m³/s）で発電に使えたとしたときの電力量（ρ g ×（落差 − 水路の損失）× 水車効率 × 発電機効率）を `E_spill` に積算します。下池が満水になったら、あふれる分は下流へ放流したとして `W_spill_l` に積算します。
- 水の収支は、`V_u + V_l + W_spill_u + W_spill_l + W_evap − W_rain` が一定であることで確かめられます（test_pumped_storage.py）。池の範囲（0〜`V_max`）を 1 ステップで越えた分も `clamp()` が数えます（上に越えた分は越流 `W_spill_u`・`W_spill_l`、下に越えた分は空の池から蒸発できなかった分として `W_evap` から引く）ので、水が湧いたり消えたりしません。
- 往復効率は `roundtrip(res)`（水 1 m³ あたりの発電量 ÷ 揚水電力量）で求めます。雨で増えた水や、週の初めと終わりの水位の差に左右されません。

## ロケットエンジンのモデル

前提は `gg_rocket.DESIGN` にまとめてあります（Wikipedia に載っている Merlin 1D の値: 海面推力 845 kN・燃焼圧 9.7 MPa・ノズル開口比 16。
ポンプ効率 75 %、GG タービン効率 45 %・入口温度 900 K などは仮定）。燃焼圧 9.7 MPa は 2011 年に発表された設計目標の値で、
推力を 845 kN に上げた現行型の燃焼圧は公表されていません（推力の公表値から逆算すると約 13 MPa。記事を参照）。

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
GGEngine(spin_torque=600.0)             # 起動トルク [N·m]（スピンスタートの強さ。止まっているときの値で、既定は 450 N·m）
GGEngine(burn_time=150.0)               # タンクに積む推進剤（設計流量 × 秒数）

GGEngine(cutoff_t=1.5)                             # 1.5 秒で停止の指令を出す（GG 弁を 0.05 秒、主弁を 0.15 秒で閉じる。どちらも LOX 弁が 20 ms 先）
GGEngine(cutoff_t=1.5, main_cut_t=(0.0, 0.05))     # 主弁を 0.05 秒で閉じる（指令から閉じ始め・閉じ切りまでの時間）
GGEngine(cutoff_t=1.5, gg_cut_t=(0.15, 0.20))      # GG 弁を閉じ始めるのを 0.15 秒遅らせる（回転数が跳ね上がる）
GGEngine(cutoff_t=1.5, gg_lox_cut_lag=0.01)        # GG の燃料弁を LOX 弁より 10 ms 先に閉じる（プラス = 燃料弁が先、マイナス = LOX 弁が先）
```

起動トルクは、外部のガスでタービンを回すスピンスタートを模擬したもので、止まっているとき `spin_torque`、回転数とともに直線で下がり、設計回転数で 0 になります
（`GGEngine.spin()`。`spin_time`、既定 0.5 秒で止めます）。起動トルクだけでは、設計回転数を超えません。
既定の起動手順は、回転数と圧力が設計値を行き過ぎないように決めたものです。主弁を開くのを遅らせたり、起動トルクを 500 N·m 以上に強くしたりすると、回転数が行き過ぎます。

GG は、中のガスの質量を 3 つ（全体 `m_gg`、酸化剤からできた分 `mo_gg`、起動前から入っていた窒素 `mi_gg`）状態量として持ちます。
よく混ざっていて反応が速いとし、温度は「中身の O/F」の燃焼温度 `T_ad` で決めます（`gg_state()`。流れ込んだ酸素は、中に残っている燃料過剰のガスとも燃えます）。
起動前の GG は 1 気圧の窒素（パージガス）で満たされているとし、窒素が残っているあいだは、その分だけ温度が下がります。
弁を開く順番がずれたときの起動時の温度のピークは、この起動前のガスの仮定に大きく左右されます（停止時のピークは左右されません）。
温度は圧力一定で混ぜたときの値で、圧力が急に変わるときの圧縮・膨張による温度変化（圧力が上がるときは高め、下がるときは低めになります）は入れていません。壁への熱の逃げも入れていません。
起動前の窒素の量は `GGEngine(gg_n2_fill=2.0)` のように変えられます（1 気圧で満たした量に対する倍率。0 以下はエラー）。
タービン出口の圧力（0.2 MPa ＝ 約 1.97 気圧）を超える分は、弁が開く前にタービンのノズルから抜けるので、約 1.97 より大きくしても結果は変わりません。
主燃焼室は、ガスが約 1.4 ms で入れ替わり、中身の O/F が流れ込む推進剤の O/F とほぼ同じなので、流れ込む推進剤の燃焼温度に、ガスの入れ替わり時間で追いつく式にしています。
真空推力には、主ノズルとタービン排気の出口の両方で背圧がなくなる分を足しています（`design_point()["F_vac"]`）。

燃焼ガスの表（燃焼温度 `T_AD`、特性排気速度 `CSTAR`、真空推力係数 `CF_VAC`）は、O/F 0.8 以上が化学平衡計算（RocketCEA、9.7 MPa）、
燃料過剰の GG の領域（O/F 0.30〜0.52）が Huzel & Huang "Design of Liquid Propellant Rocket Engines"（NASA SP-125）Table 6-4 の代表値です（そのあいだと 0.30 より下は、点をつないで補っています）。
GG のガスの性質（比熱・比熱比・ガス定数。`gg_gas()`）も同じ分け方で、O/F 0.52 までが教科書の値、0.8 以上が化学平衡計算（GG の圧力 7.76 MPa）の分子量と比熱比です。
教科書の値（分子量 23〜35）と平衡計算の値（O/F 0.8 で分子量 14）には開きがあり、そのあいだは直線でつないでいます。
O/F 0.8 以上の比熱は、比熱比とガス定数から cp = γR/(γ − 1) で計算した値で、平衡計算の比熱そのものではありません
（O/F 3 で約 2,880 J/kg/K。平衡計算の比熱は、反応が進むとみるか組成が変わらないとみるかで 約 7,070 / 約 1,970 J/kg/K と大きく違います）。
窒素で薄まったガスの温度、つまり LOX 弁が先に開いたときの起動時のピークは、この比熱の仮定でも数百 K 変わります。
燃焼温度の表は、主燃焼室にも GG にも 9.7 MPa の値を使っています。圧力が低いと、高温側は解離で下がります（O/F 3 では、4 MPa で約 140 K、1 MPa で約 340 K 低い）。
弁の順番がずれて GG の温度がピークになるとき、GG の圧力は設計値（7.76 MPa）よりずっと低いので、ピークの温度は高めに出ています。
`cea_tables.py` を実行すると、平衡計算の部分（`T_AD`、`CSTAR`、`CF_VAC`、`_M_CEA`、`_GAM_CEA`）を作り直せます（`pip install rocketcea` が必要）。
RocketCEA が入っている環境で pytest を実行すると、埋め込んだ表が平衡計算の結果と一致することも確認します（入っていなければ、そのテストだけスキップします）。

このモデルに入っていないもの: 着火の遅れ、配管の慣性と圧力波（水撃）、燃焼の遅れ（チャギング）、再生冷却の熱、弁より下流に残る推進剤、
タンク圧の変化、機体の加速度やタンクの液面によるポンプ入口圧の変化とキャビテーション。

主ノズルの推力は海面での値です。燃焼圧が約 5.7 MPa より下がると、出口の圧力が大気圧の 0.4 倍（`K_SEP`。Summerfield の目安）を下回るので、
流れが壁からはがれるとして計算します（`GGEngine.main_thrust()`。はがれる断面でノズルを切ったものとして扱います）。
はがれる圧力の目安には 0.3〜0.4 の幅があり、推力を絞ったときの推力や停止のあとの力積は、この値で数 % 変わります（停止の途中、燃焼圧が 1 MPa より下がってからの推力そのものは、もっと大きく変わります）。
設計計算 `design_point()` は、流れが出口まで壁に沿っている場合の式を使うので、設計の燃焼圧がこれより低いとエラーにします。
タービン出口圧は 0.2 MPa で一定としています（実機では排気の出口の大きさが決まっているので、流量とともに変わります）。
ポンプとタービンのマップの形は典型的なものを仮定しています（タービンの効率は速度比の放物線 `turbine_eta_ratio()` で、設計点で最大）。

グラフの日本語には日本語フォントが要ります（Noto Sans CJK JP、Yu Gothic、Meiryo、Hiragino Sans などを自動で選びます。Linux なら `fonts-noto-cjk`）。
記事の図は numpy 2.4・matplotlib 3.10 で作りました。ほかのバージョンでも数値は同じですが、図の細部（文字の位置など）が少し変わることがあります。

## ライセンス
MIT。改造・再配布自由。記事へのリンクを添えてもらえると嬉しいです。
