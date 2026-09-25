# 1D-simulation-demo (onedsim)

Minimal 1D (lumped-parameter) system simulator in pure Python + numpy, with two demo models: a pumped-storage hydro plant and a gas-generator-cycle rocket engine start-up. Companion code for the makkiblog.com article series 【1Dシミュレーション】.

## onedsim — まきブログ版・最小1Dシミュレーションエンジン

【1Dシミュレーション②】（makkiblog.com）で使った、揚水発電所とガスジェネレータサイクルロケットエンジンの
1Dモデル一式です。依存は **numpy と matplotlib だけ**。

```
onedsim.py          エンジン本体（Table1D / Model / simulate / sweep）約130行
pumped_storage.py   揚水発電所モデル（神流川発電所を参考にした数値）
gg_rocket.py        GGサイクルロケットエンジン（Merlin 1D 級）起動過渡モデル
run_studies.py      記事のパラメトリックスタディを一括再現（図 PNG と summary.json を ./fig に出力）
make_schematics.py  系統図の生成
```

## 動かし方

```
pip install -r requirements.txt
python pumped_storage.py     # 7日間の運転を計算して要約を表示（約1秒）
python gg_rocket.py          # 起動 3 秒を計算して定常値を表示（約5秒）
python run_studies.py        # 記事の全図を再生成（約2分）
```

## 自分のモデルを作る

```python
from onedsim import Model, Table1D, simulate

class Tank(Model):
    def initial_state(self):          return {"V": 100.0}          # 状態量
    def derivatives(self, t, s):      return {"V": 2.0 - 0.05 * s["V"]}   # dV/dt
    def outputs(self, t, s):          return {"level": s["V"] / 10}  # 記録したい派生量

res = simulate(Tank(), t_end=100, dt=0.1)      # res["t"], res["V"], res["level"] は numpy 配列
```

ポンプの Q-H 曲線のような試験データは `Table1D(x, y)` に入れて `table(q)` で引くだけです。

## ライセンス
MIT。改造・再配布自由。記事へのリンクを添えてもらえると嬉しいです。
