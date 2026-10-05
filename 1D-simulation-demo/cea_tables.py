"""
cea_tables.py — gg_rocket.py に埋め込んだ「燃焼ガスの特性表」を RocketCEA で作り直すスクリプト（任意）
RocketCEA（NASA CEA の Python ラッパー）が必要:  pip install rocketcea
  使い方（Windows 編）: https://makkiblog.com/rocketcea_intro/
出力: 酸燃比 O/F ごとの 断熱火炎温度 T [K]・特性排気速度 c* [m/s]・分子量 M [g/mol]（LOX/RP-1、燃焼圧 9.7 MPa、化学平衡）
"""
from rocketcea.cea_obj import CEA_Obj

PSI_PER_BAR = 14.5037738
PC_BAR, EPS = 97.0, 16.0                      # Merlin 1D 級: 燃焼圧 9.7 MPa、ノズル開口比 16
OFS = [0.6, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.2, 2.36, 2.5, 2.8, 3.0, 3.4, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0]

if __name__ == "__main__":
    cea = CEA_Obj(oxName="LOX", fuelName="RP-1")
    pc = PC_BAR * PSI_PER_BAR
    rows = []
    for of in OFS:
        ivac, cstar_ft, tc_R = cea.get_IvacCstrTc(Pc=pc, MR=of, eps=EPS)
        mw, gam = cea.get_Chamber_MolWt_gamma(Pc=pc, MR=of, eps=EPS)
        rows.append((of, tc_R / 1.8, cstar_ft * 0.3048, mw, gam, ivac))
    print("O/F     T[K]   c*[m/s]   M[g/mol]  gamma  Ivac[s]")
    for r in rows:
        print(f"{r[0]:5.2f} {r[1]:8.0f} {r[2]:8.0f} {r[3]:9.2f} {r[4]:7.3f} {r[5]:7.1f}")
    print("\nOF   =", [r[0] for r in rows])
    print("T_AD =", [round(r[1]) for r in rows])
    print("CSTAR=", [round(r[2]) for r in rows])
    print("MW   =", [round(r[3], 2) for r in rows])
    ivac, cstar_ft, tc_R = cea.get_IvacCstrTc(Pc=pc, MR=2.36, eps=EPS)
    print(f"\n設計点 O/F 2.36: Tc = {tc_R/1.8:.0f} K, c* = {cstar_ft*0.3048:.1f} m/s, 真空推力係数 CF_vac = {ivac*9.80665/(cstar_ft*0.3048):.4f}, 真空比推力 = {ivac:.1f} s")
    # 参考: 燃料過剰の GG を化学平衡で計算すると、実機（O/F 0.32 で約 900 K）より高い温度になる。gg_rocket.py が GG の領域に実測ベースの表を使う理由
    pgg = 0.8 * pc
    print(f"GG（O/F 0.324、{0.8 * PC_BAR / 10:.2f} MPa）の平衡温度 = {cea.get_Tcomb(Pc=pgg, MR=0.324) / 1.8:.0f} K、"
          f"ノズル出口の圧力比 Pc/Pe（開口比 {EPS:.0f}）= {cea.get_PcOvPe(Pc=pc, MR=2.36, eps=EPS):.1f}、燃焼ガスの音速 = {cea.get_SonicVelocities(Pc=pc, MR=2.36, eps=EPS)[0] * 0.3048:.0f} m/s")
