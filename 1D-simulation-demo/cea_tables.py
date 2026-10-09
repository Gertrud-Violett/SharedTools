"""
cea_tables.py — gg_rocket.py に埋め込んだ「燃焼ガスの特性表」を RocketCEA で作り直すスクリプト（任意）
RocketCEA（NASA CEA の Python ラッパー）が必要:  pip install rocketcea
  使い方（Windows 編）: https://makkiblog.com/rocketcea_intro/
出力: 酸燃比 O/F ごとの 断熱火炎温度 T [K]・特性排気速度 c* [m/s]・分子量 M [g/mol]、真空推力係数 CF_vac（LOX/RP-1、燃焼圧 9.7 MPa、開口比 16、化学平衡）と、
      GG の圧力（燃焼圧の 0.8 倍）での分子量・比熱比（GG のガスの性質の表の、O/F 0.8 以上の部分）
"""
from rocketcea.cea_obj import CEA_Obj

PSI_PER_BAR = 14.5037738
PC_BAR, EPS = 97.0, 16.0                      # Merlin 1D 級: 燃焼圧 9.7 MPa、ノズル開口比 16
GG_PRESSURE_RATIO = 0.8                       # GG の圧力 ÷ 燃焼圧（gg_rocket.DESIGN の k_gg と同じ）
OFS = [0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.2, 2.36, 2.5, 2.8, 3.0, 3.4, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 100.0]   # gg_rocket.py の _OF_CEA と同じ点

if __name__ == "__main__":
    cea = CEA_Obj(oxName="LOX", fuelName="RP-1")
    pc = PC_BAR * PSI_PER_BAR
    rows = []
    for of in OFS:
        ivac, cstar_ft, tc_R = cea.get_IvacCstrTc(Pc=pc, MR=of, eps=EPS)
        mw, gam = cea.get_Chamber_MolWt_gamma(Pc=pc, MR=of, eps=EPS)
        rows.append(tuple(float(x) for x in (of, tc_R / 1.8, cstar_ft * 0.3048, mw, gam, ivac)))
    print("O/F     T[K]   c*[m/s]   M[g/mol]  gamma  Ivac[s]")
    for r in rows:
        print(f"{r[0]:5.2f} {r[1]:8.0f} {r[2]:8.0f} {r[3]:9.2f} {r[4]:7.3f} {r[5]:7.1f}")
    print("\nOF   =", [r[0] for r in rows])
    print("T_AD =", [round(r[1]) for r in rows])
    print("CSTAR=", [round(r[2]) for r in rows])
    print("MW   =", [round(r[3], 2) for r in rows])
    CF_OFS = [1.5, 2.0, 2.2, 2.36, 2.5, 2.8, 3.4]                     # gg_rocket.py の CF_VAC と同じ点
    cf = []
    for of in CF_OFS:
        ivac, cstar_ft, _ = cea.get_IvacCstrTc(Pc=pc, MR=of, eps=EPS)
        cf.append(round(float(ivac) * 9.80665 / (float(cstar_ft) * 0.3048), 4))   # CF_vac = Isp_vac × g0 ÷ c*
    print("CF_VAC（O/F", CF_OFS, "）=", cf)
    pgg = GG_PRESSURE_RATIO * pc
    gg = [cea.get_Chamber_MolWt_gamma(Pc=pgg, MR=of, eps=EPS) for of in OFS]              # gg_rocket.py の _M_CEA、_GAM_CEA
    print(f"\nGG の圧力（{GG_PRESSURE_RATIO * PC_BAR / 10:.2f} MPa）での分子量と比熱比（O/F は上と同じ点）")
    print("_M_CEA   =", [round(float(m), 2) for m, _ in gg])
    print("_GAM_CEA =", [round(float(g), 3) for _, g in gg])
    ivac, cstar_ft, tc_R = cea.get_IvacCstrTc(Pc=pc, MR=2.36, eps=EPS)
    print(f"\n設計点 O/F 2.36: Tc = {tc_R/1.8:.0f} K, c* = {cstar_ft*0.3048:.1f} m/s, 真空推力係数 CF_vac = {ivac*9.80665/(cstar_ft*0.3048):.4f}, 真空比推力 = {ivac:.1f} s")
    # 参考: 燃料過剰の GG を化学平衡で計算すると、実機（O/F 0.32 で約 900 K）より高い温度になる。gg_rocket.py が GG の領域に実測ベースの表を使う理由
    print(f"GG（O/F 0.324、{GG_PRESSURE_RATIO * PC_BAR / 10:.2f} MPa）の平衡温度 = {cea.get_Tcomb(Pc=pgg, MR=0.324) / 1.8:.0f} K、"
          f"ノズル出口の圧力比 Pc/Pe（開口比 {EPS:.0f}）= {cea.get_PcOvPe(Pc=pc, MR=2.36, eps=EPS):.1f}、燃焼ガスの音速 = {cea.get_SonicVelocities(Pc=pc, MR=2.36, eps=EPS)[0] * 0.3048:.0f} m/s")
    # 参考: gg_rocket.py の燃焼温度の表は 9.7 MPa の値だけ。圧力が低いと、高温側は解離が進んで温度が下がる。弁の順番がずれて GG の中が
    # O/F 3 付近を通るとき、GG の圧力は 1〜4 MPa 程度まで下がっているので、そのときの温度は表の値より低い
    print("O/F 3.0 の燃焼温度: " + "、".join(f"{p:g} MPa で {cea.get_Tcomb(Pc=p * 10 * PSI_PER_BAR, MR=3.0) / 1.8:.0f} K" for p in (1.0, 2.0, 4.0, PC_BAR / 10)))
    # 参考: O/F 0.8 以上の GG のガスの比熱は、gg_rocket.py では比熱比とガス定数から cp = γR/(γ − 1) で計算している。平衡計算の比熱は、
    # 温度が変わるときに反応が進むとみるか（平衡）、組成が変わらないとみるか（凍結）で大きく違い、上の値はそのあいだにある
    BTU_LBM_R = 4186.8                                                # 比熱の換算 [BTU/lbm/°R] → [J/kg/K]
    print(f"O/F 3.0・{GG_PRESSURE_RATIO * PC_BAR / 10:.2f} MPa の比熱: 平衡 {cea.get_Chamber_Cp(Pc=pgg, MR=3.0, eps=EPS, frozen=0) * BTU_LBM_R:.0f} J/kg/K、"
          f"凍結 {cea.get_Chamber_Cp(Pc=pgg, MR=3.0, eps=EPS, frozen=1) * BTU_LBM_R:.0f} J/kg/K")
