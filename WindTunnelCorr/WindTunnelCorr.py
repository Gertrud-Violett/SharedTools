#====================================================
#Wind Tunnel Spec Calculator 2022 MIT License makkiblog.com

import numpy as np
import sympy as sy
import sys
import matplotlib.pyplot as plt
import pandas as pd
import toml
#mpl.rcParams['agg.path.chunksize'] = 100000

#Setup (Visual)
import seaborn as sns
sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 1.2})
sns.set_style('whitegrid')
plt.rcParams["figure.figsize"] = (16,9)
plt.rcParams['figure.facecolor'] = 'white'


#Read csv and TOML setting files====================
setting = "./TunnelSpec.toml"
resfile = "./DataResult.csv"
CD_measured = 0.341
with open(setting) as inputf:
    WINT = toml.load(inputf) #setting file
Cd_his = np.loadtxt(resfile, delimiter=',', skiprows=1) #Cd time series

H_N = WINT['WINDTUNNEL']['Nozzle']['Height']
W_N = WINT['WINDTUNNEL']['Nozzle']['Width']
dpdx_N = WINT['WINDTUNNEL']['Nozzle']['dpdx']
CR_N = WINT['WINDTUNNEL']['Nozzle']['ContracRatio']

H_C = WINT['WINDTUNNEL']['Collector']['Height']
W_C = WINT['WINDTUNNEL']['Collector']['Width']
dpdx_C = WINT['WINDTUNNEL']['Collector']['dpdx']

xm = WINT['WINDTUNNEL']['TestSection']['xm']
L = WINT['WINDTUNNEL']['TestSection']['SecLength']

M = 0.117

V_M = WINT['VEHICLE']['Spec']['Volume']
L_M = WINT['VEHICLE']['Spec']['Length']
A_M = WINT['VEHICLE']['Spec']['FrontArea']
H_M = WINT['VEHICLE']['Spec']['Height']
W_M = WINT['VEHICLE']['Spec']['Width']
yaw_M = WINT['VEHICLE']['Spec']['yaw']
casename = WINT['CALC_SETTING']['Casename']
type = WINT['CALC_SETTING']['TunnelType']



#MAIN Calculation
def WTcalc(type,C_Dm,H_N,H_C,W_N,xm,dpdx_N,dpdx_C,M):
    H = H_N
    W = W_N
    A_N = H_N*W_N
    A_C = H_C*W_C
    
    #open
    if type=="open":
        
        #Exp Constr
        xs = xm-L_M/2+(A_M/2/np.pi)**0.5
        tau = -0.032/(1-M**2)**1.5*(2*H/W+W/2/H)**0.252

        R_N = (2*A_N/np.pi)**0.5
        epQ = (A_M/2/A_N)*(1-xs/(xs**2+R_N**2)**0.5)
        eps=tau*(V_M/L_M)**0.5*A_M/(A_N/(1+epQ))**1.5
        
        #blockage
        R_C = (2*A_C/np.pi)**0.5
        epw=(A_M/A_C)*(C_Dm/4 + 0.41)
        epN = epQ * R_N**3/(xm**2+R_N**2)**1.5
        epC = (epw*R_C**3)/((L-xm)**2 + R_C**2)**1.5 

        #floating
        G = dpdx_N+dpdx_C
        dC_DHB = (1.75/A_M)*(V_M/2)*G
        
        epT=(1+eps+epN+epC)**2
        C_dor = (C_Dm + dC_DHB)/epT
        #print(H_N,xm,dpdx_N)
        return(C_dor)

    #closed
    elif type=="closed":
        
        K3 = 1.0
        taudsh = 0.406*(2*H/W+W/2/H)
        eta=0.41
        epT = (1+K3*taudsh*(2*A_M*2*V_M)/((L_M*2*V_M)**0.5*(2*A_N)**1.5)*A_M/A_N*(1/4*C_Dm+eta))**2

        #floating
        G = dpdx_N+dpdx_C
        dC_DHB = (1.75/A_M)*(V_M/2)*G
        
        C_dor = (C_Dm + dC_DHB)/epT
        return(C_dor)

    else:
        print("input error: syntax is (open/close)")

"""
#Single point
CDm=WTcalc(type,CD_measured,H_N,H_C,W_N,xm,dpdx_N,dpdx_C,M)
print(CDm)
"""

#Sensitivity Analysis
num=30
H_N_arr = np.linspace(H_N*0.2,H_N*2.0,num)
H_C_arr = np.linspace(H_C*0.2,H_C*2.0,num)
W_N_arr = np.linspace(W_N*0.2,W_N*2.0,num)
dpdx_N_arr = np.linspace(dpdx_N*-2.0,dpdx_N*5.0,num)
dpdx_C_arr = np.linspace(dpdx_C*-2.0,dpdx_C*5.0,num)
M_arr = np.linspace(M*0.1,M*8.0,num)
xm_arr = np.linspace(xm*0.2,xm*2.0,num)


line = 0
CD_arr1 = np.empty(num)
while line < num:
    CD = WTcalc(type,CD_measured,H_N_arr[line],H_C,W_N,xm,dpdx_N,dpdx_C,M) #input raw Cd value
    CD_arr1[line] = CD
    line+=1

line = 0
CD_arr2 = np.empty(num)
while line < num:
    CD = WTcalc(type,CD_measured,H_N,H_C_arr[line],W_N,xm,dpdx_N,dpdx_C,M)
    CD_arr2[line] = CD
    line+=1

line = 0
CD_arr3 = np.empty(num)
while line < num:
    CD = WTcalc(type,CD_measured,H_N,H_C,W_N,xm,dpdx_N_arr[line],dpdx_C,M)
    CD_arr3[line] = CD
    line+=1

line = 0
CD_arr4 = np.empty(num)
while line < num:
    CD = WTcalc(type,CD_measured,H_N,H_C,W_N,xm,dpdx_N,dpdx_C_arr[line],M)
    CD_arr4[line] = CD
    line+=1

line = 0
CD_arr5 = np.empty(num)
while line < num:
    CD = WTcalc(type,CD_measured,H_N,H_C,W_N,xm_arr[line],dpdx_N,dpdx_C,M)
    CD_arr5[line] = CD
    line+=1

line = 0
CD_arr6 = np.empty(num)
while line < num:
    CD = WTcalc(type,CD_measured,H_N,H_C,W_N,xm,dpdx_N,dpdx_C,M_arr[line])
    CD_arr6[line] = CD
    line+=1

line = 0
CD_arr7 = np.empty(num)
while line < num:
    CD = WTcalc(type,CD_measured,H_N,H_C,W_N_arr[line],xm,dpdx_N,dpdx_C,M)
    CD_arr7[line] = CD
    line+=1


plt.clf()
fig1,ax1=plt.subplots(2,2)
ax1[0,0].set_xlabel('Nozzle, Collector Height[m]')
ax1[0,0].set_ylabel('Corrected CD')
ax1[0,0].plot(H_N_arr,CD_arr1,label="Nozzle Height",color="red")
ax1[0,0].plot(H_C_arr,CD_arr2,label="Collector Height",color="green")

ax1[0,1].set_xlabel('dpdx[kPa/m]')
ax1[0,1].set_ylabel('Corrected CD')
ax1[0,1].plot(dpdx_N_arr,CD_arr3,label="dpdx_Nozzle")
ax1[0,1].plot(dpdx_C_arr,CD_arr4,label="dpdx_Collector")

ax1[1,0].set_xlabel('xm[m]')
ax1[1,0].set_ylabel('Corrected CD')
ax1[1,0].plot(xm_arr,CD_arr5,label="xm",color="purple")


ax1[1,1].set_xlabel('Mach[-]')
ax1[1,1].set_ylabel('Corrected CD')
ax1[1,1].plot(M_arr,CD_arr6,label="Mach",color="black")
fig1.legend(bbox_to_anchor=(0.6,0.4),loc='upper left', fontsize=12)

plt.suptitle(casename)
fig1.savefig(casename+'_plot.jpg', dpi=300)
plt.show()

plt.rcParams["figure.figsize"] = (9,6)
plt.clf()
plt.xlabel('Nozzle, Collector Height[m]')
plt.ylabel('Corrected CD')
plt.plot(H_N_arr,CD_arr1,label="Nozzle Height",color="red")
plt.plot(H_C_arr,CD_arr2,label="Collector Height",color="green")
plt.legend(bbox_to_anchor=(0.6,0.4),loc='upper left', fontsize=12)
plt.savefig(casename+'_plot_HNHC.jpg', dpi=300)

plt.clf()
plt.xlabel('dpdx[kPa/m]')
plt.ylabel('Corrected CD')
plt.plot(dpdx_N_arr,CD_arr3,label="dpdx_Nozzle")
plt.plot(dpdx_C_arr,CD_arr4,label="dpdx_Collector")
plt.legend(bbox_to_anchor=(0.6,0.4),loc='upper left', fontsize=12)
plt.savefig(casename+'_plot_dpdx.jpg', dpi=300)

plt.clf()
plt.xlabel('xm[m]')
plt.ylabel('Corrected CD')
plt.plot(xm_arr,CD_arr5,label="xm",color="purple")
plt.legend(bbox_to_anchor=(0.6,0.4),loc='upper left', fontsize=12)
plt.savefig(casename+'_plot_xm.jpg', dpi=300)

plt.clf()
plt.xlabel('Mach[-]')
plt.ylabel('Corrected CD')
plt.plot(M_arr,CD_arr6,label="Mach",color="black")
plt.legend(bbox_to_anchor=(0.6,0.4),loc='upper left', fontsize=12)
plt.savefig(casename+'_plot_Mach.jpg', dpi=300)

plt.clf()
plt.xlabel('W_N[m]')
plt.ylabel('Corrected CD')
plt.plot(M_arr,CD_arr7,label="Nozzle Width",color="red")
plt.legend(bbox_to_anchor=(0.6,0.4),loc='upper left', fontsize=12)
plt.savefig(casename+'_plot_WN.jpg', dpi=300)
