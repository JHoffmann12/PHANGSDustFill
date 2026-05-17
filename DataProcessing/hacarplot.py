#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
from astropy.io import fits
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

def thermal(cs, M):
    mass = (10**M)*1.989e33
    speed = cs*100000
    L = (mass)*6.67e-8/(2*speed**2)
    return np.log10(L/3.086e18)

def lscale(cs,L, L0):
    speed = cs*100000
    l = (10**L)*3.086e18
    M = ((2*speed**2)/6.67e-8)*l*(1+((10**L)/L0))
    return np.log10(M/1.989e33)

def fBcorr(cs, va, L, L0):
    speed = cs*100000
    alfven = va*100000
    l = (10**L)*3.086e18
    M = ((2*speed**2)/6.67e-8)*l*(1+((10**L)/L0)+(alfven/speed)**2)
    return np.log10(M/1.989e33)    

M = np.linspace(-1, 9.0, 200)
L = np.linspace(-2, 4, 200)
cs = [0.2, 2, np.sqrt(0.6**2 + 6**2), 5]
sigtot = [1,4, 8, 12]

molFils = pd.read_csv("./Fils_obsdata_public.csv")
#print(molFils)

molFils.sort_values(by="REF", axis=0, inplace=True)
molFils.set_index(keys=["REF"], drop=False, inplace=True)

simdata = pd.read_csv("./simfils_p12025.csv")
print(simdata)


# In[16]:


dataset = molFils["REF"].unique().tolist()
for set in dataset:
    setarr = molFils[molFils["Ncol_cm-2"] < 1e22]

super = simdata[simdata["Line Mass"]/simdata["Magnetic"] >= 1.0]
sub = simdata[simdata["Line Mass"]/simdata["Magnetic"] < 1.0]

linestyles= ["solid","dashed", "dotted", "dashdot", (0, (3, 5, 1, 5, 1, 5))]

plt.scatter(np.log10(molFils["Mass_Msun"]), np.log10(molFils["L_pc"]), s=7, marker ='*', edgecolors="darkgray", facecolors="white", alpha=0.5)
plt.scatter(sub["log Mass"], sub["log Length"], s=8, marker='x', facecolor="orange", alpha=0.6, label="MWGal, Subcritical Filaments")
plt.scatter(super["log Mass"], super["log Length"], s=6, marker='d', color="orchid", edgecolor="darkorchid", alpha=0.8, label="MWGal, Supercritical Filaments")
plt.scatter([np.log10(7.2e5),np.log10(1.7e6)], [np.log10(1200),np.log10(1200)], marker='x', c="deeppink", s=20, label="Maggie, Syed et al.2022")
plt.scatter([np.log10(2.9e4), np.log10(1e5), np.log10(3e5)], [np.log10(81), np.log10(162), np.log10(431)], marker='x', c="k", s=20, label="Nessie, Goodman et al. 2014")
##THERMAL LINE MASSES
plt.plot(M, thermal(0.2, M), c="lightcoral", alpha=0.8,  linestyle=linestyles[0],label=r"$c_s$= %s" % 0.2)
plt.plot(M, thermal(2.0, M), c="lightcoral", alpha=0.8,  linestyle=linestyles[1],label=r"$c_s$= %s" % 2.0)

##VIRIAL LINE MASSES, L SCALING
plt.plot(lscale(0.2, L,0.5), L, c="black", linestyle="solid", label=r"$\sigma_{tot} \propto L^{0.5}$, $c_s = 0.2$")
#plt.plot(lscale(2.0, L, 0.5), L, c="black", linestyle="dashed", label=r"$\sigma_{tot} \propto L^{0.5}$, $c_s = 2$")
#plt.plot(lscale(0.1, L, 0.5), L, c="black", linestyle="dotted", label=r"$\sigma_{tot} \propto L^{0.5}$, $c_s = 0.1$")
#plt.plot(lscale(0.2, L, 0.1), L, c="royalblue", linestyle="solid", label=r"$\sigma_{tot} \propto L^{0.5}$, $c_s = 0.2$, $L_0$ = 0.1 pc")
plt.plot(lscale(0.2, L, 5.0), L, c="royalblue", linestyle="solid", label=r"$\sigma_{tot} \propto L^{0.5}$, $c_s = 0.2$, $L_0$ = 5.0 pc")
#plt.plot(lscale(0.2, L, 0.05), L, c="blue", linestyle="dotted", label=r"$\sigma_{tot} \propto L^{0.5}$, $c_s = 0.2$, $L_0$ = 0.05 pc")

##MAGNETIC FIELD CORRECTION
plt.plot(fBcorr(0.2, 0.4, L, 5.0), L, c="forestgreen", linestyle="solid", label=r"$\sigma_{tot} \propto L^{0.5}$ w/ B-field correction, $v_A = 0.4$, $L_0$ = 5.0pc")


plt.annotate("Sub-critical", xy=(-0.3,3.5), xytext=(-0.3,3.5), fontsize=10)
plt.annotate("Super-critical", xy=(6.5,-1.5), xytext=(6.5,-1.5), fontsize=10)
plt.legend(loc="upper right", bbox_to_anchor=[1.85,0.93], fontsize=10, ncols=1)
plt.xlabel(r"log(Mass (M$_{\odot}$))", fontsize=11)
plt.ylabel("log(Length (pc))", fontsize=11)
plt.ylim(-2,4.3)
plt.xlim(-1, 8.9)


# In[ ]:




