import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from numpy import sqrt
import pandas as pd
import scipy.constants as const
from scipy.optimize import curve_fit                        # Funktionsfit:     popt, pcov = curve_fit(func, xdata, ydata) 
from uncertainties import ufloat                            # Fehler:           fehlerwert =  ulfaot(x, err)
from uncertainties import unumpy as unp 
from uncertainties.unumpy import uarray                     # Array von Fehler: fehlerarray =  uarray(array, errarray)
from uncertainties.unumpy import (nominal_values as noms,   # Wert:             noms(fehlerwert) = x
                                  std_devs as stds)         # Abweichung:       stds(fehlerarray) = errarray

# Plot 1:

md = pd.read_csv('tables/md.csv')
np.savetxt('md.txt', md.values, fmt='%.2f')
f, I_1, I_1h, I_2, I_2h = np.genfromtxt('md.txt', unpack=True, skip_header=0)

# Helmholtz
def B(I, N, A):
    mu_0 = const.mu_0
    return mu_0 * 8 * I * N / (np.sqrt(125) * A)

A_S = 0.1639
A_H = 0.1579
N_S = 11
N_H = 154

B1 = B(I_1, N_S, A_S)
B1h = B(I_1h, N_H, A_H)
B2 = B(I_2, N_S, A_S)
B2h = B(I_2h, N_H, A_H)

B1 += B1h
B2 += B2h   # 1e-3 * Tesla

# Ausgleichsfunktion
def g(x, a, b):
    return a * x + b

para, pcov = curve_fit(g, f, B1)    # 1e-9 Tesla
pcov = np.sqrt(np.diag(pcov))
a1, b1 = para
fa1, fb1 = pcov 

ua1 = ufloat(a1, fa1) 
ub1 = ufloat(b1, fb1) 

print('a85 = (%f ± %f)' % (noms(ua1), stds(ua1)), 'T*10^-3 /Hz^-1')
print('b85 = (%f ± %f)' % (noms(ub1), stds(ub1)), 'T*10^-3')

para, pcov = curve_fit(g, f, B2)    # 1e-9 Tesla
pcov = np.sqrt(np.diag(pcov))
a2, b2 = para
fa2, fb2 = pcov 

ua2 = ufloat(a2, fa2) 
ub2 = ufloat(b2, fb2) 

print('a87 = (%f ± %f)' % (noms(ua2), stds(ua2)), 'T*10^-3 /Hz^-1')
print('b87 = (%f ± %f)' % (noms(ub2), stds(ub2)), 'T*10^-3')

# plt.figure(figsize=(9, 6)) 

plt.plot(f, B1, 'xr', markersize=6 , label = 'Messdaten Rb85', alpha=1)
plt.plot(f, B2, 'xb', markersize=6 , label = 'Messdaten Rb87', alpha=1)

xx = np.linspace(-50, 1050, 10**4)

plt.plot(xx, g(xx, a1, b1), '-r', linewidth = 1, label = 'Regression Rb85', alpha = 0.5)
plt.plot(xx, g(xx, a2, b2), '-b', linewidth = 1, label = 'Regression Rb87', alpha = 0.5)

plt.xlabel(r'$f \, / \, \mathrm{kHz}$')
plt.ylabel(r'$B \, / \, \mathrm{mT}$')
plt.legend(loc="best")                  # legend position
plt.grid(True)                          # grid style
plt.xlim(-50, 1050)                   # limitation of visible scale in plot

c1 = g(0, ua1, ub1)
c2 = g(0, ua2, ub2)

c = np.mean([c1, c2]); print('mittlere Horizontalkomponente:', c, 'mT')

plt.savefig("build/geraden.pdf")
plt.clf()
