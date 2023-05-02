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

md = pd.read_csv('tables/leistung.csv')
np.savetxt('tables/leistung.txt', md.values, header='l/cm P/mW', fmt='%.1f')

l, P = np.genfromtxt('tables/leistung.txt', unpack=True, skip_header=1)
# P /= 8

def f(L,r):
    return (1 - L / r)*(1 - L / r)

r_2 = 140
L_2 = np.linspace(0, 2*140,1000)

para, pcov = curve_fit(f, l, P)
pcov = np.sqrt(np.diag(pcov))
a = para
fa = pcov 

plt.plot(L_2, f(L_2-117,a), '-b', linewidth = 1, label='konkav-konkav Regression')
plt.plot(l, P, 'xr', label='Messdaten')

plt.xlabel(r'Resonatorlänge $L \, / \, \mathrm{cm}$')
plt.ylabel(r'Leistung $P \, / \, \mathrm{mW}$')
plt.grid(True)                          # grid style
plt.legend(loc='best')
plt.xlim(0, 2*140)
plt.ylim(-1.5, 10)

plt.savefig('build/plot1.pdf', bbox_inches = "tight")
plt.clf() 



moden = pd.read_csv('tables/moden.csv')
moden = moden.iloc[:, :]
np.savetxt('tables/moden.txt', moden.values, header='x/mm I/mikroA I/mikroA', fmt='%.3f')
x, tem00, tem01 = np.genfromtxt('tables/moden.txt', unpack=True, skip_header=1)

# für den initial guess bei curvefit()
n = len(x)                             # Anzahl der Daten
mean00 = sum(x*tem00)/n                      # Mittelwert
mean01 = sum(x*tem01)/n 
sigma00 = np.sqrt(sum(tem00*(x - mean00)**2))  # Standardabweichung
sigma01 = np.sqrt(sum(tem01*(x - mean01)**2))  # Standardabweichung

# Ausgleichsrechung nach Gaußverteilung
def g00(x,a,x0,b):
    return a*np.exp(-(x-x0)**2/(b))     # b = 2*sigma**2

def g01(x,a,x0,b):
    return a*((x-x0)**2)*np.exp(-(x-x0)**2/(b))     # b = 2*sigma**2


para, pcov = curve_fit(g00, x, tem00, p0=[1,mean00,sigma00])
a, x0, b = para
pcov = np.sqrt(np.diag(pcov))
fa, fx0, fb = pcov
ua = ufloat(a, fa) 
ub = ufloat(b, fb)
unu0 = ufloat(x0, fx0)

xx = np.linspace(-22, 22, 10**4)         # Definitionsbereich

plt.plot(x, tem00, 'xr', markersize=6 , label = 'Messdaten', alpha=0.5)
plt.plot(xx, g00(xx, *para), '-b', linewidth = 1, label = 'Ausgleichsfunktion', alpha=0.5)
plt.xlabel(r'$x \, / \, \mathrm{mm}$')
plt.ylabel(r'$I \, / \, \unit{\micro \ampere}$')
plt.legend(loc="best")                  # legend position
plt.grid(True)                          # grid style
plt.xlim(-22, 22)
# plt.ylim(-0.05, 1.05)

plt.savefig('build/plot2_00.pdf', bbox_inches = "tight")
plt.clf() 



para, pcov = curve_fit(g01, x, tem01, p0=[1,mean01,sigma01])
a, x0, b = para
pcov = np.sqrt(np.diag(pcov))
fa, fx0, fb = pcov
ua = ufloat(a, fa) 
ub = ufloat(b, fb)
unu0 = ufloat(x0, fx0)

xx = np.linspace(-22, 22, 10**4)         # Definitionsbereich

plt.plot(x, tem01, 'xr', markersize=6 , label = 'Messdaten', alpha=0.5)
plt.plot(xx, g01(xx, *para), '-b', linewidth = 1, label = 'Ausgleichsfunktion', alpha=0.5)
plt.xlabel(r'$x \, / \, \mathrm{mm}$')
plt.ylabel(r'$I \, / \, \unit{\micro \ampere}$')
plt.legend(loc="best")                  # legend position
plt.grid(True)                          # grid style
plt.xlim(-22, 22)
# plt.ylim(-0.05, 1.05)

plt.savefig('build/plot2_01.pdf', bbox_inches = "tight")
plt.clf() 



pol = pd.read_csv('tables/polarisation.csv')
np.savetxt('tables/pol.txt', pol.values, header='phi/deg I/mikroA', fmt='%.3f')
phi, I = np.genfromtxt('tables/pol.txt', unpack=True, skip_header=1)

# Daten generieren
theta = np.linspace(0, 2*np.pi, 100)
phi *= 2 *np.pi/360

def f(phi, I0, phi0):
    return I0 * np.cos(phi + phi0)**2

para, pcov = curve_fit(f, phi, I)
a, b = para
pcov = np.sqrt(np.diag(pcov))
fa, fb = pcov
ua = ufloat(a, fa) 
ub = ufloat(b, fb)

xx = np.linspace(0, 360, 10**4)   

# Polarplot erstellen
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='polar')
ax.plot(phi, I, 'xr', label = 'Messdaten')
ax.plot(xx, f(xx, *para), '-b', label = 'Fit', linewidth = 0.5, alpha = 0.5)

plt.savefig('build/plot3.pdf', bbox_inches = "tight")
plt.clf() 