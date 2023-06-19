import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from numpy import sqrt
import pandas as pd
import scipy.constants as const
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit                        # Funktionsfit:     popt, pcov = curve_fit(func, xdata, ydata) 
from uncertainties import ufloat                            # Fehler:           fehlerwert =  ulfaot(x, err)
from uncertainties import unumpy as unp 
from uncertainties.unumpy import uarray                     # Array von Fehler: fehlerarray =  uarray(array, errarray)
from uncertainties.unumpy import (nominal_values as noms,   # Wert:             noms(fehlerwert) = x
                                  std_devs as stds)         # Abweichung:       stds(fehlerarray) = errarray
# Plot 1:

theta, I = np.genfromtxt('tables/detektorscan.txt', unpack=True, skip_header=1)
Imax = np.amax(I)
Inorm = I/Imax                              # Messdaten normieren

plt.plot(theta, I, 'xr', markersize=6 , label = 'Messdaten', alpha=0.5)

n = len(theta)                              # Anzahl Daten
mean = sum(theta*Inorm)/n                   # note this correction
sigma0 = np.sqrt(sum(Inorm*(theta - mean)**2))

# Regression nach Gaußverteilung
def g(theta, a, b, mu):
    return a*np.exp(-(theta-mu)**2/(b))     # b = 2*sigma**2

para, pcov = curve_fit(g, theta, Inorm)#, p0=[sigma0,mean,1,1])
a, b, mu = para                             # a = Amplitude
pcov = np.sqrt(np.diag(pcov))
fmu, fa, fb = pcov
ua = ufloat(a, fa) 
ub = ufloat(b, fb)
umu = ufloat(mu, fmu)
print('a =', ua)
print('b =', ub)
print('mu =', umu)

xx = np.linspace(-0.4, 0.4, 10**4)
plt.plot(xx, g(xx, *para)*Imax, '-b', linewidth = 1, label = 'Ausgleichsfunktion', alpha=0.5)

gauss = g(xx, *para)*Imax
peak, _ = find_peaks(gauss)
peak = peak[0]
print('I_max =', gauss[peak])

a_halb = Imax * a/2

# Funktion zur Bestimmung der Halbwertpunkte
def find_nearest(array, value):
    idx = np.argmin(np.abs(array - value))
    return idx, array[idx]

# Index und Wert des linken Halbwertpunkts
idx_l, l = find_nearest(gauss[:peak], a_halb)
# Index und Wert des rechten Halbwertpunkts
idx_r, r = find_nearest(gauss[peak:], a_halb)
idx_r += peak       # da index um peak verschoben

# Halbwertbreite
fwhm = xx[idx_r] - xx[idx_l]
print('FWHM =', fwhm)

plt.plot(0, gauss[peak], "gx", label = '$I_{max}$')
# plt.axvline(gauss[idx_left], linestyle='--', color='red', label='Halbwertpunkt')
# plt.axvline(gauss[idx_right], linestyle='--', color='red')
plt.hlines(a_halb, xx[idx_l], xx[idx_r], color="C2", label='FWHM')

plt.xlabel(r'$\theta \, / \, \mathrm{°}$')
plt.ylabel(r'$I \, / \,$ Anzahl $\, / \mathrm{s}$')
plt.legend(loc="best")                  # legend position
plt.grid(True)                          # grid style
plt.xlim(-0.32, 0.32)
# plt.ylim(-0.05, 1.05)

plt.savefig('build/plot1.pdf', bbox_inches = "tight")
plt.clf() 
