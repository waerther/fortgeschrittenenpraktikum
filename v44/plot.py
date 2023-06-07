import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit                        # Funktionsfit:     popt, pcov = curve_fit(func, xdata, ydata) 
from uncertainties import ufloat                            # Fehler:           fehlerwert =  ulfaot(x, err)
from uncertainties import unumpy as unp 
from uncertainties.unumpy import uarray                     # Array von Fehler: fehlerarray =  uarray(array, errarray)
from uncertainties.unumpy import (nominal_values as noms,   # Wert:             noms(fehlerwert) = x
                                  std_devs as stds)         # Abweichung:       stds(fehlerarray) = errarray

from scipy.stats import norm
from scipy.optimize import fsolve
from scipy.signal import argrelextrema

# Daten einlesen:
dscan_data = pd.read_csv('tables/detektorscan.txt', delim_whitespace="\t")
messung_1 = pd.read_csv('tables/Messung1.txt', delim_whitespace="\t")
messung_2 = pd.read_csv('tables/Messung2.txt', delim_whitespace="\t")
# rockingscan_1_1 = pd.read_csv('tables/rockingscan_1_1.txt', delim_whitespace="\t")
# rockingscan_1_2 = pd.read_csv('tables/rockingscan_1_2.txt', delim_whitespace="\t")
rockingscan_1_3 = pd.read_csv('tables/rockingscan_1_3.txt', delim_whitespace="\t")
rockingscan_1_4 = pd.read_csv('tables/rockingscan_1_4.txt', delim_whitespace="\t")
# rockingscan_2_1 = pd.read_csv('tables/rockingscan_2_1.txt', delim_whitespace="\t")
# rockingscan_2_2 = pd.read_csv('tables/rockingscan_2_2.txt', delim_whitespace="\t")
# rockingscan_3_1 = pd.read_csv('tables/rockingscan_3_1.txt', delim_whitespace="\t")
# rockingscan_3_2 = pd.read_csv('tables/rockingscan_3_2.txt', delim_whitespace="\t")
# rockingscan_4_1 = pd.read_csv('tables/rockingscan_4_1.txt', delim_whitespace="\t")
x_scan_1 = pd.read_csv('tables/x_scan1.txt', delim_whitespace="\t")
z_scan_1 = pd.read_csv('tables/z_scan1.txt', delim_whitespace="\t")
# z_scan_2 = pd.read_csv('tables/z_scan2.txt', delim_whitespace="\t")
# z_scan_3 = pd.read_csv('tables/z_scan3.txt', delim_whitespace="\t")

# Detektorscan

theta, Intensity = np.genfromtxt('tables/detektorscan.txt', unpack=True, skip_header=1)

def gauss(theta, A, mean, scale, b):
    return A * norm.pdf(theta, loc=mean, scale=scale) + b

popt, pcov = curve_fit(gauss, theta, Intensity, p0=[10000, 0, 0.1, 1])
A = ufloat(popt[0], np.sqrt(np.diag(pcov))[0])
mean = ufloat(popt[1], np.sqrt(np.diag(pcov))[1])
scale = ufloat(popt[2], np.sqrt(np.diag(pcov))[2])
b = ufloat(popt[3], np.sqrt(np.diag(pcov))[3])

fig, ax = plt.subplots()
x_fit = np.linspace(np.min(theta), np.max(theta), 1000)
ax.plot(x_fit, gauss(x_fit, *popt), label='Fit')
ax.scatter(theta, Intensity, marker='x', c='r', label='Messdaten')


half_max = gauss(noms(mean), *popt) / 2
roots = fsolve(lambda x: gauss(x, *popt) - half_max, x0=[-0.2,0.2])
for root in roots:
    ax.axvline(root, color='g', linestyle='--')
ax.grid('::', alpha=0.5)
ax.set(
    xlabel=r'$\omega$ / °',
    ylabel=r'Intensity / $\unit\second$',
)
ax.legend(loc='best')
plt.savefig('build/Detektorscan.pdf')
plt.clf()

# Z-Scan

z, Intensity = np.genfromtxt('tables/z_scan1.txt', unpack=True)

fig, ax = plt.subplots()

ax.scatter(z, Intensity, marker='x', c='r', label='Messdaten')
ax.axvline(z[13], color='g', linestyle='--')
ax.axvline(z[20], color='g', linestyle='--', label='Totale Breite des Strahls')

d0 = z[20] - z[13]

ax.set(
    xlabel=r'z / $\unit{\milli\meter}$',
    ylabel=r'Intensity / $\unit\second$',
)
ax.grid('::', alpha=0.5)
ax.legend(loc='best')
plt.savefig('build/z_scan.pdf')
plt.clf()

# X-Scan

x, Intensity = np.genfromtxt('tables/x_scan1.txt', unpack=True)
fig, ax = plt.subplots()

ax.scatter(x, Intensity, marker='x', c='r', label='Messdaten')
ax.grid('::', alpha=0.5)
ax.legend(loc='best')
plt.savefig('build/x_scan.pdf')
plt.clf()

# Rockingscan

theta, Intensity = np.genfromtxt('tables/rockingscan_1_3.txt', unpack=True)

fig, ax = plt.subplots(1, 2, figsize=(8,4), layout='constrained', sharey=True)
ax = np.ravel(ax)

ax[0].scatter(theta, Intensity, marker='x', c='r', label='Messdaten')
ax[0].set(
    title='Erster Rockingscan',
)

ax[0].axvline(theta[np.argmax(Intensity)] + 0.35, color='g', linestyle='--')
ax[0].axvline(theta[np.argmax(Intensity)] - 0.35, color='g', linestyle='--', label='Geometriewinkel')

theta, Intensity = np.genfromtxt('tables/rockingscan_1_4.txt', unpack=True)
ax[1].scatter(theta, Intensity, marker='x', c='r', label='Messdaten')
ax[1].set(
    title='Zweiter Rockingscan zur Feinjustage',
)

for i, axis in enumerate(ax):
    axis.grid('::', alpha=0.5)
    axis.legend(loc='best')

plt.savefig('build/rockingscan.pdf')
plt.clf()

# Messung

theta, Intensity = np.genfromtxt('tables/Messung1.txt', unpack=True)
theta, Intensity1 = np.genfromtxt('tables/Messung2.txt', unpack=True)

fig, ax = plt.subplots()

def geometriefaktor(theta):
    mask = theta > 0.223
    result = np.zeros_like(theta, dtype=float)

    result[mask] = 1
    result[~mask] = 20 * np.sin(theta[~mask]) / d0
    return result

R = (Intensity - Intensity1)[1:] / (5 * np.max((Intensity - Intensity1)[1:]))
R_korrektur =  R * geometriefaktor(theta[1:])
ax.plot(theta[1:], R, ls='-', c='r', label='Messdaten zur Reflexivität', lw = 0.9, zorder=3)
ax.plot(theta[1:], R_korrektur, ls='--', c='b', label='Korrigiert durch Geometriefaktor',lw=2)
ax.grid('::', alpha=0.5)
ax.legend(loc='best')
ax.set(
    yscale='log',
    xlabel=r'$\theta$ / °',
    ylabel='R',
)

x0, x1 = 10, 30  # Define the interval [x0, x1]
minima_indices = argrelextrema(R_korrektur, np.less, order=5)
minima_theta = theta[1:][minima_indices]
minima_R = R_korrektur[minima_indices]

ax.scatter(minima_theta, minima_R, c='black', label='Minima', s=20, marker='x', zorder=5)
ax.legend(loc='best')
plt.savefig('build/messung1.pdf')
plt.clf()