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
    xlabel=r'$\Theta$ / °',
    ylabel=r'Intensität$',
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
    ylabel=r'Intensität$',
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
ax.set(
    xlabel=r'x / $\unit{\milli\meter}$',
    ylabel=r'Intensität$',
)
plt.savefig('build/x_scan.pdf')
plt.clf()

# Rockingscan

theta, Intensity = np.genfromtxt('tables/rockingscan_1_3.txt', unpack=True)

fig, ax = plt.subplots(1, 2, figsize=(8,4), layout='constrained', sharey=True)
ax = np.ravel(ax)

ax[0].scatter(theta, Intensity, marker='x', c='r', label='Messdaten')
ax[0].set(
    title='Erster Rockingscan',
    xlabel=r'\Theta / $\unit{\degree}$',
    ylabel=r'Intensität$',
)

ax[0].axvline(theta[np.argmax(Intensity)] + 0.35, color='g', linestyle='--')
ax[0].axvline(theta[np.argmax(Intensity)] - 0.35, color='g', linestyle='--', label='Geometriewinkel')

theta, Intensity = np.genfromtxt('tables/rockingscan_1_4.txt', unpack=True)
ax[1].scatter(theta, Intensity, marker='x', c='r', label='Messdaten')
ax[1].set(
    title='Zweiter Rockingscan zur Feinjustage',
    xlabel=r'\Theta / $\unit{\degree}$',
)

for i, axis in enumerate(ax):
    axis.grid('::', alpha=0.5)
    axis.legend(loc='best')

plt.savefig('build/rockingscan.pdf')
plt.clf()

##############################################
################## Messung ###################
##############################################

# Plot 1

def geometriefaktor(theta):
    mask = theta > 0.401 / 2 # geometriewinkel_theo = 0.401, / 2 weil 2theta auf x-achse
    result = np.zeros_like(theta, dtype=float)

    result[mask] = 1
    result[~mask] = 20 * np.sin(np.deg2rad(theta[~mask])) / d0
    return result

theta, Intensity  = np.genfromtxt('tables/Messung1.txt', unpack=True)
theta_diffuse, Intensity_diffuse  = np.genfromtxt('tables/Messung2.txt', unpack=True)

I_0 = np.max(Intensity - Intensity_diffuse) * 5         
R_korrektur = (Intensity - Intensity_diffuse) / I_0
    
def fresnelreflectivity2(theta):
    alpha_c = 0.223
    return (alpha_c / (2 * theta))**4

fig, ax = plt.subplots(layout='constrained')
ax.plot(theta[1:],
        R_korrektur[1:],
        linestyle='-',
        linewidth=1,
        label = r'Messwerte $R$',
        )
ax.plot(theta[theta > 0.1],
        fresnelreflectivity2(theta[theta > 0.1]),
        linestyle=':',
        linewidth=1,
        c='r',
        label = r'Näherung des Fresnelkoeffizienten',
        )
R_korrektur =  R_korrektur / geometriefaktor(theta)
ax.plot(theta[1:], R_korrektur[1:], ls='--', c='lightblue', label='Korrigiert durch Geometriefaktor',lw=1)

minima_indices = argrelextrema(R_korrektur, np.less, order=5)
minima_theta = theta[1:][minima_indices]
minima_R = R_korrektur[minima_indices]

ax.scatter(minima_theta, minima_R, c='black', label='Minima', s=20, marker='x', zorder=5)

ax.axvline(3 * 0.223, color='g', linestyle='--', label=r'$\alpha_i > 3 \cdot \alpha_C$')
ax.set(
    xlabel = r'$2 \theta \, / \, °$',
    ylabel = r'$R$',
    yscale = 'log',
)
ax.legend(loc = 'best')
ax.grid('::')
plt.savefig('build/messung1.pdf')
plt.clf()

# Plot 2

lam = 1.541* 10**-10
diff = np.zeros(len(minima_theta) -1)
for i in np.arange(len(minima_theta) - 1):
    diff[i] = minima_theta[i+1] - minima_theta[i]
    diff[i] = np.deg2rad(diff[i])

diff = diff[diff < np.quantile(diff, 0.95)]
diff = ufloat(np.mean(diff), np.std(diff))

d = lam / (2 * diff)
print(f'{d=:}')

# d = 8.62*10**(-8)
d = 8.45*10**(-8)
# delta_2 = 1.3 * 10**(-6)
delta_2 = 1.8* 10**(-6)
# delta_3 = 5.9 * 10**(-6)
delta_3 = 6.25 * 10**(-6)
sigma_1 = 8 * 10**(-10)
sigma_2 = 6.5 * 10**(-10)

p0 = ([d, delta_2, delta_3, sigma_1, sigma_2])

def parratt_algorithm(theta, d, delta_2, delta_3, sigma_1, sigma_2):
    k = 2*np.pi/lam

    n_1 = 1
    n_2 = 1 - delta_2 - 1j * delta_2 / 200
    n_3 = 1 - delta_3 - 1j * delta_3 / 40
    
    k_z1 = k * np.sqrt(n_1**2 - np.cos(np.deg2rad(theta))**2)
    k_z2 = k * np.sqrt(n_2**2 - np.cos(np.deg2rad(theta))**2)
    k_z3 = k * np.sqrt(n_3**2 - np.cos(np.deg2rad(theta))**2)

    r_1 = np.exp(-2 * k_z1 * k_z2 * sigma_1**2) * (k_z1 - k_z2) / (k_z1 + k_z2)
    r_2 = np.exp(-2 * k_z2 * k_z3 * sigma_2**2) * (k_z2 - k_z3) / (k_z2 + k_z3)

    X_2 = np.exp(-2j*k_z2 * d) * r_2
    X_1 = (r_1 + X_2) / (1 + r_1 * X_2)

    R = np.abs(X_1)**2

    return R

fit_mask = ((theta > 0.3) & (theta < 1.3))
params_parratt, cov_parratt = curve_fit(parratt_algorithm,
                                        theta[fit_mask],
                                        R_korrektur[fit_mask],
                                        p0=p0,
                                        bounds=([10**-9, 10**-7, 10**-7, 10**-10, 10**-10], [10**-7, 10**-5, 10**-5, 10**-9, 10**-9]),
                                        maxfev=100000)

alpha_c_PS = np.rad2deg(np.sqrt(2*delta_2))
alpha_c_Si = np.rad2deg(np.sqrt(2*delta_3))

fig, ax = plt.subplots(layout='constrained')
ax.plot(theta,
        R_korrektur,
        linestyle='-',
        linewidth=1,
        label = r'Messwerte $R_\text{korrigiert}$',
        )
ax.plot(theta,
        parratt_algorithm(theta, *params_parratt),
        linestyle='--',
        linewidth=1,
        label = r'Parratt Alg. Fit',
        )
ax.axvline(alpha_c_PS,
            color='gray',
            linestyle='--',
            linewidth=1,
            label=r'$\alpha_{\text{c,PS}} = $' + f'{alpha_c_PS:=.3f}°',
            )
ax.axvline(alpha_c_Si,
            color='gray',
            linestyle='--',
            linewidth=1,
            label=r'$\alpha_{\text{c,Si}} = $' + f'{alpha_c_Si:=.3f}°',
            )
ax.set(
    xlabel = r'$2 \theta \, / \, °$',
    ylabel = r'$R$',
    yscale='log',
)
ax.legend(loc = 'best')
ax.grid('::', alpha=0.5)
plt.savefig('build/messung2.pdf')