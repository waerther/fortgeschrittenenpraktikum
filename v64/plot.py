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

# Read in the data
kontrast_data = pd.read_csv('tables/kontrast.csv')
glas_data = pd.read_csv('tables/n_glas.csv')
luft_data = pd.read_csv('tables/n_luft.csv')

####################################
# Create tables for the tex file ###
####################################

md = pd.read_csv('tables/kontrast.csv')
md = md.to_numpy()
winkel = md[:,0]
U_max = md[:,1]
U_min = md[:,2]

K = (U_max - U_min) / (U_max + U_min)
K = np.round(K,3)
mdK = np.c_[md,K]
hea = list(['Winkel in °', 'U_{max}', 'U_{min}', 'Kontrast'])
pandas_mdK = pd.DataFrame(mdK, columns=['Winkel', 'Umax', 'Umin', 'Kontrast'])
mdK = pandas_mdK.to_latex(index = False, column_format= "c c c | c", decimal=',', header=hea, label='tab:kontrast', caption='Messwerte zum Kontrast.')
with open('build/kontrast.txt', 'w') as f:
    f.write(mdK)

####################################

md = pd.read_csv('tables/n_glas.csv')
md = md.to_numpy()
T = 10**-3
def func(maxima):
    n = 1 / (1 - 632.8 * 10**(-9) * maxima / (np.deg2rad(20) * np.deg2rad(11) * T))
    return n
hea = list(['Maxima', 'Brechungsindex'])
md = np.c_[md, func(md)]

mittelwert_n = ufloat(np.mean(md[:,1]), np.std(md[:,1]))
pandas_md = pd.DataFrame(md, columns=hea)
pandas_md = pandas_md.append({
    'Maxima' : 'Median',
    'Brechungsindex' : mittelwert_n
}, ignore_index=True)
pandas_md = pandas_md.to_latex(index = False, column_format= "c c", decimal=',', header=hea, label='tab:glas', caption="Messwerte zum Brechungsindex von Glas")
with open('build/n_glas.txt', 'w') as f:
    f.write(pandas_md)
####################################

# md = pd.read_csv('tables/n_luft.csv')
# hea = list(['Maxima', 'Versuch 1', 'Versuch 2', 'Versuch 3'])
# md = md.to_latex(index = False, column_format= "c c c c", decimal=',', header=hea, label='tab:luft', caption="Messwerte zum Brechungsindex von Luft")
# with open('build/n_luft.txt', 'w') as f:
#     f.write(md)

# md = pd.read_csv('tables/n_luft.csv')
# md = md.to_numpy()
# T = const.convert_temperature(21.5, 'Celsius', 'Kelvin')

# def func(p, a, m):
#     return a + p * m

# L = 0.1 # in meter

# def n_mit_druck(p, M): 
#     return M * 632.8 * 10**(-9) / L

# n_mit_druck_gemessen = n_mit_druck()
# p = md[:,0]
# maxima1 = md[:,1]; maxima2  = md[:,2]; maxima3 = md[:,3]
# p_c = np.concatenate((p[:-1],p,p),axis=None)
# max_c = np.concatenate((maxima1[:-1],maxima2,maxima3),axis=None)
# params, cov = curve_fit(func, p_c, max_c)
# plt.plot(p_c, max_c, 'r+', label="Daten")
# plt.plot(p, func(p, *params), 'b', label="Regression")
# plt.ylabel('Kontrast')
# plt.tight_layout()
# plt.grid(':')
# plt.legend(loc="best")
# plt.savefig("build/Luft.pdf")
# plt.clf()

md = pd.read_csv('tables/n_luft.csv')
hea = list(['p \ Maxima:', 'Versuch 1', 'Versuch 2', 'Versuch 3'])
md = md.to_latex(index = False, column_format= "c c c c", decimal=',', header=hea, label='tab:luft', caption="Messwerte zum Brechungsindex von Luft")
with open('build/n_luft.txt', 'w') as f:
    f.write(md)

md = pd.read_csv('tables/n_luft.csv')
md = md.to_numpy()
T = const.convert_temperature(21.5, 'Celsius', 'Kelvin')

def func1(p, m):     # Lorentz-Lorenz
    return p * m / T + 1

L = 0.1 # in meter

def n_mit_druck(M): 
    return M * 632.8 * 10**(-9) / L + 1


p = md[:,0]
maxima1 = md[:,1]; maxima2  = md[:,2]; maxima3 = md[:,3]
p_c = np.concatenate((p[:-1],p,p),axis=None)
max_c = np.concatenate((maxima1[:-1],maxima2,maxima3),axis=None)


n_mit_druck_gemessen = n_mit_druck(max_c)
params, cov = curve_fit(func1, p_c, n_mit_druck_gemessen)
print('Parameter: ', params, '\nFehler: ', np.sqrt(np.diag(cov)))

plt.plot(p_c, n_mit_druck_gemessen, 'r+', label="Daten")
plt.plot(p, func1(p, *params), 'b', label="Regression")
plt.ylabel('n(p, T = 21,5°)')
plt.xlabel('p')
plt.tight_layout()
plt.grid(':')
plt.legend(loc="best")
plt.savefig("build/Luft.pdf")
plt.clf()

# def func2(p, T2, m):
#     return p * m / T2 + 1

# print(func2(1.013, 15, params))

####################
md = pd.read_csv('tables/kontrast.csv')
md = md.to_numpy()

def fit(winkel,delta, A):
    return A * np.abs(np.sin(2 * winkel - delta))
winkel = md[:,0]
winkel = winkel * np.pi / 180
params, cov = curve_fit(fit, winkel, K)
x = np.linspace(winkel[0] * 0.9, winkel[-1] *1.1)

plt.plot(winkel, K, 'r+', label="Daten")
plt.plot(x, fit(x, *params), 'b', label="Regression")
plt.scatter(x[fit(x, *params).argmax()], fit(x, *params).max(), c='g', zorder=3, label='Maximum')
plt.ylabel('Kontrast')
plt.xlim(x[0], x[-1])
plt.ylim(0, 1)
plt.tight_layout()
plt.grid(':')
plt.legend(loc="best")
plt.savefig("build/Kontrast.pdf")
plt.clf()

######################
