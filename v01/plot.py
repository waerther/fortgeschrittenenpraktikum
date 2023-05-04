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


# Tabelle 1:

md = pd.read_csv('tables/20ns_table.csv')
md = md.to_numpy()
x = md[:,0]
counts = md[:,1]
md_rearranged = np.c_[x[:np.int(np.round(len(x)/2))], counts[:np.int(np.round(len(x)/2))], x[np.int(np.round(len(x)/2)):], counts[np.int(np.round(len(x)/2)):]]
# md_rearranged = md.reshape((-1,4))
hea = list(['Relative Verzögerung / ns', 'Zählrate', 'Relative Verzögerung / ns', 'Zählrate'])
pandas_md = pd.DataFrame(md_rearranged, columns=hea)
md_table = pandas_md.to_latex(index = False, column_format= "c c c c", decimal=',', header=hea, label='tab:20ns_table', caption='Messreihe bei einer Impulsdauer von 20ns.')
with open('build/20ns_table.txt', 'w') as f:
    f.write(md_table)

# Plot 1

plt.hlines(md[25,1], md[0,0], md[25,0], label='Halbwertsbreite', )
plt.plot(x, counts, 'r+', label="Daten", marker='x')
plt.ylabel('Counts')
plt.xlabel('Relative Verzögerung / ns')
plt.tight_layout()
plt.grid(':')
plt.legend(loc="best")
plt.savefig("build/20ns_plot.pdf")
plt.clf()

# Tabelle 2:

md = pd.read_csv('tables/30ns_table.csv')
md = md.to_numpy()
x = md[:,0]
counts = md[:,1]
md_rearranged = np.c_[x[:np.int(np.round(len(x)/2))], counts[:np.int(np.round(len(x)/2))], x[np.int(np.round(len(x)/2)):], counts[np.int(np.round(len(x)/2)):]]
# md_rearranged = md.reshape((-1,4))
hea = list(['Relative Verzögerung / ns', 'Zählrate', 'Relative Verzögerung / ns', 'Zählrate'])
pandas_md = pd.DataFrame(md_rearranged, columns=hea)
md_table = pandas_md.to_latex(index = False, column_format= "c c c c", decimal=',', header=hea, label='tab:30ns_table', caption='Messreihe bei einer Impulsdauer von 30ns.')
with open('build/30ns_table.txt', 'w') as f:
    f.write(md_table)

# Plot 2

plt.hlines(md[40,1], md[4,0], md[40,0], label='Halbwertsbreite')
plt.plot(x, counts, 'r+', label="Daten", marker='x')
plt.ylabel('Counts')
plt.xlabel('Relative Verzögerung / ns')
plt.tight_layout()
plt.grid(':')
plt.legend(loc="best")
plt.savefig("build/30ns_plot.pdf")
plt.clf()

# Tabelle 3

md = pd.read_csv('tables/monoflop.csv')
md = md.to_numpy()
hea = list(['Zeit', 'Bins'])
pandas_md = pd.DataFrame(md, columns=hea)
md_table = pandas_md.to_latex(index = False, column_format= "c c", decimal=',', header=hea, label='tab:zeiteinteilung', caption='Messreihe zur Bestimmung der Zeiteinteilung der Bins.')
with open('build/zeiteinteilung.txt', 'w') as f:
    f.write(md_table)

# Plot 3

def func(x, m, b):
    return m * x + b

md = pd.read_csv('tables/monoflop.csv')
md = md.to_numpy()

y = md[:,0]
bins = md[:,1]

params, cov = curve_fit(func, bins, y)
print('Parameter: ', params, '\nFehler: ', np.sqrt(np.diag(cov)))

m = ufloat(params[0], np.sqrt(np.diag(cov))[0])
b = ufloat(params[1], np.sqrt(np.diag(cov))[1])

plt.plot(bins, y, 'r+', label="Messwerte",)
plt.plot(bins, func(bins, *params), 'b', label="Regression", zorder=0)
plt.ylabel('t / ns')
plt.xlabel('Bins')
plt.tight_layout()
plt.grid(':')
plt.legend(loc="best")
plt.savefig('build/zeiteinteilung.pdf')