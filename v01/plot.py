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
hea = list(['Relative Verzögerung / t', 'Zählrate', 'Relative Verzögerung / t', 'Zählrate'])
pandas_md = pd.DataFrame(md_rearranged, columns=hea)
md_table = pandas_md.to_latex(index = False, column_format= "c c c c", decimal=',', header=hea, label='tab:20ns_table', caption='Messreihe bei einer Impulsdauer von 20ns.')
with open('build/20ns_table.txt', 'w') as f:
    f.write(md_table)

# Plot 1

plt.plot(x, counts, 'r+', label="Daten", marker='x')
plt.ylabel('Counts')
plt.xlabel('Relative Verzögerung / t')
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
hea = list(['Relative Verzögerung / t', 'Zählrate', 'Relative Verzögerung / t', 'Zählrate'])
pandas_md = pd.DataFrame(md_rearranged, columns=hea)
md_table = pandas_md.to_latex(index = False, column_format= "c c c c", decimal=',', header=hea, label='tab:30ns_table', caption='Messreihe bei einer Impulsdauer von 30ns.')
with open('build/30ns_table.txt', 'w') as f:
    f.write(md_table)

# Plot 2

plt.plot(x, counts, 'r+', label="Daten", marker='x')
plt.ylabel('Counts')
plt.xlabel('Relative Verzögerung / t')
plt.tight_layout()
plt.grid(':')
plt.legend(loc="best")
plt.savefig("build/30ns_plot.pdf")
plt.clf()