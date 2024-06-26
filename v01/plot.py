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
plt.errorbar(x=x, y=counts,xerr=0, yerr=np.sqrt(counts), label="Daten", color='darkred',elinewidth=1, fmt=' ')
plt.scatter(x=x, y=counts, color='darkred', s=3)
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
plt.errorbar(x=x, y=counts,xerr=0, yerr=np.sqrt(counts), label="Daten", color='darkred',elinewidth=1, fmt=' ')
plt.scatter(x=x, y=counts, color='darkred', s=3)
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
hea = list([r't / (\mu s)', 'Binssss'])
pandas_md = pd.DataFrame(md, columns=hea)
formatters = {'Bins': lambda x: f'{int(x)}'}
md_table = pandas_md.to_latex(index = False, column_format= "c c", decimal='.', header=hea, label='tab:zeiteinteilung', caption='Messreihe zur Bestimmung der Zeiteinteilung der Bins.', formatters=formatters)
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
# print('Parameter: ', params, '\nFehler: ', np.sqrt(np.diag(cov)))

m = ufloat(params[0], np.sqrt(np.diag(cov))[0])
b = ufloat(params[1], np.sqrt(np.diag(cov))[1])

plt.plot(bins, y, 'r+', label="Messwerte",)
plt.plot(bins, func(bins, *params), 'b', label="Regression", zorder=0)
plt.ylabel(r't / ($\mu$ s)')
plt.xlabel('Bins')
plt.tight_layout()
plt.grid(':')
plt.legend(loc="best")
plt.savefig('build/zeiteinteilung.pdf')
plt.clf()

########################


# Plot Histogramm Daten

counts=np.genfromtxt('tables/myon.txt', unpack=True)
bins = np.linspace(1, 512, 512)
bin_heights = counts
plt.hist(bins, bins=bins, weights=bin_heights, log=True, color='grey')
plt.xlabel('Bins')
plt.ylabel('Counts')
# plt.title('Histogram der Zerfälle pro Bin')
plt.tight_layout()
plt.savefig('build/myonenhist.pdf')
plt.clf()

# Plot Myonen Fit

def func2(t, N_0, lam, U):
    return N_0 * np.exp(- lam * t) + U

def func3(bin):                 # Converts bins to time
    return (m.nominal_value * bin + b.nominal_value) * 10**(-6)

bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])

t = func3(bins)#- func(4, m, b).nominal_value * 10**-6
fitmask = (counts > 1)
params, cov = curve_fit(func2, t[fitmask], counts[fitmask], bounds=bounds)
N_0 = ufloat(params[0], np.sqrt(np.diag(cov))[0])
lam = ufloat(params[1], np.sqrt(np.diag(cov))[1])
U = ufloat(params[2], np.sqrt(np.diag(cov))[2])
 
print('tau', (1 / lam ) * 10**6)
print('N_0 = ', N_0)
print('lam = ', lam)
print('U = ', U)


x = np.arange(-0.5 *  10**(-6),10 * 10**(-6),0.1* 10**(-6))
plt.plot(x, func2(x, *params), ls='-', c='b', zorder=1, label='Fit')
plt.errorbar(x=t[4:], y=counts[4:],xerr=0, yerr=np.sqrt(counts[4:]), label="Messdaten", color='darkred',elinewidth=0.5, fmt=' ')
plt.scatter(x=t[4:], y=counts[4:], color='darkred', s=2)
plt.plot(t[:4],counts[:4], 'o', alpha=0.3, c='b', zorder=0, label='Ignorierte Messdaten')
plt.xlabel('t / s')
plt.ylabel('counts')
plt.grid('::', alpha=0.3)
plt.legend()
plt.savefig('build/myonenfit.pdf')
plt.clf()

# Tabelle Myon

def format_int(value):
    return f"{int(value):d}"


data1 = np.c_[bins[:40], counts[:40], bins[40:80], counts[40:80], bins[80:120], counts[80:120], bins[120:160],counts[120:160]]

data2 = np.c_[bins[160:200], counts[160:200], bins[200:240], counts[200:240], bins[240:280], counts[240:280], bins[280:320],counts[280:320]]

data3 = np.c_[bins[360:398], counts[360:398], bins[398:436], counts[398:436], bins[436:474], counts[436:474], bins[474:513], counts[474:512]]

df = pd.DataFrame(data1)
hea = list(['Bin', 'Counts','Bin', 'Counts','Bin', 'Counts','Bin', 'Counts'])
df_table = df.to_latex(index = False, column_format= "c c c c c c c c", header=hea, 
            label='tab:messdaten_myonen1', caption='Messdaten der Lebenszeitmessung der Myonen.', 
            formatters=[format_int, format_int, format_int, format_int, format_int, format_int, format_int, format_int])
with open('build/messdaten_myonen_1.txt', 'w') as f:
    f.write(df_table)
df = pd.DataFrame(data2)
df_table = df.to_latex(index = False, column_format= "c c c c c c c c", header=hea, 
            label='tab:messdaten_myonen2', caption='Messdaten der Lebenszeitmessung der Myonen.', 
            formatters=[format_int, format_int, format_int, format_int, format_int, format_int, format_int, format_int])
with open('build/messdaten_myonen_2.txt', 'w') as f:
    f.write(df_table)
df = pd.DataFrame(data3)
df_table = df.to_latex(index = False, column_format= "c c c c c c c c", header=hea, 
            label='tab:messdaten_myonen3', caption='Messdaten der Lebenszeitmessung der Myonen.', 
            formatters=[format_int, format_int, format_int, format_int, format_int, format_int, format_int, format_int])
with open('build/messdaten_myonen_3.txt', 'w') as f:
    f.write(df_table)