# LIBs
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy.polynomial import Polynomial

# Custom LIBs
from cust_libs.data_processing import filter_data
from cust_libs.misc import transf_fun
mode = 'analisi_grad'

os.chdir(r'C:\Users\rapon\Documents\UNI\Tesi Magistrale\Python\Tesi_Magistrale_Final')
#
data_path_ON = 'data/processed/Corsini2021/Corsini2021_Processed_ON.csv'

data = pd.read_csv(data_path_ON)

mpl.rcParams["font.size"] = 18

fig, ax = plt.subplots(figsize=(18, 9))

low_grad = filter_data(data, -0.001, 0.001, None, None)
gradh = filter_data(data, 0.02, None, None, None)       # 8.2 MW / min
gradmh = filter_data(data, 0.005, 0.02, None, None)
grad0 = filter_data(data, -0.005, 0.005, None, None)
gradml = filter_data(data, -0.02, -0.005, None, None)
gradl = filter_data(data, None, -0.02, None, None)

if mode == 'analisi_grad':

    gradh.reset_index().plot.scatter(x='PwrTOT_rel', y='Rendimento', label=r'$ \nabla  P  >  8.5$ MW/min ',
                                     marker='o', color='tab:red', ax=ax, alpha=0.6)
    gradmh.reset_index().plot.scatter(x='PwrTOT_rel', y='Rendimento', label=r'$ \nabla  P  \in $ [ 2.0;  8.5 ]  MW/min ',
                                      marker='o', color='tab:orange', ax=ax, alpha=0.6)
    grad0.reset_index().plot.scatter(x='PwrTOT_rel', y='Rendimento', label=r'$ \nabla  P  \in $ [-2.0;  2.0 ]  MW/min ',
                                     marker='o', color='tab:green', ax=ax, alpha=0.6)
    gradml.reset_index().plot.scatter(x='PwrTOT_rel', y='Rendimento', label=r'$ \nabla  P  \in $ [-8.5; -2.0 ]  MW/min ',
                                      marker='o', color='tab:cyan', ax=ax, alpha=0.6)
    gradl.reset_index().plot.scatter(x='PwrTOT_rel', y='Rendimento', label=r'$ \nabla  P  <  8.5$ MW/min ',
                                     marker='o', color='tab:blue', ax=ax, alpha=0.6)

    x0 = grad0['PwrTOT_rel'].values
    y0 = grad0['Rendimento'].values
    p0 = Polynomial.fit(x0, y0, 4)
    ax.plot(*p0.linspace(), color='k', linewidth=3, label=r'Reg. pol. $\nabla P > 8.5$')

    xh = gradh['PwrTOT_rel'].values
    yh = gradh['Rendimento'].values
    ph = Polynomial.fit(xh, yh, 4)
    ax.plot(*ph.linspace(), color='r', linewidth=3, label=r'Reg. pol. $ |\nabla P| < 2$')

    xl = gradl['PwrTOT_rel'].values
    yl = gradl['Rendimento'].values
    pl = Polynomial.fit(xl, yl, 4)
    ax.plot(*pl.linspace(), color='b', linewidth=3, label=r'Reg. pol. $\nabla P < - 8.5$')

elif mode == 'full':
    data.reset_index().plot.scatter(x='PwrTOT_rel', y='Rendimento', marker='o', color='tab:blue', ax=ax, alpha=0.5,
                                    label='Dati storici')
    x = data['PwrTOT_rel'].values
    y = data['Rendimento'].values
    p = Polynomial.fit(x, y, 4)
    ax.plot(*p.linspace(), color='indianred', linewidth=4, label='Regressione polinomiale grado 4')
elif mode == 'vs':
    data.reset_index().plot.scatter(x='PwrTOT_rel', y='Rendimento', marker='o', color='tab:blue', ax=ax,
                                    label='Full data', alpha=0.7)
    low_grad.reset_index().plot.scatter(x='PwrTOT_rel', y='Rendimento', marker='o', color='tab:green', ax=ax,
                                        label='Basso gradiente', alpha=0.4)
    x = low_grad['PwrTOT_rel'].values
    y = low_grad['Rendimento'].values
    p = Polynomial.fit(x, y, 4)
    ax.plot(*p.linspace(), color='indianred', linewidth=4, label='Reg. pol. basso gradiente')

rel2tot, tot2rel = transf_fun(data)

ax.set_xlim([0, 0.9])
ax.set_ylim([0, 0.55])
locs_x = ax.get_xticks()
ax.set_xticks(locs_x, np.round(locs_x * 100, 1))
ax.set_xlabel('Potenza Relativa [%]')
locs_y = ax.get_yticks()
ax.set_yticks(locs_y, np.round(locs_y * 100, 1))
ax.set_ylabel('Rendimento [%]')
secax1_x = ax.secondary_xaxis('top', functions=(rel2tot, tot2rel))
secax1_x.set_xlabel(r'Potenza $[MW]$')
plt.legend()
plt.savefig("temp/scatter_" + mode + ".png", bbox_inches='tight')

# fig.suptitle('Curva di rendimento Impianto Corsini - Dataset 2021')
plt.show()
