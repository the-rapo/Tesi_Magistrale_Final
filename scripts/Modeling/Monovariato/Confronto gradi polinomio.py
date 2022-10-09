import os

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib as mpl
import matplotlib.collections as collections

from cust_libs.modeling import RMSE, MAE, RMSE_MAE_plot, Save_Poly_Model
import numpy as np
from cust_libs.misc import transf_fun
import matplotlib.pyplot as plt
# Param
filtered = False
#
os.chdir(r'C:\Users\rapon\Documents\UNI\Tesi Magistrale\Python\Tesi_Magistrale_Final')
n = 1000
censura = True
rend_max = 0.545
#
data_path = 'data/processed/Corsini2021/Corsini2021_Processed_ON.csv'
data = pd.read_csv(data_path)
save_path = 'IO/'
x = data['PwrTOT_rel'].values.reshape(-1, 1)
y = data['Rendimento'].values.reshape(-1, 1)

rel2tot, tot2rel = transf_fun(data)

poly_trans4 = PolynomialFeatures(degree=4, include_bias=False)
x_trans4 = poly_trans4.fit_transform(x)

model4 = LinearRegression()
model4.fit(x_trans4, y)

poly_trans5 = PolynomialFeatures(degree=5, include_bias=False)
x_trans5 = poly_trans5.fit_transform(x)

model5 = LinearRegression()
model5.fit(x_trans5, y)

poly_trans6 = PolynomialFeatures(degree=6, include_bias=False)
x_trans6 = poly_trans6.fit_transform(x)

model6 = LinearRegression()
model6.fit(x_trans6, y)

poly_trans7 = PolynomialFeatures(degree=7, include_bias=False)
x_trans7 = poly_trans7.fit_transform(x)

model7 = LinearRegression()
model7.fit(x_trans7, y)
mpl.rcParams["font.size"] = 18

fig, ax = plt.subplots(figsize=(18, 9))

x = np.linspace(0.1, 1.0, n, endpoint=True).reshape(-1, 1)

y4 = model4.predict(poly_trans4.transform(x))
y5 = model5.predict(poly_trans5.transform(x))
y6 = model6.predict(poly_trans6.transform(x))
y7 = model7.predict(poly_trans7.transform(x))

x = x.flatten()
ax.plot(x, y4, color='k', linewidth=4, label='Grado 4')
ax.plot(x, y5, color='r', linewidth=4, label='Grado 5')
ax.plot(x, y6, color='g', linewidth=4, label='Grado 6')
ax.plot(x, y7, color='b', linewidth=4, label='Grado 7')

collection = collections.BrokenBarHCollection.span_where(
    x, ymin=0, ymax=1, where=np.logical_and(x > 0.1, x < 0.9), facecolor='green', alpha=0.2,
    label='Alto numero di entrate')

ax.add_collection(collection)

collection = collections.BrokenBarHCollection.span_where(
    x, ymin=0, ymax=1, where=(x > 0.9), facecolor='red', alpha=0.2,
    label='Basso numero di entrate')

ax.add_collection(collection)

ax.set_ylim([0, 0.6])

if censura:
    locs_y = ax.get_yticks()
    ax.set_yticks(locs_y, np.round(locs_y * 100 / rend_max, 1))
    ax.set_ylabel('Rendimento Rel. [%]')
else:
    locs_y = ax.get_yticks()
    ax.set_yticks(locs_y, np.round(locs_y * 100, 1))
    ax.set_ylabel('Rendimento [%]')

ax.set_xlabel(r'$P_{Rel}$')
secax_x = ax.secondary_xaxis(
    'top', functions=(rel2tot, tot2rel))
secax_x.set_xlabel(r'$P\ $ [MW]')

plt.legend(loc='best')

plt.savefig(save_path + 'gradi_pol_mono.png', bbox_inches='tight')
plt.show()