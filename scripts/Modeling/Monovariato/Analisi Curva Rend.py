# LIBs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.collections as collections

import os
# Custom LIBs
from cust_libs.modeling import Load_Poly_model, Predict_Point_Poly
from cust_libs.misc import transf_fun
#
os.chdir(r'C:\Users\rapon\Documents\UNI\Tesi Magistrale\Python\Tesi_Magistrale_Final')
#
model_fld = 'models/univariate/Poly/deg4_low'
poly_modelname = 'Paper'
data = pd.read_csv('data/processed/Corsini2021/Corsini2021_Processed_ON.csv')
save_path = 'temp/'
#
mpl.rcParams["font.size"] = 18
model, poly = Load_Poly_model(model_fld)
rel2tot, tot2rel = transf_fun(data)

x_plot = np.linspace(0.4, 0.9, 2000, endpoint=True)
x_plot_2 = poly.transform(x_plot.reshape(-1, 1))
y_rend = model.predict(x_plot_2).flatten()
y_rend_grad = np.gradient(y_rend)
nabla_eta45_6 = float(Predict_Point_Poly(model, poly, 0.6) - Predict_Point_Poly(model, poly, 0.45))
nabla_eta6_7 = float(Predict_Point_Poly(model, poly, 0.7) - Predict_Point_Poly(model, poly, 0.6))
nabla_eta7_9 = float(Predict_Point_Poly(model, poly, 0.9) - Predict_Point_Poly(model, poly, 0.7))
text_napla1 = r' $\nabla$ $\eta$ = ' "{:.4f}".format(nabla_eta45_6)
text_napla2 = r' $\nabla$ $\eta$ = ' "{:.4f}".format(nabla_eta6_7)
text_napla3 = r' $\nabla$ $\eta$ = ' "{:.4f}".format(nabla_eta7_9)

# PLOT
'''
fig, ax = plt.subplots(2, 1, figsize=(16, 18))
ax[0].plot(x_plot, y_rend, color='k', linewidth=4)

ax[0].set_ylabel(r'Rendimento')
ax[0].set_xlabel(r'Potenza Relativa')
secax_x0 = ax[0].secondary_xaxis(
    'top', functions=(rel2tot, tot2rel))
secax_x0.set_xlabel(r'$P\ $ [MW]')
ax[0].axvline(x=0.45, color='black', linestyle='dashed', linewidth=1)
ax[0].axvline(x=0.6, color='green', linestyle='dashed', linewidth=1)
ax[0].axvline(x=0.7, color='blue', linestyle='dashed', linewidth=1)
ax[0].axvline(x=0.9, color='red', linestyle='dashed', linewidth=1)
ax[0].axhline(y=Predict_Point_Poly(model, poly, 0.45), color='black', linestyle='dashed', linewidth=1)
ax[0].axhline(y=Predict_Point_Poly(model, poly, 0.6), color='green', linestyle='dashed', linewidth=1)
ax[0].axhline(y=Predict_Point_Poly(model, poly, 0.7), color='blue', linestyle='dashed', linewidth=1)
ax[0].axhline(y=Predict_Point_Poly(model, poly, 0.9), color='red', linestyle='dashed', linewidth=1)

ax[1].plot(x_plot, y_rend_grad, color='k', linewidth=4)
ax[1].set_yscale('log')
ax[1].set_ylabel(r' $\nabla$ Rendimento - Step 0.01')
ax[1].set_xlabel(r'Potenza Relativa')
secax_x1 = ax[1].secondary_xaxis(
    'top', functions=(rel2tot, tot2rel))
secax_x1.set_xlabel(r'$P\ $ [MW]')
ax[1].grid(color='black', which='major', axis='y', alpha=0.5, linestyle='dashed', linewidth=1)
ax[1].grid(color='black', which='minor', axis='y', alpha=0.5, linestyle='dashed', linewidth=0.5)
ax[1].axvline(x=0.45, color='k', linestyle='dashed', linewidth=1)
ax[1].axvline(x=0.6, color='green', linestyle='dashed', linewidth=1)
ax[1].axvline(x=0.7, color='blue', linestyle='dashed', linewidth=1)
ax[1].axvline(x=0.9, color='red', linestyle='dashed', linewidth=1)
ax[0].text(0.505, 0.45, text_napla1, fontsize=15, color='k', ha='left')
ax[0].text(0.6265, 0.45, text_napla2, fontsize=15, color='k', ha='left')
ax[0].text(0.782, 0.45, text_napla3, fontsize=15, color='k', ha='left')

collection = collections.BrokenBarHCollection.span_where(
    x_plot, ymin=0, ymax=1, where=np.logical_and(x_plot > 0.45, x_plot < 0.6), facecolor='green', alpha=0.2,
    label='Alta Potenza')

ax[0].add_collection(collection)

collection = collections.BrokenBarHCollection.span_where(
    x_plot, ymin=0, ymax=1, where=np.logical_and(x_plot > 0.6, x_plot < 0.7), facecolor='blue', alpha=0.2,
    label='Alta Potenza')
ax[0].add_collection(collection)

collection = collections.BrokenBarHCollection.span_where(
    x_plot, ymin=0, ymax=1, where=np.logical_and(x_plot > 0.7, x_plot < 0.9), facecolor='red', alpha=0.2,
    label='Alta Potenza')

ax[0].add_collection(collection)

plt.savefig(save_path + 'Analisi_rend.png', bbox_inches='tight')
'''

fig, ax = plt.subplots(figsize=(18, 9))
ax.plot(x_plot, y_rend, color='k', linewidth=4)

ax.set_ylabel(r'Rendimento')
ax.set_xlabel(r'Potenza Relativa')
secax_x = ax.secondary_xaxis(
    'top', functions=(rel2tot, tot2rel))
secax_x.set_xlabel(r'$P\ $ [MW]')
ax.axvline(x=0.45, color='black', linestyle='dashed', linewidth=1)
ax.axvline(x=0.6, color='green', linestyle='dashed', linewidth=1)
ax.axvline(x=0.7, color='blue', linestyle='dashed', linewidth=1)
ax.axvline(x=0.9, color='red', linestyle='dashed', linewidth=1)
ax.axhline(y=Predict_Point_Poly(model, poly, 0.45), color='black', linestyle='dashed', linewidth=1)
ax.axhline(y=Predict_Point_Poly(model, poly, 0.6), color='green', linestyle='dashed', linewidth=1)
ax.axhline(y=Predict_Point_Poly(model, poly, 0.7), color='blue', linestyle='dashed', linewidth=1)
ax.axhline(y=Predict_Point_Poly(model, poly, 0.9), color='red', linestyle='dashed', linewidth=1)

ax.text(0.505, 0.45, text_napla1, fontsize=15, color='k', ha='left')
ax.text(0.6265, 0.45, text_napla2, fontsize=15, color='k', ha='left')
ax.text(0.782, 0.45, text_napla3, fontsize=15, color='k', ha='left')

collection = collections.BrokenBarHCollection.span_where(
    x_plot, ymin=0, ymax=1, where=np.logical_and(x_plot > 0.45, x_plot < 0.6), facecolor='green', alpha=0.2,
    label='Alta Potenza')

ax.add_collection(collection)

collection = collections.BrokenBarHCollection.span_where(
    x_plot, ymin=0, ymax=1, where=np.logical_and(x_plot > 0.6, x_plot < 0.7), facecolor='blue', alpha=0.2,
    label='Alta Potenza')
ax.add_collection(collection)

collection = collections.BrokenBarHCollection.span_where(
    x_plot, ymin=0, ymax=1, where=np.logical_and(x_plot > 0.7, x_plot < 0.9), facecolor='red', alpha=0.2,
    label='Alta Potenza')

ax.add_collection(collection)

# plt.savefig(save_path + 'Analisi_rend.png', bbox_inches='tight')
plt.show()