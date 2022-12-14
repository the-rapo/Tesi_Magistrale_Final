# LIBs
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
# Custom LIBs

os.chdir(r'C:\Users\rapon\Documents\UNI\Tesi Magistrale\Python\Tesi_Magistrale_Final')

data_path_ON = 'data/processed/Corsini2021/Corsini2021_Processed_ON.csv'
# mpl.rcParams["font.size"] = 18

# - DATI
low_eta = 0.5
high_eta = 0.515
censura = True
rend_max = 0.545

data_ON = pd.read_csv(data_path_ON)
rendimento = data_ON['Rendimento'].values
alto_rendimento = data_ON[data_ON.Rendimento > high_eta]
basso_rendimento = data_ON[data_ON.Rendimento < low_eta]
low_perc = len(basso_rendimento) / len(data_ON) * 100
high_perc = len(alto_rendimento) / len(data_ON) * 100
'''
rel_pwr = data_ON['PwrTOT_rel'].values
LowP_data = filter_data(data_ON, None, None, 0, low_P)
HighP_data = filter_data(data_ON, None, None, high_P, 1)
low_perc = len(LowP_data) / len(data_ON) * 100
high_perc = len(HighP_data) / len(data_ON) * 100
text_lowP = 'P < ' + str(low_P) + ' = ' "{:.1f}".format(low_perc) + '%'
text_highP = 'P > ' + str(high_P) + ' = ' "{:.1f}".format(high_perc) + '%'
'''
# - GENERALE GRAFICO
fig = plt.figure(figsize=(18, 5), constrained_layout=False)
spec = fig.add_gridspec(1, 3)
ax0 = fig.add_subplot(spec[0, 0:2])
ax1 = fig.add_subplot(spec[0, 2])

# - ISTOGRAMMA (ax0)
cm = plt.cm.get_cmap('turbo')
_, bins, patches = ax0.hist(rendimento, bins=100, density=False, range=(0.46, 0.54))
bin_centers = 0.5 * (bins[:-1] + bins[1:])
col = bin_centers - min(bin_centers)
col /= max(col)
for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', cm(c))

start, end = ax0.get_xlim()
ax0.xaxis.set_ticks(np.arange(0.46, 0.54, 0.005))

locs_y = ax0.get_yticks()
locs_x = ax0.get_xticks()
ax0.set_yticks(locs_y, np.round(locs_y / len(rendimento) * 100, 1))
ax0.set_ylabel('Frequenza percentuale [%]')
if censura:
    ax0.set_xticks(locs_x, np.round(locs_x * 100 / rend_max, 1))
    ax0.set_xlabel('Rendimento Rel. [%]')

else:
    ax0.set_xticks(locs_x, np.round(locs_x * 100, 1))
    ax0.set_xlabel('Rendimento [%]')


ax0.axvline(x=low_eta, color='black', linestyle='dashed', linewidth=1, alpha=0.5)
ax0.axvline(x=high_eta, color='black', linestyle='dashed', linewidth=1, alpha=0.5)

ax0.text(low_eta, locs_y[-1] * 0.9, 'Soglia \n Basso ' r'$\eta$', ha='center', va='center', rotation='vertical',
         backgroundcolor='white', color='k')
ax0.text(high_eta, locs_y[-1] * 0.9, 'Soglia \n Alto ' r'$\eta$', ha='center', va='center', rotation='vertical',
         backgroundcolor='white', color='k')


# - TORTA (ax1)
pie_labels = ["Basso "  r'$\eta$' "\n "  "{:.1f}".format(low_perc) + '%',
              "Medio "  r'$\eta$' "\n "  "{:.1f}".format(100 - high_perc - low_perc) + '%',
              "Alto "   r'$\eta$' "\n "  "{:.1f}".format(high_perc) + '%']

pie_data = [low_perc, 100 - low_perc - high_perc, high_perc]

wedges, texts = ax1.pie(pie_data, wedgeprops=dict(width=0.5), startangle=-40,
                        colors=['tab:blue', 'tab:green', 'tab:red'])

bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
kw = dict(arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props, zorder=0, va="center")

for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    kw["arrowprops"].update({"connectionstyle": connectionstyle})
    ax1.annotate(pie_labels[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                 horizontalalignment='center', **kw, fontsize=11)

# plt.show()

plt.savefig("IO/rendimento_hist.png", bbox_inches='tight', dpi=300)
