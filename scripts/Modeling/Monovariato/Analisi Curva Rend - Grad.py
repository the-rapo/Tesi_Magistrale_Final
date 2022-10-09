# LIBs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import os
# Custom LIBs
from cust_libs.modeling import Load_ML_Model
from cust_libs.misc import transf_fun
#
os.chdir(r'C:\Users\rapon\Documents\UNI\Tesi Magistrale\Python\Tesi_Magistrale_Final')
#
model_path = 'models/multivariate/ML/RND_FOR/RND_FOR_02.joblib'
poly_modelname = 'Mono-4'
data = pd.read_csv('data/processed/Corsini2021/Corsini2021_Processed_ON.csv')
save_path = 'IO/'
power_1 = 0.5
power_2 = 0.6
power_3 = 0.7
censura = True
rend_max = 0.545
n = 100
#
mpl.rcParams["font.size"] = 18
model, _ = Load_ML_Model(model_path)
rel2tot, tot2rel = transf_fun(data)

grad = np.linspace(-0.04, 0.04, n, endpoint=True)
pwr1 = np.linspace(power_1, power_1, n, endpoint=True)
y_rend1 = model.predict(np.vstack((pwr1.flatten(), grad.flatten())).T)

pwr2 = np.linspace(power_2, power_2, n, endpoint=True)
y_rend2 = model.predict(np.vstack((pwr2.flatten(), grad.flatten())).T)

pwr3 = np.linspace(power_3, power_3, n, endpoint=True)
y_rend3 = model.predict(np.vstack((pwr3.flatten(), grad.flatten())).T)

fig, ax = plt.subplots(figsize=(18, 9))
ax.plot(grad, y_rend1, color='g', linewidth=4, label=r'$P$  = 200 MW')
ax.plot(grad, y_rend2, color='b', linewidth=4, label=r'$P$  = 250 MW')
ax.plot(grad, y_rend3, color='r', linewidth=4, label=r'$P$  = 300 MW')

if censura:
    locs_y = ax.get_yticks()
    ax.set_yticks(locs_y, np.round(locs_y * 100 / rend_max, 1))
    ax.set_ylabel('Rendimento Rel. [%]')
else:
    locs_y = ax.get_yticks()
    ax.set_yticks(locs_y, np.round(locs_y * 100, 1))
    ax.set_ylabel('Rendimento [%]')

ax.set_xlabel(r'$\nabla P_{rel}$ [$min^{-1}$]')
secax_x = ax.secondary_xaxis(
    'top', functions=(rel2tot, tot2rel))
secax_x.set_xlabel(r'$ \nabla P\ $ [MW/min]')
plt.legend(loc='best')

plt.savefig(save_path + 'Analisi_rend_grad.png', bbox_inches='tight')
plt.show()
