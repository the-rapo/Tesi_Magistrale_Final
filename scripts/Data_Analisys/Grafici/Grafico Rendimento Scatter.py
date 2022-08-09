# LIBs
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
# Custom LIBs
from cust_libs.data_processing import filter_data, transf_fun

os.chdir(r'C:\Users\rapon\Documents\UNI\Tesi Magistrale\Python\Tesi_Magistrale_Final')
#
data_path_ON = 'data/processed/Corsini2021/Corsini2021_Processed_ON.csv'

data = pd.read_csv(data_path_ON)

mpl.rcParams["font.size"] = 18

fig, ax = plt.subplots(figsize=(18, 9))

gradh = filter_data(data, 0.02, None, None, None)       # 8.2 MW / min
gradmh = filter_data(data, 0.005, 0.02, None, None)
grad0 = filter_data(data, -0.005, 0.005, None, None)
gradml = filter_data(data, -0.02, -0.005, None, None)
gradl = filter_data(data, None, -0.02, None, None)
gradh.reset_index().plot.scatter(x='PwrTOT_rel', y='Rendimento', label=r'$ \nabla  P  >  8.5$ MW/min ',
                                 marker='o', color='tab:red', ax=ax)
gradmh.reset_index().plot.scatter(x='PwrTOT_rel', y='Rendimento', label=r'$ \nabla  P  \in $ [ 2.0;  8.5 ]  MW/min ',
                                  marker='o', color='tab:orange', ax=ax)
grad0.reset_index().plot.scatter(x='PwrTOT_rel', y='Rendimento', label=r'$ \nabla  P  \in $ [-2.0;  2.0 ]  MW/min ',
                                 marker='o', color='tab:green', ax=ax)
gradml.reset_index().plot.scatter(x='PwrTOT_rel', y='Rendimento', label=r'$ \nabla  P  \in $ [-8.5; -2.0 ]  MW/min ',
                                  marker='o', color='tab:cyan', ax=ax)
gradl.reset_index().plot.scatter(x='PwrTOT_rel', y='Rendimento', label=r'$ \nabla  P  <  8.5$ MW/min ',
                                 marker='o', color='tab:blue', ax=ax)

rel2tot, tot2rel = transf_fun(data)

ax.set_xlabel(r'$P_{rel}$')
secax1_x = ax.secondary_xaxis('top', functions=(rel2tot, tot2rel))
secax1_x.set_xlabel(r'$P\ [MW]$')


# fig.suptitle('Curva di rendimento Impianto Corsini - Dataset 2021')
plt.show()
