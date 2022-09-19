import pandas as pd


def simple_ramp(low_p, high_p, t_lowp, t_highp, t_ramp):
    import numpy as np
    import pandas as pd

    p_nom = 410

    low_p_list = np.linspace(low_p, low_p, t_lowp , endpoint=True)
    ramp_list = np.linspace(low_p, high_p, t_ramp, endpoint=True)
    high_p_list = np.linspace(high_p, high_p, t_highp , endpoint=True)

    PwrTOT_rel = []
    PwrTOT_rel.extend(low_p_list)
    PwrTOT_rel.extend(ramp_list)
    PwrTOT_rel.extend(high_p_list)

    grad = np.gradient(PwrTOT_rel)
    data = pd.DataFrame()
    data['PwrTOT_rel'] = PwrTOT_rel
    data['Grad_PwrTOT_rel'] = grad
    data['PwrTOT'] = data['PwrTOT_rel'] * p_nom
    return data


def simple_sim_prov(data, bess_size, t_lowp, t_highp, low_p, high_p):
    import numpy as np
    import pandas as pd

    p_nom = 410
    pwrbess_lp = (bess_size * .95 / (t_lowp / 60)) / 410
    if pwrbess_lp + low_p > 0.65:
        pwrbess_lp = 0.65 - low_p
        expected_energy = pwrbess_lp * 410 * t_lowp / 60
        pwrbess_hp = (expected_energy * 1.05 / (t_highp / 60)) / 410
    else:
        pwrbess_hp = (bess_size * .95 / (t_highp / 60)) / 410
    new_power = []
    bess_full = False
    bess_SOC = []
    bess_pwr = []

    for i in range(data.shape[0]):

        if data['PwrTOT_rel'].iloc[i] < pwrbess_lp + low_p:
            if not bess_full:
                new_power.append(pwrbess_lp + low_p)
                bess_pwr.append((data['PwrTOT_rel'].iloc[i] - new_power[i]) * 410)
                if i == 0:
                    bess_SOC.append(0)
                else:
                    bess_SOC.append(bess_SOC[i - 1] - bess_pwr[i] / 60)
            else:
                new_power.append(data['PwrTOT_rel'].iloc[i])
                bess_pwr.append(0)
                bess_SOC.append(bess_SOC[i - 1] - bess_pwr[i] / 60)

        elif data['PwrTOT_rel'].iloc[i] < high_p - pwrbess_hp:
            new_power.append(data['PwrTOT_rel'].iloc[i])
            bess_pwr.append(0)
            bess_SOC.append(bess_SOC[i - 1] - bess_pwr[i] / 60)
        else:
            new_power.append(high_p - pwrbess_hp)
            bess_pwr.append((data['PwrTOT_rel'].iloc[i] - new_power[i]) * 410)
            bess_SOC.append(bess_SOC[i - 1] - bess_pwr[i] / 60)

        if bess_SOC[i] >= bess_size:
            bess_full = True
            bess_SOC[i] = bess_size
        else:
            bess_full = False
            if bess_SOC[i] < 0:
                bess_SOC[i] = 0

    grad = np.gradient(new_power)
    data_new = pd.DataFrame()
    data_new['PwrTOT_rel'] = new_power
    data_new['Grad_PwrTOT_rel'] = grad
    data_new['BESS_pwr'] = bess_pwr
    data_new['BESS_SOC'] = bess_SOC
    data_new['PwrTOT'] = data_new['PwrTOT_rel'] * p_nom

    return data_new


def simple_sim(L_op, H_op, bess_max, data):

    import numpy as np
    #
    bess = []
    bess_SOC = []
    first_ramp = True
    new_load = []
    bess_full = False
    bess_empty = True
    p_nom = 410
    #
    for i in range(data.shape[0]):
        if not first_ramp:
            if data['PwrTOT_rel'].iloc[i] > H_op:
                if bess_empty:
                    bess.append(0)
                    bess_SOC.append(bess_SOC[i-1])
                    new_load.append(data['PwrTOT_rel'].iloc[i])
                else:
                    bess.append((data['PwrTOT_rel'].iloc[i] - H_op) * 410)
                    bess_SOC.append(bess_SOC[i - 1] - bess[i] / 60)
                    new_load.append(H_op)
            elif data['PwrTOT_rel'].iloc[i] > L_op:
                bess.append(0)
                bess_SOC.append(bess_SOC[i - 1])
                new_load.append(data['PwrTOT_rel'].iloc[i])
            else:
                if not bess_full:
                    bess.append((data['PwrTOT_rel'].iloc[i] - L_op) * 410)
                    bess_SOC.append(bess_SOC[i - 1] - bess[i] / 60)
                    new_load.append(L_op)
                else:
                    bess.append(0)
                    bess_SOC.append(bess_SOC[i - 1])
                    new_load.append(data['PwrTOT_rel'].iloc[i])
        else:
            bess.append(0)
            bess_SOC.append(0)
            new_load.append(data['PwrTOT_rel'].iloc[i])
            if data['PwrTOT_rel'].iloc[i] > 0.4:
                first_ramp = False

        if bess_SOC[i] >= bess_max:
            bess_full = True
            bess_SOC[i] = bess_max
        else:
            bess_full = False
        if bess_SOC[i] <= 0:
            bess_empty = True
            bess_SOC[i] = 0
        else:
            bess_empty = False

    grad = np.gradient(new_load)
    data_new = pd.DataFrame()
    data_new['PwrTOT_rel'] = new_load
    data_new['Grad_PwrTOT_rel'] = grad
    data_new['BESS_pwr'] = bess
    data_new['BESS_SOC'] = bess_SOC
    data_new['PwrTOT'] = data_new['PwrTOT_rel'] * p_nom

    return data_new


def simulation(data, poi, bess_size):
    import numpy as np
    p_nom = 410
    last_processed = -1
    BESS_pwr = []
    BESS_SOC = []
    Plant_pwr_mod = []
    Nom_pwr = 410
    BESS_size = bess_size # MW
    lp_treshold = 0.65
    BESS_state = 'Empty'

    poi = np.array(poi)
    if poi.size == 4:
        poi = np.expand_dims(poi, axis=0)
    for riga in range(len(poi)):
        lp_start = poi[riga][0]
        lp_end = poi[riga][1]
        hp_start = poi[riga][2]
        hp_end = poi[riga][3]

        print(lp_start)
        print(lp_end)
        print(hp_start)
        print(hp_end)

        lp_avg = np.average(data['PwrTOT_rel'].iloc[lp_start: lp_end].values) * Nom_pwr # MW
        # lp_avg = 0.5
        print(lp_avg)
        lp_time = (lp_end - lp_start)  # min
        hp_time = (hp_end - hp_start)  # min
        hp_avg = np.average(data['PwrTOT_rel'].iloc[hp_start: hp_end].values) * Nom_pwr # MW

        if not BESS_SOC:
            BESS_avg_lp_pwr = (BESS_size - 0) * .95 / (lp_time / 60) # MW
        else:
            BESS_avg_lp_pwr = (BESS_size - BESS_SOC[-1]) * .95 / (lp_time / 60) # MW
            print(BESS_avg_lp_pwr)

        if BESS_avg_lp_pwr + lp_avg > lp_treshold * Nom_pwr:
            BESS_avg_lp_pwr = lp_treshold * Nom_pwr - lp_avg # MW
            BESS_capacity_est = BESS_avg_lp_pwr * (lp_time / 60)  # MWh
            BESS_avg_hp_pwr = BESS_capacity_est / (hp_time / 60)  # MW
        else:
            if not BESS_SOC:
                BESS_avg_hp_pwr = BESS_size / (hp_time / 60)  # MW
            else:
                BESS_avg_hp_pwr = BESS_size / (hp_time / 60)  # MW
        lp_pwr_setting = (lp_avg + BESS_avg_lp_pwr) / 410  # %
        print(lp_pwr_setting)
        hp_pwr_setting = (hp_avg - BESS_avg_hp_pwr) / 410  # %
        print(hp_pwr_setting)
        for i in range(last_processed + 1, lp_start):
            Power_demand = data['PwrTOT_rel'].iloc[i]
            Plant_pwr_mod.append(Power_demand)
            BESS_pwr.append(0)
            if not BESS_SOC:
                BESS_SOC.append(0)
            else:
                BESS_SOC.append(BESS_SOC[-1] - BESS_pwr[-1] / 60)

            if BESS_SOC[-1] >= BESS_size:
                BESS_SOC[-1] = BESS_size
                BESS_state = 'Full'
            elif BESS_SOC[-1] <= 0:
                BESS_SOC[-1] = 0
                BESS_state = 'Empty'
            else:
                BESS_state = 'Normal'

        for i in range(lp_start, lp_end):
            Power_demand = data['PwrTOT_rel'].iloc[i]
            if Power_demand < lp_pwr_setting:
                if BESS_state != 'Full':
                    BESS_pwr.append((Power_demand - lp_pwr_setting) * Nom_pwr)
                    Plant_pwr_mod.append(lp_pwr_setting)
                else:
                    BESS_pwr.append(0)
                    Plant_pwr_mod.append(Power_demand)
            else:
                if BESS_state != 'Empty':
                    BESS_pwr.append((Power_demand - lp_pwr_setting) * Nom_pwr)
                    Plant_pwr_mod.append(lp_pwr_setting)
                else:
                    BESS_pwr.append(0)
                    Plant_pwr_mod.append(Power_demand)

            BESS_SOC.append(BESS_SOC[-1] - BESS_pwr[-1] / 60)

            if BESS_SOC[-1] >= BESS_size:
                BESS_SOC[-1] = BESS_size
                BESS_state = 'Full'
            elif BESS_SOC[-1] <= 0:
                BESS_SOC[-1] = 0
                BESS_state = 'Empty'
            else:
                BESS_state = 'Normal'

        for i in range(lp_end, hp_start):
            Power_demand = data['PwrTOT_rel'].iloc[i]
            if Power_demand < lp_pwr_setting:
                if BESS_state != 'Full':
                    BESS_pwr.append((Power_demand - lp_pwr_setting) * Nom_pwr)
                    Plant_pwr_mod.append(lp_pwr_setting)
                else:
                    BESS_pwr.append(0)
                    Plant_pwr_mod.append(Power_demand)
            elif Power_demand < hp_pwr_setting:
                BESS_pwr.append(0)
                Plant_pwr_mod.append(Power_demand)
            else:
                if BESS_state != 'Empty':
                    BESS_pwr.append(Power_demand - hp_pwr_setting)
                    Plant_pwr_mod.append(hp_pwr_setting)
                else:
                    BESS_pwr.append(0)
                    Plant_pwr_mod.append(Power_demand)
            BESS_SOC.append(BESS_SOC[-1] - BESS_pwr[-1] / 60)

            if BESS_SOC[-1] >= BESS_size:
                BESS_SOC[-1] = BESS_size
                BESS_state = 'Full'
            elif BESS_SOC[-1] <= 0:
                BESS_SOC[-1] = 0
                BESS_state = 'Empty'
            else:
                BESS_state = 'Normal'

        for i in range(hp_start, hp_end + 1):
            Power_demand = data['PwrTOT_rel'].iloc[i]
            if Power_demand > hp_pwr_setting:
                if BESS_state != 'Empty':
                    BESS_pwr.append((Power_demand - hp_pwr_setting) * Nom_pwr)
                    Plant_pwr_mod.append(hp_pwr_setting)
                else:
                    BESS_pwr.append(0)
                    Plant_pwr_mod.append(Power_demand)
            else:
                if BESS_state != 'Full':
                    BESS_pwr.append((Power_demand - hp_pwr_setting) * Nom_pwr)
                    Plant_pwr_mod.append(hp_pwr_setting)
                else:
                    BESS_pwr.append(0)
                    Plant_pwr_mod.append(Power_demand)

            BESS_SOC.append(BESS_SOC[-1] - BESS_pwr[-1] / 60)

            if BESS_SOC[-1] >= BESS_size:
                BESS_SOC[-1] = BESS_size
                BESS_state = 'Full'
            elif BESS_SOC[-1] <= 0:
                BESS_SOC[-1] = 0
                BESS_state = 'Empty'
            else:
                BESS_state = 'Normal'

        last_processed = hp_end

    if last_processed != data.shape[0] - 1:
        for i in range(last_processed + 1, data.shape[0]):
            Power_demand = data['PwrTOT_rel'].iloc[i]
            BESS_pwr.append(0)
            Plant_pwr_mod.append(Power_demand)
            BESS_SOC.append(BESS_SOC[-1] - BESS_pwr[-1] / 60)

    grad = np.gradient(Plant_pwr_mod)
    data_new = pd.DataFrame()
    data_new['PwrTOT_rel'] = Plant_pwr_mod
    data_new['Grad_PwrTOT_rel'] = grad
    data_new['BESS_pwr'] = BESS_pwr
    data_new['BESS_SOC'] = BESS_SOC
    data_new['PwrTOT'] = data_new['PwrTOT_rel'] * p_nom

    return data_new

def simulation_old(data, poi):
    import numpy as np
    last_processed = -1
    BESS_pwr = []
    BESS_SOC = []
    Plant_pwr_mod = []
    Nom_pwr = 410
    BESS_size = 200 # MW
    lp_treshold = 0.65
    BESS_state = 'Empty'

    for indexes in poi:
        lp_start = indexes[0]
        lp_end = indexes[1]
        hp_start = indexes[2]
        hp_end = indexes[3]

        lp_avg = np.average(data['PwrTOT_rel'].iloc[lp_start: lp_end].values) * Nom_pwr # MW
        lp_time = (lp_end - lp_start)  # min
        hp_time = (hp_end - hp_start)  # min
        hp_avg = np.average(data['PwrTOT_rel'].iloc[hp_start: hp_end].values) * Nom_pwr # MW

        if not BESS_SOC:
            BESS_avg_lp_pwr = (BESS_size - 0) * .95 / (lp_time / 60) # MW
        else:
            BESS_avg_lp_pwr = (BESS_size - BESS_SOC[-1]) * .95 / (lp_time / 60) # MW

        if BESS_avg_lp_pwr + lp_avg > lp_treshold * Nom_pwr:
            BESS_avg_lp_pwr = lp_treshold * Nom_pwr - lp_avg # MW
            BESS_capacity_est = BESS_avg_lp_pwr * (lp_time / 60)  # MWh
            BESS_avg_hp_pwr = BESS_capacity_est / (hp_time / 60)  # MW
        else:
            if not BESS_SOC:
                BESS_avg_hp_pwr = BESS_size / (hp_time / 60)  # MW
            else:
                BESS_avg_hp_pwr = BESS_size / (hp_time / 60)  # MW
        lp_pwr_setting = (lp_avg + BESS_avg_lp_pwr) / 410  # %
        hp_pwr_setting = (hp_avg - BESS_avg_hp_pwr) / 410  # %
        for i in range(last_processed + 1, lp_start):
            Power_demand = data['PwrTOT_rel'].iloc[i]
            Plant_pwr_mod.append(Power_demand)
            BESS_pwr.append(0)
            if not BESS_SOC:
                BESS_SOC.append(0)
            else:
                BESS_SOC.append(BESS_SOC[-1] - BESS_pwr[-1] / 60)

            if BESS_SOC[-1] >= BESS_size:
                BESS_SOC[-1] = BESS_size
                BESS_state = 'Full'
            elif BESS_SOC[-1] <= 0:
                BESS_SOC[-1] = 0
                BESS_state = 'Empty'
            else:
                BESS_state = 'Normal'

        for i in range(lp_start, lp_end):
            Power_demand = data['PwrTOT_rel'].iloc[i]
            if Power_demand < lp_pwr_setting:
                if BESS_state != 'Full':
                    BESS_pwr.append((Power_demand - lp_pwr_setting) * Nom_pwr)
                    Plant_pwr_mod.append(lp_pwr_setting)
                else:
                    BESS_pwr.append(0)
                    Plant_pwr_mod.append(Power_demand)
            else:
                if BESS_state != 'Empty':
                    BESS_pwr.append((Power_demand - lp_pwr_setting) * Nom_pwr)
                    Plant_pwr_mod.append(lp_pwr_setting)
                else:
                    BESS_pwr.append(0)
                    Plant_pwr_mod.append(Power_demand)

            BESS_SOC.append(BESS_SOC[-1] - BESS_pwr[-1] / 60)

            if BESS_SOC[-1] >= BESS_size:
                BESS_SOC[-1] = BESS_size
                BESS_state = 'Full'
            elif BESS_SOC[-1] <= 0:
                BESS_SOC[-1] = 0
                BESS_state = 'Empty'
            else:
                BESS_state = 'Normal'

        for i in range(lp_end, hp_start):
            Power_demand = data['PwrTOT_rel'].iloc[i]
            if Power_demand < lp_pwr_setting:
                if BESS_state != 'Full':
                    BESS_pwr.append((Power_demand - lp_pwr_setting) * Nom_pwr)
                    Plant_pwr_mod.append(lp_pwr_setting)
                else:
                    BESS_pwr.append(0)
                    Plant_pwr_mod.append(Power_demand)
            elif Power_demand < hp_pwr_setting:
                BESS_pwr.append(0)
                Plant_pwr_mod.append(Power_demand)
            else:
                if BESS_state != 'Empty':
                    BESS_pwr.append(Power_demand - hp_pwr_setting)
                    Plant_pwr_mod.append(hp_pwr_setting)
                else:
                    BESS_pwr.append(0)
                    Plant_pwr_mod.append(Power_demand)
            BESS_SOC.append(BESS_SOC[-1] - BESS_pwr[-1] / 60)

            if BESS_SOC[-1] >= BESS_size:
                BESS_SOC[-1] = BESS_size
                BESS_state = 'Full'
            elif BESS_SOC[-1] <= 0:
                BESS_SOC[-1] = 0
                BESS_state = 'Empty'
            else:
                BESS_state = 'Normal'

        for i in range(hp_start, hp_end + 1):
            Power_demand = data['PwrTOT_rel'].iloc[i]
            if Power_demand > hp_pwr_setting:
                if BESS_state != 'Empty':
                    BESS_pwr.append((Power_demand - hp_pwr_setting) * Nom_pwr)
                    Plant_pwr_mod.append(hp_pwr_setting)
                else:
                    BESS_pwr.append(0)
                    Plant_pwr_mod.append(Power_demand)
            else:
                if BESS_state != 'Full':
                    BESS_pwr.append((Power_demand - hp_pwr_setting) * Nom_pwr)
                    Plant_pwr_mod.append(hp_pwr_setting)
                else:
                    BESS_pwr.append(0)
                    Plant_pwr_mod.append(Power_demand)

            BESS_SOC.append(BESS_SOC[-1] - BESS_pwr[-1] / 60)

            if BESS_SOC[-1] >= BESS_size:
                BESS_SOC[-1] = BESS_size
                BESS_state = 'Full'
            elif BESS_SOC[-1] <= 0:
                BESS_SOC[-1] = 0
                BESS_state = 'Empty'
            else:
                BESS_state = 'Normal'

        last_processed = hp_end

    if last_processed != data.shape[0] - 1:
        for i in range(last_processed + 1, data.shape[0]):
            Power_demand = data['PwrTOT_rel'].iloc[i]
            BESS_pwr.append(0)
            Plant_pwr_mod.append(Power_demand)
            BESS_SOC.append(BESS_SOC[-1] - BESS_pwr[-1] / 60)

    grad = np.gradient(Plant_pwr_mod)
    data_new = pd.DataFrame()
    data_new['PwrTOT_rel'] = Plant_pwr_mod
    data_new['Grad_PwrTOT_rel'] = grad
    data_new['BESS_pwr'] = BESS_pwr
    data_new['BESS_SOC'] = BESS_SOC
    data_new['PwrTOT'] = data_new['PwrTOT_rel'] * Nom_pwr
    return data_new


def add_eta(data, model_in):

    from cust_libs.modeling import Load_ML_Model, Load_Poly_model
    import pandas as pd
    import numpy as np
    import sys
    if model_in == 'RF':
        model_path = 'models/multivariate/ML/RND_FOR/RND_FOR_02.joblib'
    elif model_in == 'ALL':
        model_path = 'models/multivariate/ML/ALL/ALL_02.joblib'
    elif model_in == 'MLP':
        model_path = 'models/multivariate/ML/MLP/MLP_02.joblib'
    elif model_in == 'SVR':
        model_path = 'models/multivariate/ML/SVR/SVR_01.joblib'
    elif model_in == 'mono':
        model_path = 'models/univariate/Poly/deg4_low'
    elif model_in == 'mono_mod':
        model_path = 'models/univariate/Poly/deg4_low'
    else:
        sys.exit("Errore")

    if model_in == ('RF' or 'ALL' or 'MLP' or 'SVR'):
        model, _ = Load_ML_Model(model_path)
        x = data[['PwrTOT_rel', 'Grad_PwrTOT_rel']].values
        eta = model.predict(x)
    elif model_in == 'mono' or model_in == 'mono_mod':
        model, transformer = Load_Poly_model(model_path)
        x = data['PwrTOT_rel'].values.reshape(-1, 1)
        eta = model.predict(transformer.transform(x))
        if model_in == 'mono_mod':
            eta = np.array(eta)
            eta = eta + 0.05
    else:
        sys.exit("Errore")

    data['Rendimento'] = eta
    data['Pwr_in'] = np.divide(data['PwrTOT'], data['Rendimento'])

    return data


def plot_coatto(data_bess, data_nobess, delta_eta=None, BESS_size=None):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(2, 2, figsize=(16, 9))  # 20 10

    ax[0][0].plot(data_nobess['PwrTOT_rel'].values, color='b', alpha=0.3, label='Impianto senza BESS')
    ax[0][0].plot(data_bess['PwrTOT_rel'].values, color='r', label='Impianto con BESS')

    ax[1][0].plot(data_nobess['Rendimento'].values, color='b', alpha=0.3, label='Impianto senza BESS')
    ax[1][0].plot(data_bess['Rendimento'].values, color='r', label='Impianto con BESS')
    # ax[0][1].plot(BESS_SOC, label='SOC')
    ax[0][1].plot(data_bess['BESS_pwr'].values, label='Potenza BESS', color='r')
    ax[0][1].set_ylabel('Potenza [MW]')

    ax[1][1].plot(data_bess['BESS_SOC'].values, color='r')
    if BESS_size is not None:
        ax[1][1].set_ylabel('SOC [%]')
        locs = ax[1][1].get_yticks()
        ax[1][1].set_yticks(locs, np.round(locs / BESS_size * 100, 1))
    else:
        ax[1][1].set_ylabel('SOC [MW]')

    ax[0, 0].set_title("Curva di potenza dell'impianto")
    ax[0, 1].set_title("Curva di potenza BESS")
    ax[1, 0].set_title("Curva di rendimento dell'impianto")
    ax[1, 1].set_title("Stato di carica del BESS")

    ax[0][0].legend(loc="best")

    if delta_eta is not None:
        y_text = ax[1][0].get_yticks()[2] + 0.001

        ax[1][0].text(len(data_bess) * 0.5, y_text,
                      r'$ \Delta \eta$ = +' "{:.2f}".format(delta_eta * 100) + ' %',
                      color='r', ha='center', fontsize=16)
        print(y_text)
        print(ax[1][0].get_yticks()[1])
    ax[1][0].legend(loc="best")
    # fig.subplots_adjust(top=0.93)
    plt.show()
