def preprox(file: str, powertg: int, powertv: int, load_folder: str, folderout: str = 'IO/', met: bool = True,
            on_tresh: float = 0.35, pot_cal_met: int = 36400, drop: bool = False, verbose: bool = False) -> None:
    """
    Effettua il pre-processo dei dati.

    :param folderout: Cartella di output
    :param file: File da processare
    :param powertg: Potenza Gas Totale
    :param powertv: Potenza Vapore Totale
    :param on_tresh: Valore minimo per cui l'impianto risulta "acceso"
    :param pot_cal_met: Potere calorifico metano stm^3/h
    :param met: Flag per indicare la presenza della portata metano in ingresso all'impianto
    :param load_folder: Carella in cui si trovano i carichi Terna
    :param verbose: Flag per console-output
    :param drop: Flag per pulire le colonne relative alla frequenza di rete e alla portata del gas
    """

    # LIBs
    import pandas as pd
    from pathlib import Path
    from cust_libs.misc import mk_dir
    import numpy as np
    #

    data = pd.read_csv(file)
    filename = Path(file).stem
    plantname = filename.replace('_Raw', '')
    print('---------- ' 'Dataset ' + plantname + 'PrePROX' + ' ---------- ')

    # Datetime
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    sampling = data['DateTime'].iloc[1] - data['DateTime'].iloc[0]
    sampling = sampling.total_seconds()  # Sono secondi
    if verbose:
        print('Modalità DEBUG')
        print('--- Setting Generali')
        print('Output directory:' + folderout)
        print('Terna directory: ' + load_folder)
        if met:
            print('Analisi Rendimento: ON')
        else:
            print('Analisi Rendimento: OFF')
        print('Valore Soglia ON: ' + str(on_tresh))

        print('--- Specifiche impianto')
        print('Potenza TG: ' + str(powertg))
        print('Potenza TV: ' + str(powertv))
        print('Sampling: ' + str(sampling) + ' s')
        print('Pot. calorifico metano: ' + str(pot_cal_met) + ' KJ / stdm^3 h')

    data_start = data['DateTime'].iloc[0]
    data_end = data['DateTime'].iloc[-1]
    timespan = data_end - data_start
    data = data.set_index('DateTime')
    data = data.sort_values('DateTime')
    # Total Power
    tg = []
    tv = []
    for c in data.columns:
        if "Pwr_TG" in c:
            tg.append(c)
        if "Pwr_TV" in c:
            tv.append(c)
    data['PwrTOT'] = 0
    for v in tv:
        data['PwrTOT'] += data[v]
    for g in tg:
        data['PwrTOT'] += data[g]

    # Relative Power & Efficiency
    data['PwrTOT_rel'] = data['PwrTOT'] / (powertg + powertv)
    data['Grad_PwrTOT_rel'] = np.gradient(data['PwrTOT_rel']) * 60 / sampling   # Equivalente di MW / min
    if met:
        data['Pwr_in'] = data['Port_Met'] * pot_cal_met / 3600 / 1000
        data['Rendimento'] = data['PwrTOT'] / (data['Pwr_in'] + 1.e-30)

    # Terna_Load
    year = data.index[0].year
    load = pd.read_csv(load_folder + '/Terna_' + str(year) + '_North_Processed.csv')
    load['DateTime'] = pd.to_datetime(load['DateTime'])
    load = load.set_index('DateTime')
    load = load.sort_values('DateTime')
    data = pd.merge_asof(data, load, on="DateTime")
    # Labeling ignitions
    false_neg_tresh = 1200 / sampling  # 20 min
    data['isON'] = np.where(data['PwrTOT_rel'] > on_tresh, True, False)
    count = 0
    last_on = 0
    temp = []

    # Algorithm
    for i in range(data.shape[0]):
        if data['isON'].iloc[i]:  # È acceso
            if not data['isON'].iloc[i - 1]:  # Si è appena acceso
                if i - last_on < false_neg_tresh:  # Falso spegnimento
                    temp[last_on + 1: i] = [count] * (i - 1 - last_on)
                else:
                    count += 1
            temp.append(count)
            last_on = i
        else:
            temp.append(0)
    data['noACC'] = temp
    # Cleaning False Accensioni
    acc = 1
    while acc <= data['noACC'].max():
        if verbose:
            print('Valutando accensione ' + str(acc))
        data_acc = data[data['noACC'] == acc]
        if data_acc['PwrTOT_rel'].max() < 0.2:
            if verbose:
                print('Rilevata falsa accensione (no ' + str(acc) + ')')
            data['noACC'].replace(acc, 0, inplace=True)
            if verbose:
                print(data[data['noACC'] == acc])
            for acc_2 in range(acc + 1, data['noACC'].max() + 1, 1):
                data['noACC'].replace(acc_2, acc_2 - 1, inplace=True)
        else:
            acc += 1

    if verbose:
        if data['noACC'].max() == 1:
            print('Rilevata unica accensione in ' + str(timespan))
        else:
            print('Rilevate ' + str(data['noACC'].max()) + ' accensioni in ' + str(timespan))

    # Cleaning
    if drop:
        try:
            data.drop('Freq_Gener', inplace=True, axis=1)
            data.drop('Freq_Rete', inplace=True, axis=1)
            data.drop('Port_Met', inplace=True, axis=1)
        except KeyError:
            pass

    data.drop('isON', inplace=True, axis=1)

    # Saving Files
    plant_dir = folderout + '/' + plantname

    mk_dir(folderout, False, verbose)
    mk_dir(plant_dir, True,  verbose)

    if verbose:
        print('Salvando i Files')

    # Saving csv - processed
    data.to_csv(plant_dir + '/' + plantname + '_Processed.csv', index=False)

    if verbose:
        print('preprox eseguito correttamente')
    return
