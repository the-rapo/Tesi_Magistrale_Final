"""
##### DATA PROCESSING V2 #####
In questa libreria sono presenti una serie di funzioni relative all'analisi dati, divise nelle sezioni:
    - PREPROCESSING
    - ANALYSIS
"""
from pandas.core.frame import DataFrame

# PREPROCESSING


def terna_preprox(file: str, folderout: str = 'IO', verbose: bool = False) -> None:
    """
        Effettua il pre-processo dei dati Terna.

        :param folderout: Cartella di output
        :param file: File da processare
        :param verbose: Flag per console-output

    """

    # LIBs
    import pandas as pd
    from pathlib import Path
    from cust_libs.misc import mk_dir
    #

    filename = Path(file).stem
    # extension = Path(file).suffix
    print('---------- ' 'Dataset ' + filename + 'PrePROX' + ' ---------- ')
    data = pd.read_excel(file, engine="openpyxl", sheet_name=0)
    data.drop('Forecast Total load [MW]', inplace=True, axis=1)
    data.drop('Bidding zone', inplace=True, axis=1)
    data.rename(columns={'Date': 'DateTime', 'Total Load [MW]': 'Terna_Load'}, inplace=True)
    data['DateTime'] = pd.to_datetime(data['DateTime'], errors='coerce')
    data = data.dropna(subset=['DateTime'])

    # Max_Daily
    rel_load_daily = []
    day_start = 0
    for i in range(1, data.shape[0], 1):
        if data['DateTime'].iloc[i].day != data['DateTime'].iloc[i - 1].day:
            max_day = max(data['Terna_Load'].iloc[day_start: i])
            rel_load_daily[day_start: i] = data['Terna_Load'].iloc[day_start: i] / max_day
            day_start = i
        if i == data.shape[0] - 1:
            max_day = max(data['Terna_Load'].iloc[day_start:])
            rel_load_daily[day_start: i + 1] = data['Terna_Load'].iloc[day_start:] / max_day

    data['Terna_Load_Rel'] = rel_load_daily

    mk_dir(folderout, False, verbose)
    mk_dir(folderout + '/Terna_Load', False, verbose)
    if verbose:
        print('Salvando ' + folderout + '/Terna_Load/' + filename + '_Processed.csv')
    data.to_csv(folderout + '/Terna_Load/' + filename + '_Processed.csv', index=False)

    return


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


def split_ignitions(file: str, folderout: str = 'IO', verbose: bool = False) -> None:
    """
        Suddivide il database processato da preprox nelle diverse accensioni

        :param folderout: Cartella di output
        :param file: File da processare
        :param verbose: Flag per console-output

    """

    # LIBs
    import pandas as pd
    from pathlib import Path
    from cust_libs.misc import mk_dir
    from alive_progress import alive_bar

    #

    data = pd.read_csv(file)
    filename = Path(file).stem
    plantname = filename.replace('_Processed', '')
    mk_dir(folderout, True, verbose)
    print('---------- ' + 'Splitting dataset ' + plantname + ' ----------')

    acc_no = data['noACC'].max()
    digit_no = len(str(acc_no))
    if verbose:
        print('Percorso database: ' + file)
        print('Rilevate ' + str(acc_no) + ' accensioni')
        print('Salvando i files in ' + folderout)
    # Splitting
    with alive_bar(int(acc_no), force_tty=True) as bar:
        for acc in range(acc_no):
            acc_name = plantname + '_Acc' + str(acc + 1).zfill(digit_no)
            data_acc = data[data['noACC'] == (acc + 1)]
            with pd.option_context('mode.chained_assignment', None):
                data_acc.drop('noACC', inplace=True, axis=1)
            data_acc.to_csv(folderout + '/' + acc_name + '.csv', index=False)
            bar()

    if verbose:
        print('Split eseguito correttamente')
    return


def remove_zeroes(file: str, folderout: str = 'IO', verbose: bool = False) -> None:
    """
            A partire dal database processato si ottiene un database senza gli zeri

            :param folderout: Cartella di output
            :param file: File da processare
            :param verbose: Flag per console-output

            """

    # LIBs
    import pandas as pd
    from pathlib import Path
    from cust_libs.misc import mk_dir
    #

    data = pd.read_csv(file)
    filename = Path(file).stem
    plantname = filename.replace('_Processed', '')
    print('---------- ' + 'Rimuovendo gli zeri da ' + plantname + ' ----------')

    mk_dir(folderout, False, verbose)

    data_on = data[data['noACC'] != 0]
    if verbose:
        print('Salvando il dataset in: ' + folderout)
    data_on.to_csv(folderout + '/' + plantname + '_Processed_ON' + '.csv', index=False)
    if verbose:
        print('remove_zeroes eseguito correttamente')
    return


# DATA_ANALISYS

def filter_data(data: DataFrame, grad_inf, grad_sup, rel_pw_inf, rel_pw_sup):
    """
                A partire dal df indicato ritorna un df filtarto secondo gradiente e potenza relativa

                :param data: df da filtrare
                :param grad_inf: limite inferiore gradiente di potenza
                :param grad_sup: limite superiore gradiente di potenza
                :param rel_pw_inf: limite inferiore potenza relativa
                :param rel_pw_sup: limite superiore potenza relativa
                """

    # LIBs
    #

    if grad_inf is not None or grad_sup is not None:
        if grad_inf is None:
            data_filt_1 = data[data.Grad_PwrTOT_rel < grad_sup]
        elif grad_sup is None:
            data_filt_1 = data[data.Grad_PwrTOT_rel > grad_inf]
        else:
            data_filt_1 = data[(data.Grad_PwrTOT_rel > grad_inf) & (data.Grad_PwrTOT_rel < grad_sup)]
    else:
        data_filt_1 = data

    if rel_pw_inf is not None and rel_pw_sup is not None:
        data_filt_2 = data_filt_1[(data_filt_1.PwrTOT_rel > rel_pw_inf) & (data_filt_1.PwrTOT_rel < rel_pw_sup)]
    else:
        data_filt_2 = data_filt_1

    return data_filt_2


def save_rampe(data_path, rampe):
    """
    Salva i punti d'inizio-fine alta-bassa potenza (POI) indicati nella matrice "rampe" la cui riga è così
    composta: [LOWP_START, LOWP_END, HIGHP_START, HIGHP_END]

    :param data_path: path df accensione
    :param rampe: matrice in cui sono indicati i POI
    """

    # LIBs
    from joblib import dump
    from cust_libs.misc import mk_dir
    from pathlib import Path
    #

    folder = Path(data_path).parent
    print('Analizzando' + Path(data_path).stem)
    rampe_folder = str(folder) + '/' + 'Rampe'
    rampe_name = Path(data_path).stem + '_Rampe.joblib'
    mk_dir(rampe_folder)
    dump(rampe, rampe_folder + '/' + rampe_name)
    return


def load_rampe(data_path):
    """
    Carica i punti d'inizio-fine alta-bassa potenza (POI) salvati precedentemente con save_rampe

    :param data_path: path df accensione
    """

    # LIBs
    from joblib import load
    from pathlib import Path
    #

    folder = Path(data_path).parent
    print('Analizzando' + Path(data_path).stem)
    rampe_folder = str(folder) + '/' + 'Rampe'
    rampe_name = Path(data_path).stem + '_Rampe.joblib'
    rampe = load(rampe_folder + '/' + rampe_name)
    return rampe


def compute(data, opt):

    import sys
    import numpy as np
    if opt == 'avg_eta':
        tot_energy_in = sum(data['Pwr_in'].values) / 60
        tot_energy_out = sum(data['PwrTOT'].values) / 60
        rendimento = tot_energy_out / tot_energy_in
        return rendimento

    elif opt == 'avg_eta_t':
        rendimento_t = np.average(data['Rendimento'].values)
        return rendimento_t

    elif opt == 'tot_en_out':
        tot_energy_out = sum(data['PwrTOT'].values) / 60
        return tot_energy_out
    elif opt == 'tot_en_in':
        tot_energy_in = sum(data['Pwr_in'].values) / 60
        return tot_energy_in

    else:
        sys.exit("Errore")
