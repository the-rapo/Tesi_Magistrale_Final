"""
##### MISC V2 #####
In questa libreria sono presenti una serie di funzioni generiche, ripetute più volte nel codice.
"""


def mk_dir(folder: str, clear: bool = False, verbose: bool = False) -> None:
    """
        Crea cartelle o ne elimina il contenuto se già presenti [opt.].

        :param folder: Percorso della cartella
        :param clear: Flag per indicare la completa sovrascrittura della cartella
        :param verbose: Flag per console-output

    """
    # LIBs
    import os
    import shutil
    #
    if not os.path.exists(folder):
        os.mkdir(folder)
        if verbose:
            print('Creando il percorso ' + folder)
    else:
        if verbose:
            print('Il percorso ' + folder + ' esiste')

    if clear and os.path.exists(folder):
        shutil.rmtree(folder)
        os.mkdir(folder)
        if verbose:
            print('Eliminando i contenuti di ' + folder)

    return


def folder_tree(folder: str, ext: str = '.*', from_root: bool = True, verbose: bool = False) -> list:
    """
               Riporta i file contenuti in una cartella come lista

               :param folder: Cartella da analizzare
               :param from_root: Riporta il percorso completo del singolo file
               :param verbose: Flag per console-output
               :param ext: Estensione dei file da riportare

    """
    # LIBs
    from os import listdir
    from os.path import isfile, join
    from pathlib import Path
    #
    # extension = Path(file).suffix

    if verbose:
        print('Analizzando il contenuto di ' + folder)
    onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
    if from_root:
        onlyfiles = [folder + '/' + f for f in onlyfiles]

    if ext != '.*':
        if verbose:
            print('Riportando i files con estensione ' + ext)
        for i in onlyfiles:
            extension = Path(i).suffix
            if extension != ext:
                if verbose:
                    print('Rimuovendo ' + i)
                onlyfiles.remove(i)

    if verbose:
        print('-- Contenuto di ' + folder)
        print(*onlyfiles, sep="\n")

    return onlyfiles


def transf_fun(par) -> any:
    """
    Definisce le funzioni di trasferimento tra potenza e potenza relativa

    :param par: Potenza Nominale impianto (se df la calcola in automatico)

    """
    # LIBs
    import pandas as pd
    #

    if isinstance(par, pd.DataFrame):
        nom_pwr = par['PwrTOT'].iloc[100] / par['PwrTOT_rel'].iloc[100]
    elif par is not (int or float):
        print('Errore in tranf_fun')
    else:
        nom_pwr = par

    def rel2tot(x):
        return nom_pwr * x

    def tot2rel(x):
        return x / nom_pwr

    return rel2tot, tot2rel


def Train_Data_Split(acc_fldr: str, train_perc: float, out_long: bool = False, seed: int = None,
                     verbose: bool = True) -> tuple:
    """
                  Divide le accensioni in Train/Test per ML

                  :param acc_fldr: Cartella in cui si trovano le accensioni
                  :param train_perc: Percentuale di training
                  :param verbose: Flag per console-output
                  :param out_long: Riporta in output anche il dataset intero
                  :param seed: Random seed

                  """
    # LIBs
    import random
    import pandas as pd
    from pathlib import Path

    #
    if seed is not None:
        random.seed(seed)

    files = folder_tree(acc_fldr, verbose=verbose)
    filename = Path(files[0]).stem
    plantname = filename.replace('_Acc01', '')
    print('---------- ' + 'Train/Test Split Database ' + plantname + ' ----------')

    if verbose:
        print('Modalità DEBUG')
        print('--- Setting Generali')
        print('Ignition directory:' + acc_fldr)
        print('% Training: ' + str(train_perc))
        print('Output Completo: ' + str(out_long))
        if seed is None:
            print('Random Seed Casuale')
        else:
            print('Random Seed : ' + str(seed))

    tot_data = 0

    for f in files:
        tot_data += len(pd.read_csv(f))

    train_cutoff = train_perc * tot_data

    order = list(range(len(files)))
    random.shuffle(order)

    x_train = pd.DataFrame()
    x_test = pd.DataFrame()
    y_train = pd.DataFrame()
    y_test = pd.DataFrame()

    if verbose:
        print('--- Ordine accensioni')
        print(order)
        print('----------------------------')

    train_data = 0
    for f in order:
        temp = pd.read_csv(files[f])
        if train_data < train_cutoff:
            x_train = pd.concat([x_train, temp[['PwrTOT_rel', 'Grad_PwrTOT_rel']]], axis=0, sort=False,
                                ignore_index=True)
            y_train = pd.concat([y_train, temp['Rendimento']], axis=0, sort=False, ignore_index=True)
            train_data += len(temp)
        else:
            x_test = pd.concat([x_test, temp[['PwrTOT_rel', 'Grad_PwrTOT_rel']]], axis=0, sort=False, ignore_index=True)
            y_test = pd.concat([y_test, temp['Rendimento']], axis=0, sort=False, ignore_index=True)

    if verbose:
        print('--- Data properties')
        print('Data entries: ' + str(tot_data))
        print('Train entries: ' + str(len(x_train)))
        print('Test entries: ' + str(len(x_test)))
        print('Actual Train ratio: ' + str(len(x_train) / tot_data))

    x = pd.concat([x_train, x_test], axis=0, sort=False, ignore_index=True)
    y = pd.concat([y_train, y_test], axis=0, sort=False, ignore_index=True)
    x_train.columns = ['PwrTOT_rel', 'Grad_PwrTOT_rel']
    y_train.columns = ['Rendimento']
    x_test.columns = ['PwrTOT_rel', 'Grad_PwrTOT_rel']
    y_test.columns = ['Rendimento']
    x.columns = ['PwrTOT_rel', 'Grad_PwrTOT_rel']
    y.columns = ['Rendimento']

    x_train = x_train.values
    x_test = x_test.values
    y_train = y_train.values
    y_test = y_test.values
    y = y.values
    x = x.values

    if verbose:
        print('rand_data_training eseguito correttamente')

    if out_long:
        return x_train, y_train, x_test, y_test, x, y
    else:
        return x_train, y_train, x_test, y_test
