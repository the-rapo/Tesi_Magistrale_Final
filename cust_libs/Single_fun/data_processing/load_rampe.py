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
