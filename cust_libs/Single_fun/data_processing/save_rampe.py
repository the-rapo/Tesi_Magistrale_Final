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
