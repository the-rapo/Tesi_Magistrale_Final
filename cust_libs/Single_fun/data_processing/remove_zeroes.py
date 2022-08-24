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
