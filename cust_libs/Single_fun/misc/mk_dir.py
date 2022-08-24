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
