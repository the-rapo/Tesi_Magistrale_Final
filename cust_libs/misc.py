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


