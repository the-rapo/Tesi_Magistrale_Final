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
