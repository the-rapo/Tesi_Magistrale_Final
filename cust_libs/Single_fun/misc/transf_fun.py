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
