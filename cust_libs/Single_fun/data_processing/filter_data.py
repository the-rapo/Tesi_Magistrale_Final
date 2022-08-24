from pandas.core.frame import DataFrame


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
