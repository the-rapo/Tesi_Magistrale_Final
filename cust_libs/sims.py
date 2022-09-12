def simple_ramp(low_p, high_p, t_lowp, t_highp, t_ramp):
    import numpy as np
    import pandas as pd

    p_nom = 410

    low_p_list = np.linspace(low_p, low_p, t_lowp , endpoint=True)
    ramp_list = np.linspace(low_p, high_p, t_ramp, endpoint=True)
    high_p_list = np.linspace(high_p, high_p, t_highp , endpoint=True)

    PwrTOT_rel = []
    PwrTOT_rel.extend(low_p_list)
    PwrTOT_rel.extend(ramp_list)
    PwrTOT_rel.extend(high_p_list)

    grad = np.gradient(PwrTOT_rel)
    data = pd.DataFrame()
    data['PwrTOT_rel'] = PwrTOT_rel
    data['Grad_PwrTOT_rel'] = grad
    data['PwrTOT'] = data['PwrTOT_rel'] * p_nom
    return data

