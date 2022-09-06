def simple_ramp(low_p, high_p, t_lowp, t_highp, t_ramp):
    import numpy as np

    low_p_list = np.linspace(low_p, low_p, t_lowp - 1)
    ramp_list = np.linspace(low_p, high_p, t_ramp)
    grad_ramp = ramp_list[1] - ramp_list[0]
    grad_ramp_list = np.linspace(grad_ramp, grad_ramp, t_ramp)
    ramp = np.array([ramp_list, grad_ramp_list])
    ramp = np.transpose(ramp)
    high_p_list = np.linspace(high_p, high_p, t_highp - 1)

    PwrTOT_rel = []
    PwrTOT_rel.extend(low_p_list)
    PwrTOT_rel.extend(ramp_list)
    PwrTOT_rel.extend(high_p_list)

    return PwrTOT_rel
