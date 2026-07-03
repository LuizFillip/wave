import numpy as np

 
def fit_harmonic_phase(t, y, period):
    """
    Ajuste harmônico por mínimos quadrados:
        y(t) = A*cos(w t) + B*sin(w t) + C

    result : dict
        Dicionário com coeficientes, amplitude, fase e ajuste.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

 

    w = 2 * np.pi / period

    X = np.column_stack([
        np.cos(w * t),
        np.sin(w * t),
        np.ones_like(t)
    ])

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    A, B, C = beta

    amplitude = np.sqrt(A**2 + B**2)
    phase_rad = np.arctan2(B, A)

    # fase convertida para dias
    phase_days = (phase_rad / (2 * np.pi)) * period

    # opcional: colocar no intervalo [0, period)
    phase_days_mod = phase_days % period

    y_fit = X @ beta

    return {
        'A': A,
        'B': B,
        'C': C,
        'amplitude': amplitude,
        'phase_rad': phase_rad,
        'phase_days': phase_days,
        'phase_days_mod': phase_days_mod,
        'y_fit': y_fit,
        't_fit': t
    }

 
def vertical_phase_speed(alt_km, phase_days):
    """
    Estima a velocidade vertical de fase a partir de um ajuste linear
    fase (dias) versus altitude (km).

    """
    alt_km = np.asarray(alt_km, dtype=float)
    phase_days = np.asarray(phase_days, dtype=float)
 
    slope, intercept = np.polyfit(alt_km, phase_days, 1)

    vz_km_day = 1.0 / slope
    vz_m_s = vz_km_day * 1000.0 / 86400.0

    phase_fit = slope * alt_km + intercept

    return {
        'slope_days_per_km': slope,
        'intercept_days': intercept,
        'vz_km_day': vz_km_day,
        'vz_m_s': vz_m_s,
        'phase_fit': phase_fit
    }


def unwrap_phase_days(phase_days, period):
    phase_days = np.asarray(phase_days, dtype=float).copy()

    for i in range(1, len(phase_days)):
        while phase_days[i] - phase_days[i-1] > period/2:
            phase_days[i] -= period
        while phase_days[i] - phase_days[i-1] < -period/2:
            phase_days[i] += period

    return phase_days




def get_phases_mod(ds, period = 13, unwrap = True):
   
    phases = []
    altitudes = ds.columns
   
    for col in altitudes:
        fit = fit_harmonic_phase(ds.index, ds[col], period)
        phases.append(fit['phase_days_mod'])
    if unwrap:
        return unwrap_phase_days(phases, period= period)
    else:
        return phases 

 

