import SABER as sb 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import base as b 


 
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



def join_heights(df_main, alts, ref_lon = -36):
    out = []
    for alt in alts:
        df =  sb.saber_pivot_longitude(
            df_main, 
            alt = alt, 
            lat_min = -10,
            lat_max = 0,
            res = 1,
            sigma = (6, 1),
            norm = True
            )
         
        ds = df.loc[df.index == ref_lon].T
        ds = ds.rename(columns = {ref_lon: alt})
        out.append(ds)
        
    return pd.concat(out, axis = 1)

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

 

def plot_heights_vs_phase(phases, alts, res, period):
    
    slope = res['slope_days_per_km']
    vz_km_day = res['vz_km_day']
    vz_ms = res['vz_m_s']
    lambda_z = res['vz_km_day'] * period
 
     
    infos = (f'dt/dz = {slope:.2f} days/km\n' +
             f'Vz = {vz_km_day:.2f} km/day\n' + 
             f'Vz = {vz_ms:.2f} m/s\n' + 
             f'$\lambda_z$ = {lambda_z:.2f} km')
    
    b.sci_format()
    
    fig, ax = plt.subplots(
        
        figsize= (9, 6), 
        dpi = 300
        )
    ax.plot(phases, alts, 'o')
    ax.plot( res['phase_fit'], alts, '-',
            )
    ax.set(
           ylabel = 'Altitude (km)', 
           xlabel = 'Phase (days)', 
           ylim = [0, 110]
           )
    
    ax.text(
        0.55, 0.1, infos, 
            fontsize = 20,
            transform = ax.transAxes)
   
fn = 'saber_all_alts'
# fn = 'saber_test2'
df = sb.saber_data(fn)
 
alts = np.arange(20, 110, 10)
# alts = [30, 60, 90]

#%%%%
ds = join_heights(df, alts, ref_lon = -36)
 
ds = ds.loc[(ds.index >= 40) & (ds.index <= 80)] 
 
period = 13
 

phases = get_phases_mod(ds, period = period)


res = vertical_phase_speed(alts, phases)


# plot_heights_vs_phase(phases, alts, res, period)

 

fig, ax = plt.subplots(figsize=(12, 4))

cols = ds.columns
colors = plt.cm.get_cmap('turbo', len(cols))
offset = 0.5
for i, col in enumerate(cols):
    y = ds[col].values + i * offset
    ax.plot(
        ds.index, y, lw=2, 
        color=colors(i), 
        label=col
        )
ds 