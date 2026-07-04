import SABER as sb
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import waves as wv
from tqdm import tqdm 

def join_heights(
        df_main, alts,
        bandpass = (3, 15),
        lat_center = -7, ref_lon = -35):
    out = []
    desc = 'Selecting Layers'
    for alt in tqdm(alts, desc):
        df = sb.box_groupy_process(
                df_main, 
                f'temp_{alt}', 
                lat_center = lat_center,
                bandpass = bandpass,
               
                )
        ds = df.loc[df.index == ref_lon].T
        ds = ds.rename(columns = {ref_lon: alt})
        out.append(ds)
        
    return pd.concat(out, axis = 1)


 


def get_phases_mod(ds, period = 13, unwrap = True):
   
    phases = []
    altitudes = ds.columns
   
    for col in altitudes:
        fit = wv.fit_harmonic_phase(ds.index, ds[col], period)
        phases.append(fit['phase_days_mod'])
        
    if unwrap:
        return wv.unwrap_phase_days(phases, period= period)
    else:
        return phases 

def plot_heights_vs_phase(
        ax, phase_days, altitudes, period):
    res = wv.vertical_phase_parameters(altitudes, phase_days, period)

    vz_ms = res['vz_m_s']
    lambda_z = res['vz_km_day'] * period
    
    infos = (
             f'T = {period} days\n' + 
             f'Vz = {vz_ms:.2f} m/s\n' + 
             f'$\lambda_z$ = {lambda_z:.2f} km')
    

   
    ax.scatter(phase_days, altitudes, marker = 'o', s = 100)
    ax.plot( res['phase_fit'], altitudes, '-', lw = 3, color = 'red')
    ax.set(
           ylabel = 'Altitude (km)', 
           xlabel = 'Phase (days)', 
           ylim = [0, 110],
           xlim = [-4, 8]
           )
    
    ax.text(
        0.45, 0.01, 
        infos, 
        fontsize = 20,
        transform = ax.transAxes
        )

df_main = sb.saber_data( )
 
alts = np.arange(20, 110, 10)
 
ds_all = join_heights(df_main, alts, ref_lon = -35)
 

#%%%

fig, ax = plt.subplots(
     ncols = 2,
     sharey = True,
     figsize= (12, 10), 
     dpi = 300
     )

ds = ds_all.loc[
    (ds_all.index >= 50) & 
    (ds_all.index <= 100)
    ].copy()

 
period = 5
 
phases = get_phases_mod(ds, period = period)

plot_heights_vs_phase(ax[0], phases, alts, period)

def plot_fluctuations_by_height(ax, ds):
    for alt in ds.columns:
        ax.plot(ds[alt] + alt)
    
    ax.set(xlabel = 'Day of year')
        
    
plot_fluctuations_by_height(ax[1], ds)