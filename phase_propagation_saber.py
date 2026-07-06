import SABER as sb
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from tqdm import tqdm 
import waves as wv
import base as b 
b.sci_format()

def join_heights(
        df_main,
        alts,
        bandpass = (3, 15),
        lat_center = -7, 
        ref_lon = -35,
        lon_step = 20,
        lat_step = 20,
        normalize = True
        ):
    out = []
    desc = 'Selecting Layers'
    for alt in tqdm(alts, desc):
        df = sb.box_groupy_process(
                df_main, 
                f'temp_{alt}', 
                lat_center = lat_center,
                bandpass = bandpass,
                normalize = normalize,
                lon_stride = None, 
                lon_step = lon_step,
                lat_step = lat_step,
               
                )
        ds = df.loc[df.index == ref_lon].T
        ds = ds.rename(columns = {ref_lon: alt})
        out.append(ds)
        
    return pd.concat(out, axis = 1)


args = dict(
    linestyle = 'none',
    fillstyle = 'none',
    marker = 'o',
    markersize = 20,
    capsize = 5
    )


   
def plot_heights_vs_phase(
        ax, ds, period
        ):
    
    res = wv.vertical_phase_parameters(
        ds.index, ds['phase_days_mod'].values, 
        period
        )
    
    print(res)
    
    lz_std = round(res['lambda_z_std'])

    vz_ms = abs(res['vz_km_day'])
    lambda_z = abs(res['vz_km_day'] * period)
    
    infos = (
        f'T = {period} days\n' + 
        f'Vz = {vz_ms:.2f} km/day\n' + 
        f'$\lambda_z$ = {lambda_z:.2f} $\pm$ {lz_std} km '
        )
   
 
   
    ax.errorbar(
        ds['phase_days_mod'], 
        ds.index,
        xerr = ds['phase_std_days'], 
        color = 'red', 
        **args 
        )
    
    phase_max = ds['phase_days_mod'].max() + 2
    phase_max = round(phase_max)
    ax.plot( res['phase_fit'], ds.index, '-', lw = 3, color = 'red')
    ax.set(
           ylabel = 'Altitude (km)', 
           xlabel = 'Phase (days)', 
           ylim = [10, 110],
           xlim = [-phase_max, phase_max],
           xticks = np.arange(-phase_max, phase_max, 2)
           )
    
    ax.text(
        0.05, 0.7, 
        infos, 
        fontsize = 25,
        transform = ax.transAxes
        )
    
    return None 



def plot_heights_vs_amplitude(ax, ds, color = 'blue'):
    
    ax.errorbar(
        ds['amplitude'], 
        ds.index,
        xerr = ds['amplitude_std'], 
        color = color, 
        **args 
        )
    
    amplitude_max = ds['amplitude'].max() *1.2
    ax.set( 
        xlabel = 'Amplitude (K)', 
        ylim = [10, 110],
        xlim = [-0.2, amplitude_max], 
        )
     
    coef, cov = np.polyfit(
        ds.index,
        ds['amplitude'],
        1,
        cov=True
    )
    
    slope, intercept = coef
    
    slope_std = np.sqrt(cov[0,0])
    intercept_std = np.sqrt(cov[1,1])
    
    amplitude_fit = slope * ds.index + intercept
    
    ax.plot(amplitude_fit, ds.index, '-', lw = 3, color = color)
    
    # print(slope)
    
    
def plot_phase_amplitude_propagations(ds, period):
    fig, ax = plt.subplots(
         ncols = 2,
         sharey = True,
         figsize= (14, 7), 
         dpi = 300
         )
    
    plt.subplots_adjust(wspace = 0.1)
    plot_heights_vs_phase(ax[0], ds, period)
    plot_heights_vs_amplitude(ax[1], ds)
    
    return fig, ax




def filter_interval_and_fitting_data(
        ds_all, start, end, period
        ):
    ds = ds_all.loc[
        (ds_all.index >= start) & 
        (ds_all.index <= end )
        ].copy()
       
    phases = []
    altitudes = ds.columns
    
    for col in altitudes:
    
        fit = wv.fit_harmonic_phase(ds.index, ds[col], period)
        phases.append(pd.DataFrame(fit, index = [col]) )
        
    df = pd.concat(phases)
    
    df['phase_days_mod'] = wv.unwrap_phase_days(
        df['phase_days_mod'], 
        period = period
        )
        
    return df 


df_main = sb.saber_data( )
 

alts = np.arange(20, 110, 10)
lat_center = -7
ref_lon = -30


ds_all = join_heights(
    df_main, 
    alts,
    bandpass = (2.2, 13),
    lat_center = lat_center,
    ref_lon = ref_lon
    )
 
 

#%%%%
 

period =  5.5
start, end = 50, 100

df =  filter_interval_and_fitting_data(ds_all, start, end, period)
fig, ax = plot_phase_amplitude_propagations(df, period)

fig.suptitle(f'Center latitude {lat_center}°, Longitude {ref_lon}°, DOYs ({start}-{end})')


