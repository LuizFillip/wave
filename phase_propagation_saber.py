import SABER as sb
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np  
import waves as wv
import base as b 
b.sci_format()




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
        ds.index, 
        ds['phase_days_mod'].values, 
        period
        )
    
    
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
    
    ax.plot(
        amplitude_fit, 
        ds.index, '-',
        lw = 3,
        color = color
        )
    
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




def get_wave_parameters_for_alts( ds, period):

       
    phases = []
    altitudes = ds.columns
    
    for col in altitudes:
    
        fit = wv.fit_harmonic_phase(ds.index, ds[col], period)
        phases.append(
            pd.DataFrame(fit, index = [col]) 
            )
        
    df = pd.concat(phases)
    
    df['phase_days_mod'] = wv.unwrap_phase_days(
        df['phase_days_mod'], 
        period = period
        )
        
    return df 

  
def phase_analysis(
        ds_all, 
        alts, 
        lat_center, 
        ref_lon,
        period = 5.5,
        limits =( 50, 100)
        ):
    
        
    df_main = sb.saber_data('SABER/data/saber_mean_alts')
       
      
    alts = np.arange(20, 110, 10)
    lat_center =  -7
    ref_lon = -30
    ds_all = sb.join_heights_by_lon_ref(
         df_main, 
         bandpass = (2.2, 15),
         lat_center = lat_center,
         ref_lon = ref_lon
         )
     
      
    start, end = limits
    
    df =  get_wave_parameters_for_alts(ds_all,  period)
    
    df.index = alts
    fig, ax = plot_phase_amplitude_propagations(df, period)
    
    fig.suptitle(f'Center latitude {lat_center}°, Longitude {ref_lon}°, DOYs ({start}-{end})')
    
    df 
    


# phase_analysis(
    # ds_all, 
    # alts, 
    # lat_center, 
    # ref_lon,
    # period = 6,
    # limits =( 50, 100)
    # )