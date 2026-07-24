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

def show_parameters_infos(ax, res, period):
 
 
    # lz_std = round(res['lambda_z_std'])

    vz_ms = int(abs(res['vz_km_day']))
    lambda_z = int(abs(res['vz_km_day'] * period))
    # vz_std = round(abs(res['vz_km_day_std']))
    
    return (
        f'$V_z$ = {vz_ms} km/day\n' + 
        f'$\lambda_z$ = {lambda_z} km '
        )
   
   

    
    
def plot_phase_amplitude_propagations(
        ds, period, 
        infos_coords = (0.5, 0.7),
        fontsize = 20
        ):
    fig, ax = plt.subplots(
         ncols = 2,
         sharey = True,
         figsize= (10, 7), 
         dpi = 300
         )
    
    plt.subplots_adjust(wspace = 0.1)
    plot_heights_vs_phase(
        ax[0], ds, 
        period, 
        infos_coords = (0.5, 0.7),
        fontsize = 20
        )
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



def plot_heights_vs_phase(
        ax, ds, period,
        infos_coords = (0.5, 0.7),
        fontsize = 20,
        color = 'k', 
        ):
    
    x_line = ds.index
    res = wv.vertical_phase_parameters(
        ds.index, 
        ds['phase_days_mod'].values, 
        period
        )
    
    infos = show_parameters_infos(ax, res, period)
 
    if  infos_coords is not None:
        x, y = infos_coords
        ax.text(
           x, y, 
           infos, 
           fontsize = fontsize,
           transform = ax.transData,
           color = color
           )
   
    ax.errorbar(
        ds['phase_days_mod'], 
        ds.index,
        xerr = ds['phase_std_days'], 
        color = 'k', 
        **args 
        )
    
    phase_max = ds['phase_days_mod'].max() + 2
    phase_max = round(phase_max)
    ax.plot( res['phase_fit'], ds.index, '-', lw = 3, color = color)
    
    x_mid = np.mean(x_line)
    # y_mid = np.mean(y_line)


    ax.set(
           ylabel = 'Altitude (km)', 
           xlabel = 'Phase (days)', 
           ylim = [0, 120],
           xlim = [-phase_max, phase_max],
           xticks = np.arange(-phase_max, phase_max, 2)
           )
    
#     ax.annotate(
#     rf'$V_z = {vz:.1f}\ \mathrm{{km/day}}$' '\n'
#     rf'$\lambda_z = {lambda_z:.1f}\ \mathrm{{km}}$',
#     xy=(x_mid, y_mid),
#     xytext=(25, 0),
#     textcoords='offset points',
#     color='red',
#     fontsize=20,
#     ha='left',
#     va='center'
# )
    
    
    return None 



def plot_heights_vs_amplitude(ax, ds, color = 'blue'):
    
    ax.errorbar(
        ds['amplitude'], 
        ds.index,
        xerr = ds['amplitude_std'], 
        color = 'k', 
        **args 
        )
    
    amplitude_max = ds['amplitude'].max() *1.2
    ax.set( 
        xlabel = 'Amplitude (K)', 
        ylim = [0, 120],
        xlim = [-0.2, amplitude_max], 
        )
     
    coef, cov = np.polyfit(
        ds.index,
        ds['amplitude'],
        1,
        cov=True
    )
    
    slope, intercept = coef
    
    # slope_std = np.sqrt(cov[0,0])
    # intercept_std = np.sqrt(cov[1,1])
    
    amplitude_fit = slope * ds.index + intercept
    
    ax.plot(
        amplitude_fit, 
        ds.index, '-',
        lw = 3,
        color = color
        )
    
    return None 
    


def phase_analysis(
        ds_all, 
        alts,
        lat_center =  -7, 
        ref_lon = -30,
        period = 5.5,
        limits =( 50, 100)
        ):
    
 

    start, end = limits
    
    ds_lim = ds_all.loc[ 
        (ds_all.index >= start) & (ds_all.index <= end)].copy()
    
    df =  get_wave_parameters_for_alts(ds_lim,  period)
    
    df.index = alts
    
    bellow_50  = df.loc[df.index <= 50]
    above_50 = df.loc[df.index > 50]
    
    fig, ax = plt.subplots(
         ncols = 2,
         sharey = True,
         figsize= (10, 7), 
         dpi = 300
         )
    
    plt.subplots_adjust(wspace = 0.1)
    coords = [(6, 25), (6, 75)]
    colors =  ['blue', 'red']
    for i, da in enumerate([bellow_50, above_50]):
    
        plot_heights_vs_phase(
            ax[0], da, 
            period, 
            infos_coords = coords[i],
            fontsize = 20,
            color = colors[i]
            )
        
        ax[0].set(xlim = [-10, 10])
        
        plot_heights_vs_amplitude(ax[1], da, color = colors[i])
        ax[1].set(xlim = [-2, 5])
        
    fig.suptitle(f'Center latitude {lat_center}°, Longitude {ref_lon}°, DOYs ({start}-{end})')
    
   
    
    return fig 
    
def main():
    df_main = sb.saber_data('SABER/data/saber_mean_2025')
       
      
    
    ds_all = sb.join_heights_by_lon_ref(
         df_main, 
         bandpass = (2.2, 15),
         lat_center = -7,
         ref_lon = -30,
         normalize = False,
         ) 
    alts = [int(c[5:]) for c in df_main.columns if 'temp' in c]
    
    fig = phase_analysis(
        ds_all,
        lat_center =  -7,
        period = 6.25,
        limits = (60, 90)
        )
    
    
    ds_all 