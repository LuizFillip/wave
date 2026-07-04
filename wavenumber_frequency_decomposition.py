import waves as wv 
import numpy as np
import matplotlib.pyplot as plt
import SABER as sb 
from scipy import signal 
import base as b 


df = sb.saber_data()
 
value_col = "temp_90"

ds = sb.box_groupy_process(
        df, 
        value_col, 
        lat_center = -7, 
        bandpass = (2.1, 20) 
        )
 
 
def hanning_remove_tendency(T):
    
    # T = T - np.nanmean(T)
    # T = T - np.nanmean(T, axis=0, keepdims=True)
    # T = T - np.nanmean(T, axis=1, keepdims=True)
 
    T = signal.detrend(T, axis=0, type="linear")
    T = signal.detrend(T, axis=1, type="linear")
 
    wlon = np.hanning(T.shape[0])[:, None]
    wtime = np.hanning(T.shape[1])[None, :]
    T = T * wlon * wtime
    return T
 
def zonal_propagation(ds, period_min=3, period_max=20, smax=None):
    lon = ds.index.values.astype(float)
    doy = ds.columns.values.astype(float)
  
    T =  hanning_remove_tendency(ds.values.astype(float))
     
    dlon = (lon[1] - lon[0]) / 360.0
    dt = doy[1] - doy[0]

    F = np.fft.fft2(T)

    power = np.abs(F)**2

    # remove explicitamente o modo k=0
    # s_raw = np.fft.fftfreq(len(lon), d=dlon)
    # idx_k0 = np.argmin(np.abs(s_raw))
    # power[idx_k0, :] = np.nan

    # normalização antes do log
    #
    power = np.abs( np.log10(power)) 
    # power = power / np.nanmax(power)
    
    s_shift = np.fft.fftfreq(len(lon), d=dlon)
    freq = np.fft.fftfreq(len(doy), d=dt)

    s_shift = np.fft.fftshift(s_shift)
    power_shift = np.fft.fftshift(power, axes=0)

    idx_f = freq >= 0
    freq_pos = freq[idx_f]
    period_pos = 1 / freq_pos
    power_pos = power_shift[:, idx_f]

    idx_p = (period_pos >= period_min) & (period_pos <= period_max)

    period_pos = period_pos[idx_p]
    power_pos = power_pos[:, idx_p]

    S, P = np.meshgrid(s_shift, period_pos, indexing="ij")
    return  S, P, power_pos
 
def plot_zonalnumber_decomposition( ax,  ds ):
    
    
    S,  P, power_pos = zonal_propagation(
        ds, period_min=2, period_max=15, smax=None)
 
    img = ax.contourf(
        S,  P, power_pos,
        levels=50,
        cmap="turbo"
    )

    ax.axvline(0, color="k", linewidth=1.2)
    
    y = 0.9
    fontsize = 20
    ax.text(
        0.05, y, "Westward", 
        transform=ax.transAxes,
        ha="left", va="bottom", 
        fontsize=fontsize, color="k"
        )

    ax.text(
        0.75, y, "Eastward", 
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=fontsize, color="k"
        )

    for p in [5, 10]:
        ax.axhline(p, linestyle = ":", color="k", linewidth=1)
        
    ax.axvline(-1, linestyle ='--')
    ax.set(
        xlabel="Zonal wave number",
        ylabel="Period (days)",
        # ylim=(period_min, period_max)
    )


    b.colorbar(
            img, 
            ax, 
            # ticks, 
            label = 'Normalized log power', 
            height = "100%", 
            width = "3%",
            orientation = "vertical", 
            anchor = (.1, 0., 1, 1)
            )
    # cbar.set_label("Normalized log power")
 
   
    return None

#%%%


import pandas as pd 
from scipy.ndimage import gaussian_filter

fig, ax = plt.subplots(
    figsize = (12, 10),
    nrows = 2, 
    dpi = 300
    )

plot_zonalnumber_decomposition( ax[1], ds )


ds1 = pd.DataFrame(
     gaussian_filter( 
         ds.to_numpy(), 
         sigma = 0.5
         ),
     index = ds.index,
     columns = ds.columns
 )
 
ax[0].contourf(
     ds1.columns, 
     ds1.index, 
     ds1.values,
     cmap = 'seismic',
     levels = 50
     # levels = np.linspace(-1, 1, 30)
     )


# ds = wv.create_synthetic_wave_grid()


