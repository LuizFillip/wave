import numpy as np
from scipy import signal 
import base as b 



def hanning_remove_tendency(T):
    
    T = T - np.nanmean(T)
    T = T - np.nanmean(T, axis=0, keepdims=True)
    T = T - np.nanmean(T, axis=1, keepdims=True)
 
    T = signal.detrend(T, axis=0, type="linear")
    T = signal.detrend(T, axis=1, type="linear")
 
    wlon = np.hanning(T.shape[0])[:, None]
    wtime = np.hanning(T.shape[1])[None, :]
    T = T * wlon * wtime
    return T
  
def zonal_propagation(ds, period_min=3, period_max=20):
    lon = ds.index.values.astype(float)
    doy = ds.columns.values.astype(float)
  
    T =  hanning_remove_tendency(ds.values.astype(float))
    T = ds.values.astype(float)
      
    dlon = np.nanmedian(np.diff(lon)) / 360.0
    dt = np.nanmedian(np.diff(doy))
    
    F = np.fft.fft2(T)

    power = np.abs(F)**2
 
    s_shift = np.fft.fftfreq(len(lon), d=dlon)
    freq = np.fft.fftfreq(len(doy), d=dt)

    s_shift = np.fft.fftshift(s_shift)
    power_shift = np.fft.fftshift(power, axes=0)
    
    idx_f = np.where(freq > 0)[0]   # exclui freq = 0
    freq_pos = freq[idx_f]
    period_pos = 1 / freq_pos
    power_pos = power_shift[:, idx_f]

    idx_p = (period_pos >= period_min) & (period_pos <= period_max)

    period_pos = period_pos[idx_p]
    power_pos = power_pos[:, idx_p]

    S, P = np.meshgrid(s_shift, period_pos, indexing="ij")
    

    return  S, P, power_pos

 
def plot_zonalnumber_decomposition(
        ax,  ds,
        period_min = 2.5, 
        period_max = 20,
        y = 0.85,
        x = 0.05,
        fontsize = 30,
        colorbar = False,
        color_s = 'w'
        ):
    
    
    S,  P, power_pos = zonal_propagation(
        ds, 
        period_min = period_min, 
        period_max = period_max
        )
  
    img = ax.contourf(
        S, P, 
        power_pos,
        levels = 50,
        cmap = "turbo"
    )

    ax.axvline(0, color="w", linewidth=1.2)
    
   
    ax.text(
        x, y, "Westward", 
        transform = ax.transAxes,
        ha ="left",
        va = "bottom", 
        fontsize=fontsize, 
        color=color_s
        )

    ax.text(
        0.6 + x, y, "Eastward", 
        transform = ax.transAxes,
        ha = "left",
        va = "bottom",
        fontsize = fontsize, 
        color=color_s
        )

 
    for v in [-3, -1, 0, 1, 3]:
        ax.axvline(v, linestyle ='--')
        
    ax.set(
        xlabel= "Zonal wave number",
        ylabel= "Period (days)",
        xlim = [-5, 5],
        xticks = np.arange(-5, 6, 1),
        ylim = [period_min, period_max - 5],
        yticks = np.arange(3, 15, 2),
        
    )
    
    if colorbar:

        b.colorbar(
            img, 
            ax,  
            label = 'Normalized log power', 
            height = "100%", 
            width = "3%",
            orientation = "vertical", 
            anchor = (.1, 0., 1, 1)
            ) 
     
   
    return None

