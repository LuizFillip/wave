import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import SABER as sb 
from scipy import signal 


df = sb.saber_data()
 
value_col = "temp_90"

ds = sb.box_groupy_process(
        df, 
        value_col, 
        lat_center = -7, 
        bandpass = (3, 20) 
        )
 

 
def zonal_propagation(ds, period_min=3, period_max=20, smax=None):
    lon = ds.index.values.astype(float)
    doy = ds.columns.values.astype(float)

    T = ds.values.astype(float)

    # remove média global, zonal e temporal
    T = T - np.nanmean(T)
    T = T - np.nanmean(T, axis=0, keepdims=True)
    T = T - np.nanmean(T, axis=1, keepdims=True)

    # detrend
    

    T = signal.detrend(T, axis=0, type="linear")
    T = signal.detrend(T, axis=1, type="linear")

    # janela 2D
    wlon = np.hanning(T.shape[0])[:, None]
    wtime = np.hanning(T.shape[1])[None, :]
    T = T * wlon * wtime

    dlon = (lon[1] - lon[0]) / 360.0
    dt = doy[1] - doy[0]

    F = np.fft.fft2(T)

    power = np.abs(F)**2

    # remove explicitamente o modo k=0
    s_raw = np.fft.fftfreq(len(lon), d=dlon)
    idx_k0 = np.argmin(np.abs(s_raw))
    power[idx_k0, :] = np.nan

    # normalização antes do log
    power = power / np.nanmax(power)
    power = np.log10(power + 1e-12)

    s = np.fft.fftfreq(len(lon), d=dlon)
    freq = np.fft.fftfreq(len(doy), d=dt)

    s_shift = np.fft.fftshift(s)
    power_shift = np.fft.fftshift(power, axes=0)

    idx_f = freq > 0
    freq_pos = freq[idx_f]
    period_pos = 1 / freq_pos
    power_pos = power_shift[:, idx_f]

    idx_p = (period_pos >= period_min) & (period_pos <= period_max)

    period_plot = period_pos[idx_p]
    power_plot = power_pos[:, idx_p]

    S, P = np.meshgrid(s_shift, period_plot, indexing="ij")

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    img = ax.contourf(
        S,
        P,
        power_plot,
        levels=50,
        cmap="jet"
    )

    ax.axvline(0, color="k", linewidth=1.2)

    ax.text(0.05, 0.90, "Westward", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=25, color="w")

    ax.text(0.65, 0.90, "Eastward", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=25, color="w")

    for p in [6, 10, 13, 16]:
        ax.axhline(p, linestyle="--", color="w", linewidth=1)

    ax.set(
        xlabel="Zonal wave number",
        ylabel="Period (days)",
        ylim=(period_min, period_max)
    )

    if smax is not None:
        ax.set_xlim(-smax, smax)

    ax.invert_yaxis()

    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label("Normalized log power")

    plt.tight_layout()

    return fig, ax


ds = create_synthetic_wave_grid()

fig, ax = zonal_propagation(
    ds,
    period_min=3,
    period_max=20,
    smax=15
)