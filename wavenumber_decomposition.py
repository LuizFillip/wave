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

import JAWARA as jw 
import matplotlib.pyplot as plt
import pandas as pd

ds = jw.concat_datasets(
    "U",
    latitude=-7,
    zonal_mean=False
)
  
#%%%
 

 

def wavenumber_jawara(
    ds,
    variable="u",
    altitude=50,
    doy_range=(50, 90),
    bandpass=(2.2, 13),
    period_min=2.1,
    period_max=20,
    remove_time_mean=True,
    figsize=(12, 10),
    cmap="seismic",
    levels=31,
):
    """
    Calcula e plota:

    1. Diagrama longitude-tempo filtrado.
    2. Decomposição número de onda-período.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset contendo longitude, level/altitude, time/doy e a variável.
    variable : str
        Nome da variável, por exemplo 'u'.
    altitude : float
        Altitude-alvo em km.
    doy_range : tuple
        Intervalo de dia do ano, por exemplo (50, 90).
    bandpass : tuple
        Banda de períodos usada em pass_band_2d.
    period_min, period_max : float
        Limites do eixo de período na decomposição espectral.
    remove_time_mean : bool
        Remove a média temporal em cada longitude antes do filtro.
    """

    # ---------------------------------------------------------
    # 1. Seleção da variável
    # ---------------------------------------------------------
    if variable not in ds:
        raise KeyError(
            f"A variável {variable!r} não foi encontrada. "
            f"Variáveis disponíveis: {list(ds.data_vars)}"
        )

    da = ds[variable]

    # ---------------------------------------------------------
    # 2. Coordenada de altitude
    # ---------------------------------------------------------
    if "altitude" not in da.coords:
        ds_alt = jw.add_log_pressure_height(ds)
        da = ds_alt[variable]

    if "level" in da.dims and "altitude" in da.coords:
        da = da.swap_dims(
            {"level": "altitude"}
        )

    da = da.sortby("altitude")

    # ---------------------------------------------------------
    # 3. Coordenada DOY
    # ---------------------------------------------------------
    # if "doy" not in da.coords:
        
    doy = (
        da.time.dt.dayofyear
        + da.time.dt.hour / 24
        + da.time.dt.minute / 1440
        + da.time.dt.second / 86400
    )

    da = da.assign_coords(
        doy=("time", doy.values)
    ).sel(
    time=da.time.dt.hour == 0
)


    # Transforma doy em dimensão indexada
    if "time" in da.dims:
        da = da.swap_dims(
            {"time": "doy"}
        )

    da = da.sortby("doy")
    # ---------------------------------------------------------
    # 4. Seleção de período e altitude
    # ---------------------------------------------------------
    doy_start, doy_end = doy_range

    da = da.sel(
        doy=slice(doy_start, doy_end)
    )

    da = da.sel(
        altitude=altitude,
        method="nearest"
    )

    selected_altitude = float(
        da.altitude.values
    )

    # ---------------------------------------------------------
    # 5. Longitude: 0–360 para -180–180
    # ---------------------------------------------------------
    longitude = (
        (da.longitude + 180) % 360
    ) - 180

    da = (
        da
        .assign_coords(longitude=longitude)
        .sortby("longitude")
        .transpose("longitude", "doy")
    )

    # ---------------------------------------------------------
    # 6. Remoção da média temporal
    # ---------------------------------------------------------
    if remove_time_mean:
        da_anom = da - da.mean(
            dim="doy",
            skipna=True
        )
    else:
        da_anom = da.copy()

    # ---------------------------------------------------------
    # 7. DataFrame para as funções existentes
    # ---------------------------------------------------------
    df = da_anom.to_pandas()

    # Segurança adicional
    df = (
        df
        .sort_index()
        .sort_index(axis=1)
    )

    # ---------------------------------------------------------
    # 8. Filtro
    # ---------------------------------------------------------
    filtered = b.pass_band_2d(
        df,
        bandpass=bandpass
    )

    # Garante DataFrame caso a função retorne ndarray
    if isinstance(filtered, np.ndarray):
        filtered = pd.DataFrame(
            filtered,
            index=df.index,
            columns=df.columns
        )

    # ---------------------------------------------------------
    # 9. Escala simétrica
    # ---------------------------------------------------------
    vmax = np.nanpercentile( np.abs(filtered.values), 98 )

    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    contour_levels = np.linspace(
        -vmax,
        vmax,
        levels
    )

    # ---------------------------------------------------------
    # 10. Figura
    # ---------------------------------------------------------
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=figsize,
        dpi=200,
        constrained_layout=True
    )

    im = axes[0].contourf(
        filtered.columns,
        filtered.index,
        filtered.values,
        levels=contour_levels,
        cmap=cmap,
        extend="both"
    )

    axes[0].set_title(
        f"{variable.upper()} filtrado em "
        f"{selected_altitude:.1f} km"
    )

    axes[0].set_xlabel(
        "Day of Year"
    )

    axes[0].set_ylabel(
        "Longitude (°)"
    )

    axes[0].set_ylim(
        -180,
        180
    )

    axes[0].set_yticks(
        np.arange(-180, 181, 60)
    )

    fig.colorbar(
        im,
        ax=axes[0],
        orientation="vertical",
        pad=0.02,
        label=f"{variable.upper()} anomaly"
    )

    plot_zonalnumber_decomposition(
        axes[1],
        filtered,
        period_min=period_min,
        period_max=period_max,
        fontsize=16,
        x=0.1
    )

    axes[1].set_title(
        "Zonal wavenumber–period decomposition"
    )

    return {
        "figure": fig,
        "axes": axes,
        "raw": da,
        "anomaly": da_anom,
        "filtered": filtered,
        "selected_altitude": selected_altitude
    }

wavenumber_jawara(ds)

 