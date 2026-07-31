

from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
 
cmap_vik = "coolwarm"


# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

onda = 6
dn = "2025-02-21"

filepath = Path(f"JAWARA/data/ep_divs_q{onda}.nc")
 
R_E = 6371.0
H = 7.0

lat_min_lim = 0.0
lat_max_lim = 90.0

z_min_lim = 10.0
z_max_lim = 120.0

stride_lat = 1
stride_z = 2

escala_viz = 25.0
fator_escala_z = 200.0


# =============================================================================
# 1. LEITURA DAS COORDENADAS
# =============================================================================

def read_coordinates(ds):
    """Lê latitude, altitude geopotencial e tempo do arquivo NetCDF."""

    lat = np.asarray(ds.variables["lat"][:], dtype=float)
    z_geopotential = np.asarray(ds.variables["z"][:], dtype=float) / 1000.0

    time_var = ds.variables["time"]

    times_raw = nc.num2date(
        time_var[:],
        units=time_var.units,
        calendar=getattr(time_var, "calendar", "standard")
    )

    times = pd.to_datetime([
        time.strftime("%Y-%m-%d %H:%M:%S")
        for time in times_raw
    ])

    return times, z_geopotential, lat


# =============================================================================
# 2. CONVERSÃO DE ALTURA
# =============================================================================

def geopotential_to_geometric_height(z_geopotential, earth_radius=6371.0):
    """Converte altura geopotencial em altura geométrica."""

    return earth_radius * z_geopotential / (earth_radius - z_geopotential)


# =============================================================================
# 3. SELEÇÃO TEMPORAL
# =============================================================================

def get_day_indices(times, date_string):
    """Retorna todos os índices horários correspondentes à data escolhida."""

    target_date = pd.Timestamp(date_string)
    start_date = target_date.normalize()
    end_date = start_date + pd.Timedelta(days=1)

    indices = np.where(
        (times >= start_date) &
        (times < end_date)
    )[0]

    if indices.size == 0:
        raise ValueError(
            f"A data {date_string} não foi encontrada.\n"
            f"Intervalo disponível: {times.min()} até {times.max()}."
        )

    return target_date, indices


# =============================================================================
# 4. SELEÇÃO ESPACIAL
# =============================================================================

def filter_space_indices(
    lat_global,
    z_global,
    lat_min=0.0,
    lat_max=87.0,
    z_min=5.0,
    z_max=90.0
):
    """Seleciona os índices de latitude e altitude do domínio desejado."""

    idx_lat = np.where(
        (lat_global >= lat_min) &
        (lat_global <= lat_max)
    )[0]

    idx_z = np.where(
        (z_global >= z_min) &
        (z_global <= z_max)
    )[0]

    if idx_lat.size == 0:
        raise ValueError("Nenhuma latitude encontrada no intervalo definido.")

    if idx_z.size == 0:
        raise ValueError("Nenhuma altitude encontrada no intervalo definido.")

    lat = lat_global[idx_lat]
    z = z_global[idx_z]

    order_lat = np.argsort(lat)
    order_z = np.argsort(z)

    lat = lat[order_lat]
    z = z[order_z]

    return idx_lat, idx_z, order_lat, order_z, lat, z


# =============================================================================
# 5. EXTRAÇÃO DA MÉDIA DIÁRIA
# =============================================================================

def extract_daily_mean(
    ds,
    variable_name,
    idx_day,
    idx_z,
    idx_lat,
    order_z,
    order_lat
):
    """
    Extrai uma variável com dimensões (time, z, lat) e calcula a média diária.

    Retorno:
        Matriz com dimensões (z, lat).
    """

    variable = ds.variables[variable_name]

    expected_dimensions = ("time", "z", "lat")

    if variable.dimensions != expected_dimensions:
        raise ValueError(
            f"A variável {variable_name} possui dimensões "
            f"{variable.dimensions}. Esperado: {expected_dimensions}."
        )

    data = np.asarray(variable[idx_day, :, :], dtype=float)

    data = data[:, idx_z, :]
    data = data[:, :, idx_lat]

    data = np.nanmean(data, axis=0)

    data = data[order_z, :]
    data = data[:, order_lat]

    return data


# =============================================================================
# 6. VERIFICAÇÃO DAS MATRIZES
# =============================================================================

def validate_fields(lat, z, **fields):
    """Confere se todas as matrizes possuem a forma (z, lat)."""

    expected_shape = (len(z), len(lat))

    for name, field in fields.items():
        if field.shape != expected_shape:
            raise ValueError(
                f"A matriz {name} possui shape {field.shape}. "
                f"Esperado: {expected_shape}."
            )

        if not np.any(np.isfinite(field)):
            raise ValueError(
                f"A matriz {name} não possui valores finitos."
            )


# =============================================================================
# 7. ESCALA DE CORES
# =============================================================================

def calculate_color_limits(acceleration):
    """Calcula uma escala de cores simétrica em torno de zero."""

    maximum = np.nanmax(np.abs(acceleration))
    limit = np.ceil(maximum)

    if not np.isfinite(limit) or limit == 0:
        limit = 1.0

    color_range = (-limit, limit)

    return color_range


# =============================================================================
# 8. PREPARAÇÃO DOS VETORES
# =============================================================================

def prepare_vectors(
    lat,
    z,
    f_phi,
    f_z,
    stride_latitude=1,
    stride_altitude=2,
    scale_height=7.0,
    vertical_factor=100.0
):
    """Subamostra, organiza e aplica o escalonamento físico aos vetores."""

    lat_sub = lat[::stride_latitude]
    z_sub = z[::stride_altitude]

    u_sub = f_phi[::stride_altitude, ::stride_latitude]
    v_sub = f_z[::stride_altitude, ::stride_latitude]

    grid_lat, grid_z = np.meshgrid(
        lat_sub,
        z_sub,
        indexing="xy"
    )

    if grid_lat.shape != u_sub.shape:
        raise ValueError(
            f"Grade e campo incompatíveis: "
            f"grade={grid_lat.shape}, campo={u_sub.shape}."
        )

    grid_lats = grid_lat.ravel()
    grid_zs = grid_z.ravel()

    u_vector = u_sub.ravel()
    v_vector = v_sub.ravel()

    valid = (
        np.isfinite(grid_lats) &
        np.isfinite(grid_zs) &
        np.isfinite(u_vector) &
        np.isfinite(v_vector)
    )

    grid_lats = grid_lats[valid]
    grid_zs = grid_zs[valid]

    u_vector = u_vector[valid]
    v_vector = v_vector[valid]

    latitude_factor = np.cos(np.deg2rad(grid_lats))
    density_factor = np.exp(grid_zs / scale_height)

    u_physical = u_vector * latitude_factor * density_factor

    v_physical = (
        v_vector *
        latitude_factor *
        density_factor *
        vertical_factor
    )

    return grid_lats, grid_zs, u_physical, v_physical


# =============================================================================
# 9. CORREÇÃO DO ASPECTO VISUAL
# =============================================================================

def correct_vector_aspect(
    fig,
    ax,
    u_physical,
    v_physical,
    lat_min,
    lat_max,
    z_min,
    z_max,
    visual_scale=15.0
):
    """
    Corrige o tamanho relativo das componentes horizontal e vertical
    considerando a dimensão do eixo na figura.
    """

    fig.canvas.draw()

    bbox = ax.get_window_extent()

    width_pixels = bbox.width
    height_pixels = bbox.height

    delta_lat = lat_max - lat_min
    delta_z = z_max - z_min

    x_factor = delta_lat / width_pixels
    y_factor = delta_z / height_pixels

    u_plot = u_physical * x_factor
    v_plot = v_physical * y_factor

    magnitude = np.hypot(u_plot, v_plot)
    maximum_magnitude = np.nanmax(magnitude)

    if not np.isfinite(maximum_magnitude) or maximum_magnitude == 0:
        raise ValueError(
            "Não foi possível normalizar os vetores: magnitude máxima inválida."
        )

    normalization = visual_scale / maximum_magnitude

    u_plot = u_plot * normalization
    v_plot = v_plot * normalization

    return u_plot, v_plot


# =============================================================================
# 10. GRÁFICO
# =============================================================================

def plot_ep_flux(
        fig, ax,
    lat,
    z,
    acceleration,
    grid_lats,
    grid_zs,
    u_physical,
    v_physical,
    date,
    wave,
    color_range,
    lat_min,
    lat_max,
    z_min,
    z_max,
    visual_scale=15.0,
    vmax = 5
):

    # ax.set_xlim(lat_min, lat_max)
    # ax.set_ylim(z_min, z_max)

    # ax.set_xticks(np.arange(lat_min, lat_max + 1, 10))
    # ax.set_yticks(np.arange(z_min, z_max + 1, 10))
 
    u_plot, v_plot = correct_vector_aspect(
        fig=fig,
        ax=ax,
        u_physical=u_physical,
        v_physical=v_physical,
        lat_min=lat_min,
        lat_max=lat_max,
        z_min=z_min,
        z_max=z_max,
        visual_scale=visual_scale
    )

    # levels_fill = np.linspace( color_range[0],  color_range[1], 31 )
    # print(color_range[0],  color_range[1])
    levels_fill = np.linspace(-vmax, vmax, 31 )
    contour_fill = ax.contourf(
        lat,
        z,
        acceleration,
        levels=levels_fill,
        cmap= 'seismic', #cmap_vik,
        extend="both"
    )

    levels_line = np.linspace(
        color_range[0],
        color_range[1],
        9
    )

    contour_lines = ax.contour(
        lat,
        z,
        acceleration,
        levels=levels_line,
        colors="black",
        linewidths=0.5,
        alpha=0.8
    )

    ax.clabel(
        contour_lines,
        inline=True,
        fontsize=11,
        fmt="%.1f",
        colors="black"
    )

    ax.quiver(
        grid_lats,
        grid_zs,
        u_plot,
        v_plot,
        angles="xy",
        scale_units="xy",
        scale=1,
        pivot="middle",
        color="black",
        width=0.0025,
        headwidth=5,
        headlength=5,
        headaxislength=4.5
    )

    # colorbar = fig.colorbar(
    #     contour_fill,
    #     ax=ax,
    #     fraction=0.046,
    #     pad=0.04
    # )

    # colorbar.set_label(
    #     "Zonal acceleration (m s$^{-1}$ day$^{-1}$)"
    # )

    return fig, ax


# =============================================================================
# 11. FUNÇÃO PRINCIPAL
# =============================================================================

 

def plot_latitude_height_EP(filepath, dn, fig, ax, vmax = 5):
    ds = nc.Dataset(filepath, "r")
    
    times, z_geopotential, lat_global = read_coordinates(ds)

    z_global = geopotential_to_geometric_height(
        z_geopotential,
        earth_radius=R_E
    )

    target_date, idx_day = get_day_indices( times,   dn )

    (
        idx_lat,
        idx_z,
        order_lat,
        order_z,
        lat,
        z
    ) = filter_space_indices(
        lat_global=lat_global,
        z_global=z_global,
        lat_min=lat_min_lim,
        lat_max=lat_max_lim,
        z_min=z_min_lim,
        z_max=z_max_lim
    )

    f_phi = extract_daily_mean(
        ds=ds,
        variable_name="F_phi",
        idx_day=idx_day,
        idx_z=idx_z,
        idx_lat=idx_lat,
        order_z=order_z,
        order_lat=order_lat
    )

    f_z = extract_daily_mean(
        ds=ds,
        variable_name="F_z",
        idx_day=idx_day,
        idx_z=idx_z,
        idx_lat=idx_lat,
        order_z=order_z,
        order_lat=order_lat
    )

    acceleration = extract_daily_mean(
        ds=ds,
        variable_name="accel",
        idx_day=idx_day,
        idx_z=idx_z,
        idx_lat=idx_lat,
        order_z=order_z,
        order_lat=order_lat
    )

    validate_fields(
        lat=lat,
        z=z,
        F_phi=f_phi,
        F_z=f_z,
        acceleration=acceleration
    )

    color_range = calculate_color_limits(
        acceleration
    )

    (
        grid_lats,
        grid_zs,
        u_physical,
        v_physical
    ) = prepare_vectors(
        lat=lat,
        z=z,
        f_phi=f_phi,
        f_z=f_z,
        stride_latitude=stride_lat,
        stride_altitude=stride_z,
        scale_height=H,
        vertical_factor=fator_escala_z
    )


    plot_ep_flux(
        fig, ax,
        lat=lat,
        z=z,
        acceleration=acceleration,
        grid_lats=grid_lats,
        grid_zs=grid_zs,
        u_physical=u_physical,
        v_physical=v_physical,
        date=target_date,
        wave=onda,
        color_range=color_range,
        lat_min=lat_min_lim,
        lat_max=lat_max_lim,
        z_min=z_min_lim,
        z_max=z_max_lim,
        visual_scale=escala_viz, 
        vmax = vmax
    )
 
     