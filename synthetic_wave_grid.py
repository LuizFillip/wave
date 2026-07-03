import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


def create_synthetic_wave_grid(
    nlon=36,
    ndays=121,
    noise=0.3,
    seed=42
):
    np.random.seed(seed)

    lon = np.linspace(0, 360, nlon, endpoint=False)
    doy = np.arange(1, ndays + 1)

    LON, TIME = np.meshgrid(lon, doy, indexing="ij")

    # Ondas artificiais
    # Forma geral:
    # cos(2π * (k*lon/360 - t/period))  -> eastward
    # cos(2π * (k*lon/360 + t/period))  -> westward

    wave_east_1 = 2.0 * np.cos(
        2 * np.pi * (3 * LON / 360 - TIME / 8)
    )

    wave_east_2 = 1.2 * np.cos(
        2 * np.pi * (6 * LON / 360 - TIME / 5)
    )

    wave_west_1 = 1.8 * np.cos(
        2 * np.pi * (4 * LON / 360 + TIME / 12)
    )

    wave_west_2 = 1.0 * np.cos(
        2 * np.pi * (2 * LON / 360 + TIME / 16)
    )

    # Componente estacionária fraca
    stationary = 0.8 * np.cos(
        2 * np.pi * (1 * LON / 360)
    )

    # Ruído
    random_noise = noise * np.random.randn(nlon, ndays)

    T = (
        wave_east_1
        # + wave_east_2
        # + wave_west_1
        # + wave_west_2
        + stationary
        + random_noise
    )

    ds = pd.DataFrame(
        T,
        index=lon,
        columns=doy
    )

    ds.index.name = "lon_box"
    ds.columns.name = "doy"

    return ds

ds = create_synthetic_wave_grid()

Z = ds.values
x, y = ds.columns, ds.index 

fig, ax = plt.subplots()

levels = np.linspace(-1, 1, 30)
img = ax.contourf(
    x, y, Z,
    cmap='turbo',
    
    levels = 70
    # levels = levels
    )