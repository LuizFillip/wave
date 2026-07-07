import numpy as np 
from scipy import stats
import pandas as pd 
import waves as wv 
import SABER as sb 


    


def evaluate_vertical_wave_periods(results_by_period):
    """
    results_by_period: dict
        Chaves = período testado, ex: 4.0, 4.5, 5.0
        Valores = DataFrame com índice = altitude e colunas:
            'amplitude', 'phase_days', 'phase_std_days'
    """

    rows = []

    for period, df in results_by_period.items():

        data = df.copy()
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna(subset=["phase_days", "amplitude"])

        z = data.index.values.astype(float)
        phase = data["phase_days"].values.astype(float)
        amp = data["amplitude"].values.astype(float)

        if len(data) < 4:
            continue

        # unwrap da fase em dias
        phase_rad = 2 * np.pi * phase / period
        phase_unwrap_rad = np.unwrap(phase_rad)
        phase_unwrap_days = phase_unwrap_rad * period / (2 * np.pi)

        # regressão fase x altitude
        slope, intercept, r_value, p_value, stderr = stats.linregress(
            z, phase_unwrap_days
        )

        phase_fit = intercept + slope * z
        residuals = phase_unwrap_days - phase_fit

        r2 = r_value**2
        rmse = np.sqrt(np.mean(residuals**2))

        # comprimento de onda vertical
        # slope = dias/km
        lambda_z = period / slope if slope != 0 else np.nan

        # velocidade vertical aparente
        # Vz = lambda_z / period
        # km/dia -> m/s
        vz_ms = (lambda_z / period) * 1000 / 86400 if np.isfinite(lambda_z) else np.nan

        # erro médio da fase
        if "phase_std_days" in data.columns:
            phase_std_mean = data["phase_std_days"].mean()
            phase_std_median = data["phase_std_days"].median()
        else:
            phase_std_mean = np.nan
            phase_std_median = np.nan

        # coerência circular da fase
        circ_R = np.abs(np.mean(np.exp(1j * phase_rad)))

        # tendência da amplitude com altitude
        amp_slope, amp_intercept, amp_r, amp_p, amp_stderr = stats.linregress(z, amp)

        rows.append({
            "period": period,
            "n_altitudes": len(data),
            "r2_phase_altitude": r2,
            "rmse_phase_days": rmse,
            "phase_std_mean_days": phase_std_mean,
            "phase_std_median_days": phase_std_median,
            "circular_phase_coherence": circ_R,
            "lambda_z_km": lambda_z,
            "vz_ms": vz_ms,
            "amp_altitude_corr": amp_r,
            "amp_altitude_p": amp_p,
            "phase_slope_days_per_km": slope,
            "phase_slope_stderr": stderr,
            "phase_slope_p": p_value,
        })

    metrics = pd.DataFrame(rows)

    if metrics.empty:
        return metrics

    # score simples: maior R² e coerência, menor RMSE e menor erro de fase
    def normalize_good(x):
        x = x.astype(float)
        return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-12)

    def normalize_bad(x):
        x = x.astype(float)
        return 1 - normalize_good(x)

    metrics["score"] = (
        0.40 * normalize_good(metrics["r2_phase_altitude"]) +
        0.25 * normalize_bad(metrics["rmse_phase_days"]) +
        0.20 * normalize_bad(metrics["phase_std_median_days"]) +
        0.15 * normalize_good(metrics["circular_phase_coherence"])
    )

    metrics = metrics.sort_values("score", ascending=False).reset_index(drop=True)

    return metrics



df_main = sb.saber_data('SABER/data/saber_mean_2025')
 


alts = np.arange(20, 110, 5)
lat_center = -7
ref_lon = -30


ds_all = wv.join_heights(
    df_main, 
    bandpass = (2.2, 13),
    lat_center = lat_center,
    ref_lon = ref_lon
    )


#%%%%


start, end = 36, 51

ds = ds_all.loc[
    (ds_all.index >= start) & 
    (ds_all.index <= end )
    ].copy()
  

ds.columns = alts
ds = ds.loc[:, 
            (ds.columns >= 30) & 
            (ds.columns <= 100)]

results_by_period = {}
for period in np.arange(4, 7, 0.25):
    ds1 = wv.filter_interval_and_fitting_data(
            ds, start, end, period
            )
    
    results_by_period[period] = ds1
    
metrics = evaluate_vertical_wave_periods(results_by_period)
 
print(metrics[[
    "period",
    "score",
    "r2_phase_altitude", 
    "lambda_z_km",
    "vz_ms"
]])

# best_period = metrics.iloc[0]["period"]
# print("Best period:", best_period)