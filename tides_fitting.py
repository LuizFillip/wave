import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def remove_solar_tides(df, time_col, value_col, periods_hours=[24, 12, 8, 6]):
    """
    Remove componentes de maré solar por ajuste harmônico.

    Parameters
    ----------
    df : DataFrame
        Série temporal.
    time_col : str
        Nome da coluna de tempo.
    value_col : str
        Nome da coluna da variável.
    periods_hours : list
        Períodos das marés a remover, em horas.

    Returns
    -------
    DataFrame com:
        tide_fit : marés ajustadas
        residual : série sem marés
    """

    data = df[[time_col, value_col]].dropna().copy()
    data[time_col] = pd.to_datetime(data[time_col])

    # Tempo em horas desde o início
    t = (data[time_col] - data[time_col].iloc[0]).dt.total_seconds() / 3600

    y = data[value_col].values

    # Matriz do modelo
    X = [np.ones(len(t))]

    for P in periods_hours:
        omega = 2 * np.pi / P
        X.append(np.cos(omega * t))
        X.append(np.sin(omega * t))

    X = np.column_stack(X)

    # Ajuste por mínimos quadrados
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    tide_fit = X @ coef
    residual = y - tide_fit

    data["tide_fit"] = tide_fit
    data["residual"] = residual

    return data, coef

# Criar série horária com 30 dias
time = pd.date_range("2025-01-01", periods=30*24, freq="h")

t = np.arange(len(time))

# Sinal sintético: marés + oscilação de 10 dias + ruído
signal = (
    10 * np.cos(2*np.pi*t/24) +      # maré diurna
    5  * np.cos(2*np.pi*t/12) +      # maré semidiurna
    3  * np.cos(2*np.pi*t/8)  +      # terdiurna
    2  * np.cos(2*np.pi*t/6)  +      # quarterdiurna
    4  * np.cos(2*np.pi*t/(10*24)) + # onda de 10 dias
    np.random.normal(0, 1, len(t))
)

df = pd.DataFrame({
    "time": time,
    "value": signal
})

result, coef = remove_solar_tides(
    df,
    time_col="time",
    value_col="value",
    periods_hours=[24, 12, 8, 6]
)

plt.figure(figsize=(12, 5))
plt.plot(result["time"], result["value"], label="Série original", alpha=0.6)
plt.plot(result["time"], result["tide_fit"], label="Marés ajustadas", linewidth=2)
plt.legend()
plt.xlabel("Tempo")
plt.ylabel("Amplitude")
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(result["time"], result["residual"], label="Série sem marés")
plt.legend()
plt.xlabel("Tempo")
plt.ylabel("Resíduo")
plt.show()