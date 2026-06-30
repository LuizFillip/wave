# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 20:39:50 2026

@author: Luiz
"""

def join_heights(df_main, alts, ref_lon = -36):
    out = []
    for alt in alts:
        df =  sb.saber_pivot_longitude(
            df_main, 
            alt = alt, 
            lat_min = -10,
            lat_max = 0,
            res = 1,
            sigma = (6, 1),
            norm = True
            )
         
        ds = df.loc[df.index == ref_lon].T
        ds = ds.rename(columns = {ref_lon: alt})
        out.append(ds)
        
    return pd.concat(out, axis = 1)

def plot_heights_vs_phase(phases, alts, res, period):
    
    slope = res['slope_days_per_km']
    vz_km_day = res['vz_km_day']
    vz_ms = res['vz_m_s']
    lambda_z = res['vz_km_day'] * period
 
     
    infos = (f'dt/dz = {slope:.2f} days/km\n' +
             f'Vz = {vz_km_day:.2f} km/day\n' + 
             f'Vz = {vz_ms:.2f} m/s\n' + 
             f'$\lambda_z$ = {lambda_z:.2f} km')
    
    b.sci_format()
    
    fig, ax = plt.subplots(
        
        figsize= (9, 6), 
        dpi = 300
        )
    ax.plot(phases, alts, 'o')
    ax.plot( res['phase_fit'], alts, '-',
            )
    ax.set(
           ylabel = 'Altitude (km)', 
           xlabel = 'Phase (days)', 
           ylim = [0, 110]
           )
    
    ax.text(
        0.55, 0.1, infos, 
            fontsize = 20,
            transform = ax.transAxes)
   
fn = 'saber_all_alts'
# fn = 'saber_test2'
df = sb.saber_data(fn)
 
alts = np.arange(20, 110, 10)
# alts = [30, 60, 90]
 
ds = join_heights(df, alts, ref_lon = -36)
 
ds = ds.loc[(ds.index >= 40) & (ds.index <= 80)] 
 
period = 13
 

phases = get_phases_mod(ds, period = period)


res = vertical_phase_speed(alts, phases)


# plot_heights_vs_phase(phases, alts, res, period)

 

fig, ax = plt.subplots(figsize=(12, 4))

cols = ds.columns
colors = plt.cm.get_cmap('turbo', len(cols))
offset = 0.5
for i, col in enumerate(cols):
    y = ds[col].values + i * offset
    ax.plot(
        ds.index, y, lw=2, 
        color=colors(i), 
        label=col
        )
ds 