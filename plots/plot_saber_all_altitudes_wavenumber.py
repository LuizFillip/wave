import matplotlib.pyplot as plt
import SABER as sb 
import base as b 
import numpy as np
import waves as wv 
def test_data():
    df = sb.saber_data()
     
    value_col = "temp_100"
    lat_center = 0
    step = 20
    ds = sb.box_groupy_process(
            df, 
            value_col, 
            lat_center = lat_center, 
            bandpass = (2.1, 15),
            lon_step = step,
            lat_step = step,
            lon_stride = None
            )
     
    start, end = 50, 100
    ds1 = ds.loc[:,
        (ds.columns >= start) &  (ds.columns <= end )
        ].copy()

def wavenumber_for_all_altitudes(
        
        df ,
        lat_center = -7,
        start = 60, 
        end = 90,
        bandpass = (2.1, 12),
        normalize = True,
        lon_step = 20,
        lat_step = 20,
        lon_stride = None, 
        year = 2025
        ):
 
    alts = np.arange(20, 110, 10)
    
    fig, ax = plt.subplots(
        figsize = (16, 12),
        nrows = 3,
        ncols = 3, 
        sharex = True, 
        sharey = True
        )
    
    plt.subplots_adjust(wspace = 0.1)
    l = b.letters()
    
    for i, ax in enumerate(ax.flat):
        alt = alts[i]
        value_col = f'temp_{alt}'
        
        ds = sb.box_groupy_process(
                df, 
                value_col, 
                lat_center = lat_center,
                bandpass = bandpass,
                normalize = normalize,
                lon_step = lon_step,
                lat_step = lat_step,
                lon_stride = lon_stride
                )
        
        
        ds1 = ds.loc[:,
            (ds.columns >= start) & 
            (ds.columns <= end )
            ].copy()
        
        wv.plot_zonalnumber_decomposition(
                ax, 
                ds1,
                period_min = 2.1, 
                period_max = 20, 
                fontsize = 20
                )
        
        ax.set(title = f'({l[i]}) {alt} km',)
        if i < 6:
            ax.set(
                # yticklabels = [],
                # xticklabels = [],
                xlabel = '',
                ylabel = ''
                
                )
            
    return None 

def main():
    path = 'SABER/data/saber_mean_2025'
    df = sb.saber_data(path)
    
    step = 20
    wavenumber_for_all_altitudes(
            
            df ,
            lat_center = -7,
            start = 50,
            bandpass = (2.1, 12),
            normalize = True,
            lon_step = step,
            lat_step = step,
            lon_stride = None, 
            year = 2025
            )
    
main()