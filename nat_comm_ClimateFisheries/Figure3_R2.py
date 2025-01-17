'''
Script to generate Figure 3 of the Climate fisheries paper
(SLP patterns).


data source: https://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanalysis.surface.html
https://www.esrl.noaa.gov/psd/cgi-bin/db_search/DBListFiles.pl?did=195&tid=71800&vid=676

This code is static and not updated since re-submission 2
Frederic.Cyr@mi.mun.ca
2025-01-17
'''


import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
#import datetime
os.environ['PROJ_LIB'] = '/home/cyrf0006/anaconda3/share/proj'
from mpl_toolkits.basemap import Basemap
# For regression stats
import statsmodels.api as sm

# 2024 revision
years_list = [
 [1951, 1971],
 [1972, 1978],
 [1979, 1981],
 [1982, 1998],
 [1999, 2006],
 [2007, 2009],
 [2010, 2013],
 [2014, 2017],
 [2018, 2023]]

months = [1, 12] # months to keep

# For map limits
lllon = -100.
urlon = 10.
lllat = 0.
urlat = 90.

# For study area
lat1 = 65
lat2 = 47
lon1 =  -47
lon2 = -65
lon1 =  360-40.986870
lon2 = 360-60.704367

#v = np.arange(990, 1030) # SLP values
v = np.arange(995, 1025) # SLP values

# Define a bootstrap model for linear regression
def lreg_bootstrap(x, y, nboot=1000, ci=95):
    # initialize empty dict
    summary = {}

    # find index of CI bounds
    idx1 = int(np.round(nboot*((100-ci)/2)/100))
    idx2 = int(np.round(nboot*(100-(100-ci)/2)/100))
    idxm = int(np.round(nboot/2))
    
    # regular x
    x_fit = sm.add_constant(x)
    indices = np.arange(len(x))
    
    # Run nboot iterations
    boot_slope = np.array([np.nan] * nboot)
    boot_intcp = np.array([np.nan] * nboot)
    for i in range(nboot):
        # randomly select indices with replacements
        rdm_idx = choices(indices, k=len(x))
        # Ordinary Linear Regresion
        fit_results = sm.OLS(y[rdm_idx], x_fit[rdm_idx]).fit()
        # Store slope and intercept
        boot_slope[i] = fit_results.params[1]
        boot_intcp[i] = fit_results.params[0]

    # Sort and remove wings of the distribution
    boot_slope = np.sort(boot_slope)
    boot_intcp = np.sort(boot_intcp)
    summary['mean_slope'] = np.round(np.mean(boot_slope), 3)
    summary['mean_intcp'] = np.round(np.mean(boot_intcp), 3)
    summary['median_slope'] = np.round(boot_slope[idxm], 3)
    summary['median_intcp'] = np.round(boot_intcp[idxm], 3)
    summary['slope_SE'] = np.round(np.std(boot_slope), 3)
    summary['intcp_SE'] = np.round(np.std(boot_intcp), 3)
    summary['slope_CI'] = np.round([boot_slope[idx1], boot_slope[idx2]], 3)
    summary['intcp_CI'] = np.round([boot_intcp[idx1], boot_intcp[idx2]], 3)        

    return summary

# Load SLP data from NOAA ESRL
ds = xr.open_dataset('slp.mon.mean.nc')

da = ds['slp']
p = da.to_dataframe() # deprecated
p = p[(p.index.get_level_values('time').year<=2020)]

# Compute climatology 1950-2000
p_clim = p[(p.index.get_level_values('time').year>=1950) & (p.index.get_level_values('time').year<=2000)]
df_clim = p_clim.unstack().groupby(level=['lat']).mean()

## Load ecosystem data
df_data = pd.read_csv('ecosystem_data.csv', index_col='Year')

# Load Groundfish 
df_ncam = df_data['delta_groundfish_biomass_kt']
df_cap3 = df_data['capelin_biomass_index_kt']
df_bio_ave = df_data['multispecies_biomass_density_kt_km-2']
df_PP = df_data['primary_prod_mgC_m-3_d-1']
df_cal_ab = df_data['calfin_density_log10_ind_m-2']

####  ---- Loop on years --- ####
for years in years_list:
    print(years)
    p_year = p[(p.index.get_level_values('time').year>=years[0]) & (p.index.get_level_values('time').year<=years[1])]

    # average all years
    df = p_year.unstack()
    df = df.groupby(level=['lat']).mean() 

    #### ---- Anomaly ---- ####
    df_anom = df - df_clim
    fig_name = 'anom_SLP_' + str(years[0]) + '-' + str(years[1]) + '.png'
    fig_name2 = 'anom_SLP_' + str(years[0]) + '-' + str(years[1]) + '.svg'
    print(fig_name)
    plt.clf()
    fig2, ax = plt.subplots(nrows=1, ncols=1)

    m = Basemap(projection='ortho',lon_0=-40,lat_0=40, resolution='l', llcrnrx=-4000000, llcrnry=-2000000, urcrnrx=5000000, urcrnry=7000000)
    m.drawcoastlines()
    m.fillcontinents(color='tan', zorder=10)
    # draw parallels and meridians.
    m.drawparallels(np.arange(-90.,120.,30.), zorder=10)
    m.drawmeridians(np.arange(0.,420.,60.), zorder=10)

    x,y = m(*np.meshgrid(df.columns.droplevel(None) ,df.index))
    c = m.contourf(x, y, df_anom.values, np.linspace(-1.8, 1.8, 10), cmap=plt.cm.seismic, extend='both');
    ct = m.contour(x, y, df_clim.values, 10, colors='k', zorder=50);
    cb = plt.colorbar(c)
    cb.set_label('SLP anomaly (mb)', fontsize=15)
    xBox, yBox = m([lon2, lon1, lon1, lon2, lon2], [lat2, lat2, lat1, lat1, lat2])
    m.plot(xBox, yBox, '--k', linewidth=2, zorder=50)
    plt.text(8400000, 12800000, str(years[0]) + '-' + str(years[1]), fontsize=18, fontweight='bold')

    #### ---- Add ecosystem trends ---- ####
    ## Trends Cod (Regular's process error)
    df_tmp2 = df_ncam[(df_ncam.index>=years[0]) & (df_ncam.index<=years[1])].dropna()
    if (len(df_tmp2)>0) & (years[0]>1975):
        # new trend calculation (2025):
        summary = lreg_bootstrap(df_tmp2.index, df_tmp2.values, ci=CI)
        trend = summary['mean_slope'].round(2)
        if trend>0:
            plt.annotate('+' + "{:.0f}".format(np.abs(trend)) + r'$\rm \,kt\,yr^{-1}$', xy=(.985, 0.015), xycoords='axes fraction', color='g', fontsize=20, fontweight='bold', horizontalalignment='right', verticalalignment='bottom', backgroundcolor="w", zorder=100)
        elif trend<0:
            plt.annotate('-' + "{:.0f}".format(np.abs(trend)) + r'$\rm \,kt\,yr^{-1}$', xy=(.985, 0.015), xycoords='axes fraction', color='r', fontsize=20, fontweight='bold', horizontalalignment='right', verticalalignment='bottom', backgroundcolor="w", zorder=100)
        
   ## Trends biomass density (Mariano)
    df_tmp4 = df_bio_ave[(df_bio_ave.index>=years[0]) & (df_bio_ave.index<=years[1])].dropna()
    if len(df_tmp4)>2:
        # new trend calculation (2025):
        summary = lreg_bootstrap(df_tmp4.index, df_tmp4.values, ci=CI)
        trend = summary['mean_slope'].round(2)
        if trend>0:
            plt.annotate('+' + "{:.1f}".format(np.abs(trend)) + r'$\rm \,t\,km^{-2}\,yr^{-1}$', xy=(.985, .10), xycoords='axes fraction', color='g', fontsize=20, fontweight='bold', horizontalalignment='right', verticalalignment='bottom', backgroundcolor="w", zorder=100)
        elif trend<0:
            plt.annotate('-' + "{:.1f}".format(np.abs(trend)) + r'$\rm \,t\,km^{-2}\,yr^{-1}$', xy=(.985,.10), xycoords='axes fraction', color='r', fontsize=20, fontweight='bold', horizontalalignment='right', verticalalignment='bottom', backgroundcolor="w", zorder=100)

    ## Trends Capelin (New 2022 index)
    df_tmp = df_cap3[(df_cap3.index>=years[0]) & (df_cap3.index<=years[1])].dropna()
    if len(df_tmp)>0:
        # new trend calculation (2025):
        summary = lreg_bootstrap(df_tmp.index, df_tmp.values, ci=CI)
        trend = summary['mean_slope'].round(2)
        if (years[0]==1999) | (years[0]==2007):
            plt.annotate('+' + "{:.0f}".format(np.abs(trend)) + r'$\rm \,kt\,yr^{-1}$', xy=(.985, .19), xycoords='axes fraction', color='gray', fontsize=20, fontweight='bold', horizontalalignment='right', verticalalignment='bottom', backgroundcolor="w", zorder=100)            
        elif trend>0:
            plt.annotate('+' + "{:.0f}".format(np.abs(trend)) + r'$\rm \,kt\,yr^{-1}$', xy=(.985, .19), xycoords='axes fraction', color='g', fontsize=20, fontweight='bold', horizontalalignment='right', verticalalignment='bottom', backgroundcolor="w", zorder=100)
        elif trend<0:
            plt.annotate('-' + "{:.0f}".format(np.abs(trend)) + r'$\rm \,kt\,yr^{-1}$', xy=(.985, .19), xycoords='axes fraction', color='r', fontsize=20, fontweight='bold', horizontalalignment='right', verticalalignment='bottom', backgroundcolor="w", zorder=100)

    ## Trends PP (cmems)
    df_tmp6 = df_PP[(df_PP.index>=years[0]) & (df_PP.index<=years[1])].dropna()
    if len(df_tmp6)>0:
        anom = (df_tmp6.mean() - df_PP.mean()).values  #mg m-3 day-1
        if anom>0:
            plt.annotate('+' + "{:.2f}".format(np.abs(anom[0])) + r'$\rm \,mgC\,m^{-3}\,d^{-1}$', xy=(.98,.84), xycoords='axes fraction', color='g', fontsize=20, fontweight='bold', horizontalalignment='right', verticalalignment='bottom', backgroundcolor="w", zorder=100)
        elif anom<0:
            plt.annotate('-' + "{:.2f}".format(np.abs(anom[0])) + r'$\rm \,mgC\,m^{-3}\,d^{-1}$', xy=(.98,.84), xycoords='axes fraction', color='r', fontsize=20, fontweight='bold', horizontalalignment='right', verticalalignment='bottom', backgroundcolor="w", zorder=100)

    ## Trends calfin (azmp)
    df_tmp7 = df_cal_ab[(df_cal_ab.index>=years[0]) & (df_cal_ab.index<=years[1])].dropna()
    if len(df_tmp7)>0:
        anom = (df_tmp7.mean() - df_cal_ab.mean())  #mg m-3 day-1
        if (years[0]==1999):
            plt.annotate('+' + "{:.1f}".format(np.abs(anom)) + r'$\rm\,log_{10}(ind\,m^{-2})$', xy=(.98,.76), xycoords='axes fraction', color='gray', fontsize=20, fontweight='bold', horizontalalignment='right', verticalalignment='bottom', backgroundcolor="w", zorder=100)
        elif anom>0:
            plt.annotate('+' + "{:.1f}".format(np.abs(anom)) + r'$\rm\,log_{10}(ind\,m^{-2})$', xy=(.98,.76), xycoords='axes fraction', color='g', fontsize=20, fontweight='bold', horizontalalignment='right', verticalalignment='bottom', backgroundcolor="w", zorder=100)
        elif anom<0:
            plt.annotate('-' + "{:.1f}".format(np.abs(anom)) + r'$\rm\,log_{10}(ind\,m^{-2})$', xy=(.98,.76), xycoords='axes fraction', color='r', fontsize=20, fontweight='bold', horizontalalignment='right', verticalalignment='bottom', backgroundcolor="w", zorder=100)

    fig2.set_size_inches(w=8, h=6)
    fig2.savefig(fig_name, dpi=150)
    os.system('convert -trim ' + fig_name + ' ' + fig_name)
    plt.close()

plt.close('all')

os.system('montage anom_SLP_1951-1971.png anom_SLP_1972-1978.png anom_SLP_1979-1981.png anom_SLP_1982-1998.png anom_SLP_1999-2006.png anom_SLP_2007-2009.png anom_SLP_2010-2013.png anom_SLP_2014-2017.png -tile 2x4 -geometry +10+10  -background white  SLP_anom_capelin.png')

