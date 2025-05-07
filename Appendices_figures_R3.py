'''
Script to generate Appendix Figures of the Climate fisheries paper
(trends and levels during climate phases).

This script generates the "ecosystem_data.csv" data file.

This code is static and not updated since re-submission 3
Frederic.Cyr@mi.mun.ca
2025-05-07
'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn as sn
import statsmodels.api as sm
plt.style.use('tableau-colorblind10')

NLCI_XLIMS = [-1.55, 1.55]

years_list = [
[0.5, 21.5],
[21.5, 28.5],
[28.5, 31.5],
[31.5, 48.5],
[48.5, 56.5],
[56.5, 59.5],
[59.5, 63.5],
[63.5, 67.5],
[67.5, 73.5]
]

# Colorblind frienly colors
# To show codes:
# plt.style.use('tableau-colorblind10')
years_colors = [
['#C85200'],#orange
['#5F9ED1'],#blue
['#C85200'],#...
['#5F9ED1'],
['#C85200'],
['#5F9ED1'],
['#C85200'],
['#5F9ED1'],
['#C85200']]

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


## Load ecosystem data
df_data = pd.read_csv('ecosystem_data.csv', index_col='Year')
# Extract individual time series
nlci = df_data['NLCI']
df_ncam = df_data['delta_groundfish_biomass_kt']
df_cap3 = df_data['capelin_biomass_index_kt']
df_bio_ave = df_data['multispecies_biomass_density_kt_km-2']
df_PP = df_data['primary_prod_mgC_m-3_d-1']
df_cal_ab = df_data['calfin_density_log10_ind_m-2']


#### ---- Plot Primary Production ---- ####
plt.clf()
fig, ax = plt.subplots(nrows=1, ncols=1)
n = 5 # xtick every n years
nlci.plot(kind='bar', ax=ax, color='gray', zorder=100, alpha=0.5)
ticks = ax.xaxis.get_ticklocs()
ticklabels = [l.get_text() for l in ax.xaxis.get_ticklabels()]
ax.xaxis.set_ticks(ticks[::n])
ax.xaxis.set_ticklabels(ticklabels[::n])
plt.legend(loc='upper center')
plt.grid('on')
plt.ylabel('NLCI', color='gray')
for years in years_list:
    plt.plot([years[0], years[0]], [NLCI_XLIMS[0], NLCI_XLIMS[1]], '--k')
ax.set_ylim([NLCI_XLIMS[0], NLCI_XLIMS[1]])
ax2 = ax.twinx()
df_PP.index = index.loc[df_PP.index]
df_PP.plot( ax=ax2, color='Green', linewidth=3, alpha=.7, zorder=200)
ax2.legend(['Net Primary Production'])
plt.ylabel(r'PP ($\rm mgC\,m^{-3}\,d^{-1}$)', color='green')
plt.title('Global ocean biogeochemistry hindcast')
# add mean +- sd shade
PPmean = np.array(df_PP).mean()
PPstd = np.array(df_PP).std()
ax2.plot([df_PP.index[0], df_PP.index[-1]], [PPmean, PPmean], linestyle=':', color='green', linewidth=2, alpha=0.5)

for years in years_list:
    PP_tmp = df_PP[(df_PP.index>=years[0]) & (df_PP.index<=years[1])]
    if len(PP_tmp)>0:
        ax2.plot([PP_tmp.index[0], PP_tmp.index[-1]], [PP_tmp.mean(), PP_tmp.mean()], linestyle='--', color='green', linewidth=2)
        # add mean +- std shade
        PPmean = np.array(PP_tmp).mean().round(2)
        PPstd = np.array(PP_tmp).std().round(2)
        ax2.fill_between([PP_tmp.index[0]-.5, PP_tmp.index[-1]+.5], [PPmean - 1.28*PPstd/np.sqrt(len(PP_tmp))], [PPmean + 1.28*PPstd/np.sqrt(len(PP_tmp))], facecolor='green', alpha=.3)
        # print values
        yyyy = index.loc[(index>=years[0]) & (index<=years[1])]
        yyyy = [yyyy.index[0], yyyy.index[-1]]
        # CI:
        print('PP ' + str(yyyy) + ':' + str(PPmean.round(2)) + ' +- ' +  str((1.28*PPstd/np.sqrt(len(PP_tmp))).round(2)))
        print('  -> ' + str((PP_tmp.mean() - df_PP.mean()).round(2)) + ' (' + str(((PP_tmp.mean() - df_PP.mean())/df_PP.std()).round(2)) + ')')
        print(' ')
# warm/cold shades
for idx, years in enumerate(years_list):
    c = ax.fill_between([years[0], years[1]], [NLCI_XLIMS[0], NLCI_XLIMS[0]], [NLCI_XLIMS[1], NLCI_XLIMS[1]], facecolor=years_colors[idx], alpha=.2) 
# Set xlim
ax.set_xlim([index.loc[1975], index.loc[2023]])
# Save fig
fig.set_size_inches(w=8, h=6)
fig_name = 'PP_ruptures.png'
fig.savefig(fig_name, dpi=150)
os.system('convert -trim ' + fig_name + ' ' + fig_name)

#### ---- Plot Zooplankton ---- ####
fig, ax = plt.subplots(nrows=1, ncols=1)
n = 5 # xtick every n years
nlci.plot(kind='bar', ax=ax, color='gray', zorder=100, alpha=0.5)
ticks = ax.xaxis.get_ticklocs()
ticklabels = [l.get_text() for l in ax.xaxis.get_ticklabels()]
ax.xaxis.set_ticks(ticks[::n])
ax.xaxis.set_ticklabels(ticklabels[::n])
plt.legend(loc='upper center')
plt.grid('on')
plt.ylabel('NLCI', color='gray')
for years in years_list:
    plt.plot([years[0], years[0]], [NLCI_XLIMS[0], NLCI_XLIMS[1]], '--k', linewidth=1)
ax.set_ylim([NLCI_XLIMS[0], NLCI_XLIMS[1]])
ax2 = ax.twinx()
df_cal_ab.index = index.loc[df_cal_ab.index]
df_cal_ab.plot(ax=ax2, color='tab:brown', linewidth=3, alpha=.7, zorder=200)
ax2.legend(['Calfin'])
plt.ylabel(r'$\rm log_{10}(ind\,m^{-2})$', color='tab:brown')
plt.title('2J3KLNO Calanus finmarchicus density')
ax2.set_ylim([8.5, 9.5])
# add mean +- sd shade
ZPmean = np.array(df_cal_ab).mean()
ZPstd = np.array(df_cal_ab).std()
ax2.plot([df_cal_ab.index[0], df_cal_ab.index[-1]], [ZPmean, ZPmean], linestyle=':', color='brown', linewidth=2, alpha=0.5)

for years in years_list:
    cal_tmp = df_cal_ab[(df_cal_ab.index>=years[0]) & (df_cal_ab.index<=years[1])]
    if len(cal_tmp)>0:
        ax2.plot([cal_tmp.index[0], cal_tmp.index[-1]], [cal_tmp.mean(), cal_tmp.mean()], linestyle='--', color='tab:brown', linewidth=2)
        # add mean +- std shade
        ZPmean = np.array(cal_tmp).mean()
        ZPstd = np.array(cal_tmp).std()
        ax2.fill_between([cal_tmp.index[0]-.5, cal_tmp.index[-1]+.5], [ZPmean - 1.28*ZPstd/np.sqrt(len(cal_tmp))], [ZPmean + 1.28*ZPstd/np.sqrt(len(cal_tmp))], facecolor='tab:brown', alpha=.3)
        # print values
        yyyy = index.loc[(index>=years[0]) & (index<=years[1])]
        yyyy = [yyyy.index[0], yyyy.index[-1]]
        # CI:
        print('Calfin ' + str(yyyy) + ':' + str(ZPmean.round(2)) + ' +- ' +  str((1.28*ZPstd/np.sqrt(len(cal_tmp))).round(2)))
        print('  -> ' + str(((cal_tmp.mean() - df_cal_ab.mean())).round(2)) + ' (' + str(((cal_tmp.mean() - df_cal_ab.mean())/df_cal_ab.std()).round(2)) + ')')
        print(' ')        
#Warm/Cold shades    
for idx, years in enumerate(years_list):
    c = ax.fill_between([years[0], years[1]], [NLCI_XLIMS[0], NLCI_XLIMS[0]], [NLCI_XLIMS[1], NLCI_XLIMS[1]], facecolor=years_colors[idx], alpha=.2)
# Set xlim
ax.set_xlim([index.loc[1975], index.loc[2023]])
# Save fig
fig.set_size_inches(w=8, h=6)
fig_name = 'calanus_ruptures.png'
fig.savefig(fig_name, dpi=150)
os.system('convert -trim ' + fig_name + ' ' + fig_name)

#### ---- Plot Biomass density ---- ####
plt.clf()
fig, ax = plt.subplots(nrows=1, ncols=1)
n = 5 # xtick every n years
nlci.plot(kind='bar', ax=ax, color='gray', zorder=100, alpha=0.5)
ticks = ax.xaxis.get_ticklocs()
ticklabels = [l.get_text() for l in ax.xaxis.get_ticklabels()]
ax.xaxis.set_ticks(ticks[::n])
ax.xaxis.set_ticklabels(ticklabels[::n])
plt.legend(loc='upper center')
plt.grid('on')
plt.ylabel('NLCI', color='gray')
for years in years_list:
    plt.plot([years[0], years[0]], [NLCI_XLIMS[0], NLCI_XLIMS[1]], '--k', linewidth=1)
ax.set_ylim([NLCI_XLIMS[0], NLCI_XLIMS[1]])
ax2 = ax.twinx()
df_bio_ave.index = index.loc[df_bio_ave.index]
df_bio_ave.plot(ax=ax2, color='red', linewidth=3, alpha=.7, zorder=200)
ax2.legend(['multispecies'])
plt.ylabel(r'biomass density ($\rm t\,km^{-2}$)', color='red')
plt.title('Multispecies scientific trawl survey')
ax2.set_ylim([0, 32])
for years in years_list[3:-1]:
    bio_tmp = df_bio_ave[(df_bio_ave.index>=years[0]) & (df_bio_ave.index<=years[1])]
    if len(bio_tmp)>0:
        x=bio_tmp.index
        y=bio_tmp.values
        summary = lreg_bootstrap(x, y, ci=CI)
        sn.regplot(x=x, y=y, n_boot=1000, ci=80, color='red')
        # print values
        yyyy = index.loc[(index>=years[0]) & (index<=years[1])]
        yyyy = [yyyy.index[0], yyyy.index[-1]]
        print('trawl ' + str(yyyy) + ':' + str(summary['mean_slope'].round(2)) + ' +- ' + str(summary['slope_SE'].round(2)))
        print('  -> [' + str((summary['mean_slope'] - summary['slope_SE']).round(2)) + ' , ' + str((summary['mean_slope'] + summary['slope_SE']).round(2)) + ']')
        print('  -> conf. int.: ' + str(summary['slope_CI']))
        print(' ')
#Warm/Cold shades    
for idx, years in enumerate(years_list):
    c = ax.fill_between([years[0], years[1]], [NLCI_XLIMS[0], NLCI_XLIMS[0]], [NLCI_XLIMS[1], NLCI_XLIMS[1]], facecolor=years_colors[idx], alpha=.2)
# Set xlim
ax.set_xlim([index.loc[1975], index.loc[2023]])
# Save fig
fig.set_size_inches(w=8, h=6)
fig_name = 'trawl_ruptures.png'
fig.savefig(fig_name, dpi=150)
os.system('convert -trim ' + fig_name + ' ' + fig_name)

#### ---- Plot Total Catches ---- ####
plt.clf()
fig, ax = plt.subplots(nrows=1, ncols=1)
n = 5 # xtick every n years
nlci.plot(kind='bar', ax=ax, color='gray', zorder=100, alpha=0.5)
ticks = ax.xaxis.get_ticklocs()
ticklabels = [l.get_text() for l in ax.xaxis.get_ticklabels()]
ax.xaxis.set_ticks(ticks[::n])
ax.xaxis.set_ticklabels(ticklabels[::n])
plt.legend(loc='upper center')
plt.grid('on')
plt.ylabel('NLCI', color='gray')
for years in years_list:
    plt.plot([years[0], years[0]], [NLCI_XLIMS[0], NLCI_XLIMS[1]], '--k', linewidth=1)
ax.set_ylim([NLCI_XLIMS[0], NLCI_XLIMS[1]])
ax2 = ax.twinx()
df_catch_ave.index = index.loc[df_catch_ave.index]
df_catch_ave.plot(ax=ax2, color='indianred', linewidth=3, alpha=.7, zorder=200)
ax2.legend(['Statlan21'])
plt.ylabel(r'Catches ($\rm kt$)', color='indianred')
plt.title('NAFO Statlan21 Catches (NL)')
#ax2.set_ylim([0, 23])
for years in years_list[0:-1]:
    catch_tmp = df_catch_ave[(df_catch_ave.index>=years[0]) & (df_catch_ave.index<=years[1])]
    if len(catch_tmp)>0:
        x=catch_tmp.index
        y=catch_tmp.values
        summary = lreg_bootstrap(x, y, ci=CI)
        sn.regplot(x=x, y=y, ci=80, color='indianred')
        #fit_results = simple_regplot(x, y)
        # print values
        yyyy = index.loc[(index>=years[0]) & (index<=years[1])]
        yyyy = [yyyy.index[0], yyyy.index[-1]]
        print('satlan21 ' + str(yyyy) + ':' + str(summary['mean_slope'].round(2)) + ' +- ' + str(summary['slope_SE'].round(2)))
        print('  -> [' + str((summary['mean_slope'] - summary['slope_SE']).round(2)) + ' , ' + str((summary['mean_slope'] + summary['slope_SE']).round(2)) + ']')
        print('  -> conf. int.: ' + str(summary['slope_CI']))
        print(' ')
#Warm/Cold shades    
for idx, years in enumerate(years_list):
    c = ax.fill_between([years[0], years[1]], [NLCI_XLIMS[0], NLCI_XLIMS[0]], [NLCI_XLIMS[1], NLCI_XLIMS[1]], facecolor=years_colors[idx], alpha=.2)
# Set xlim
ax.set_xlim([index.loc[1960], index.loc[2023]])
# Save fig
fig.set_size_inches(w=8, h=6)
fig_name = 'catch_ruptures.png'
fig.savefig(fig_name, dpi=150)
os.system('convert -trim ' + fig_name + ' ' + fig_name)


#### ---- Plot Capelin Biomass ---- ####
plt.clf()
fig, ax = plt.subplots(nrows=1, ncols=1)
n = 5 # xtick every n years
nlci.plot(kind='bar', ax=ax, color='gray', zorder=100, alpha=0.5)
ticks = ax.xaxis.get_ticklocs()
ticklabels = [l.get_text() for l in ax.xaxis.get_ticklabels()]
ax.xaxis.set_ticks(ticks[::n])
ax.xaxis.set_ticklabels(ticklabels[::n])
plt.legend(loc='upper center')
plt.grid('on')
plt.ylabel('NLCI', color='gray')
for years in years_list:
    plt.plot([years[0], years[0]], [NLCI_XLIMS[0], NLCI_XLIMS[1]], '--k', linewidth=1)
ax.set_ylim([NLCI_XLIMS[0], NLCI_XLIMS[1]])
ax2 = ax.twinx()
df_cap3.index = index.loc[df_cap3.index]
df_cap3.interpolate().plot(ax=ax2, color='tab:blue', linewidth=3, alpha=.6)
ax2.legend(['Capelin'])
plt.ylabel(r'Biomass index ($\rm kt$)', color='tab:blue')
plt.title('Capelin Spring Acoustic Survey')
ax2.set_ylim([0, 6000])
for years in years_list:
    cap_tmp = df_cap3[(df_cap3.index>=years[0]) & (df_cap3.index<=years[1])]
    if len(cap_tmp)>1:
        x=cap_tmp.index
        y=cap_tmp.values
        summary = lreg_bootstrap(x, y, ci=CI)
        sn.regplot(x=x, y=y, ci=80, color='tab:blue')
        # print values
        yyyy = index.loc[(index>=years[0]) & (index<=years[1])]
        yyyy = [yyyy.index[0], yyyy.index[-1]]
        print('capelin ' + str(yyyy) + ':' + str(summary['mean_slope'].round(2)) + ' +- ' + str(summary['slope_SE'].round(2)))
        print('  -> [' + str((summary['mean_slope'] - summary['slope_SE']).round(2)) + ' , ' + str((summary['mean_slope'] + summary['slope_SE']).round(2)) + ']')
        print('  -> conf. int.: ' + str(summary['slope_CI']))
        print(' ')
#Warm/Cold shades    
for idx, years in enumerate(years_list):
    c = ax.fill_between([years[0], years[1]], [NLCI_XLIMS[0], NLCI_XLIMS[0]], [NLCI_XLIMS[1], NLCI_XLIMS[1]], facecolor=years_colors[idx], alpha=.2)
# Set xlim
ax.set_xlim([index.loc[1975], index.loc[2023]])
# Save fig
fig.set_size_inches(w=8, h=6)
fig_name = 'capelin_ruptures.png'
fig.savefig(fig_name, dpi=150)
os.system('convert -trim ' + fig_name + ' ' + fig_name)

#### ---- Plot Groundfish ---- ####
plt.clf()
fig, ax = plt.subplots(nrows=1, ncols=1)
n = 5 # xtick every n years
nlci.plot(kind='bar', ax=ax, color='gray', zorder=100, alpha=0.5)
ticks = ax.xaxis.get_ticklocs()
ticklabels = [l.get_text() for l in ax.xaxis.get_ticklabels()]
ax.xaxis.set_ticks(ticks[::n])
ax.xaxis.set_ticklabels(ticklabels[::n])
plt.legend(loc='upper center')
plt.grid('on')
plt.ylabel('NLCI', color='gray')
for years in years_list:
    plt.plot([years[0], years[0]], [NLCI_XLIMS[0], NLCI_XLIMS[1]], '--k', linewidth=1)
ax.set_ylim([NLCI_XLIMS[0], NLCI_XLIMS[1]])
ax2 = ax.twinx()
df_ncam.index = index.loc[df_ncam.index]
df_ncam.plot(ax=ax2, color='tab:orange', linewidth=3, alpha=.7, zorder=200)
ax2.legend(['Groundfish'])
plt.ylabel(r'Excess biomass ($\rm kt$)', color='tab:orange')
plt.title('Groundfish surplus production model')
for years in years_list[:-1]:
    gf_tmp = df_ncam[(df_ncam.index>=years[0]) & (df_ncam.index<=years[1])]
    if len(gf_tmp)>1:
        x=gf_tmp.index
        y=gf_tmp.values
        summary = lreg_bootstrap(x, y, ci=CI)
        sn.regplot(x=x, y=y, ci=80, color='tab:orange')
        # print values
        yyyy = index.loc[(index>=years[0]) & (index<=years[1])]
        yyyy = [yyyy.index[0], yyyy.index[-1]]
        print('groundfish ' + str(yyyy) + ':' + str(summary['mean_slope'].round(2)) + ' +- ' + str(summary['slope_SE'].round(2)))
        print('  -> [' + str((summary['mean_slope'] - summary['slope_SE']).round(2)) + ' , ' + str((summary['mean_slope'] + summary['slope_SE']).round(2)) + ']')
        print('  -> conf. int.: ' + str(summary['slope_CI']))
        print(' ')
#Warm/Cold shades    
for idx, years in enumerate(years_list):
    c = ax.fill_between([years[0], years[1]], [NLCI_XLIMS[0], NLCI_XLIMS[0]], [NLCI_XLIMS[1], NLCI_XLIMS[1]], facecolor=years_colors[idx], alpha=.2)
# Set xlim
ax.set_xlim([index.loc[1975], index.loc[2023]])
# Save fig
fig.set_size_inches(w=8, h=6)
fig_name = 'groundfish_ruptures.png'
fig.savefig(fig_name, dpi=150)
os.system('convert -trim ' + fig_name + ' ' + fig_name)

#### ---- Plot Cod catches (Schjins et al. 2021) ---- ####
plt.clf()
fig, ax = plt.subplots(nrows=1, ncols=1)
n = 5 # xtick every n years
nlci.plot(kind='bar', ax=ax, color='gray', zorder=100, alpha=0.5)
ticks = ax.xaxis.get_ticklocs()
ticklabels = [l.get_text() for l in ax.xaxis.get_ticklabels()]
ax.xaxis.set_ticks(ticks[::n])
ax.xaxis.set_ticklabels(ticklabels[::n])
plt.legend(loc='upper center')
plt.grid('on')
plt.ylabel('NLCI', color='gray')
for years in years_list:
    plt.plot([years[0], years[0]], [NLCI_XLIMS[0], NLCI_XLIMS[1]], '--k', linewidth=1)
ax.set_ylim([NLCI_XLIMS[0], NLCI_XLIMS[1]])
ax2 = ax.twinx()
df_cod_catch.index = index.loc[df_cod_catch.index]
df_cod_catch.plot(ax=ax2, color='darkgoldenrod', linewidth=3, alpha=.7, zorder=200)
ax2.legend(['Cod'])
plt.ylabel(r'Catches ($\rm kt$)', color='darkgoldenrod')
plt.title('Cod catches re-construction (Schjins et al.,2021)')
for years in years_list[:-1]:
    gf_tmp = df_cod_catch[(df_cod_catch.index>=years[0]) & (df_cod_catch.index<=years[1])]
    if len(gf_tmp)>1:
        x=gf_tmp.index
        y=gf_tmp.values
        summary = lreg_bootstrap(x, y, ci=CI)
        sn.regplot(x=x, y=y, ci=80, color='darkgoldenrod')
        # print values
        yyyy = index.loc[(index>=years[0]) & (index<=years[1])]
        yyyy = [yyyy.index[0], yyyy.index[-1]]
        print('cod catches ' + str(yyyy) + ':' + str(summary['mean_slope'].round(2)) + ' +- ' + str(summary['slope_SE'].round(2)))
        print('  -> [' + str((summary['mean_slope'] - summary['slope_SE']).round(2)) + ' , ' + str((summary['mean_slope'] + summary['slope_SE']).round(2)) + ']')
        print('  -> conf. int.: ' + str(summary['slope_CI']))
        print(' ')
#Warm/Cold shades    
for idx, years in enumerate(years_list):
    c = ax.fill_between([years[0], years[1]], [NLCI_XLIMS[0], NLCI_XLIMS[0]], [NLCI_XLIMS[1], NLCI_XLIMS[1]], facecolor=years_colors[idx], alpha=.2)
# Set xlim
ax.set_xlim([index.loc[1950], index.loc[2023]])
# Save fig
fig.set_size_inches(w=8, h=6)
fig_name = 'cod_ruptures.png'
fig.savefig(fig_name, dpi=150)
os.system('convert -trim ' + fig_name + ' ' + fig_name)


#### ----- Climate index with repsective period of productivity ---- ####
YLIM = [-1.55, 1.55]
# Plot
fig, ax = plt.subplots(nrows=1, ncols=1)
n = 10 # xtick every n years
nlci.plot(kind='bar', ax=ax, color='gray', zorder=100, alpha=0.5, width=.9)
ticks = ax.xaxis.get_ticklocs()
ticklabels = [l.get_text() for l in ax.xaxis.get_ticklabels()]
ax.xaxis.set_ticks(ticks[::n])
ax.xaxis.set_ticklabels(ticklabels[::n], rotation=0)
plt.grid()
ax.set_ylim(YLIM)
plt.xlabel('')
#Loop on year, add period and save.
for idx, years in enumerate(years_list):
    if idx==1: # only keep legend for first set
        ax.get_legend().remove()
    c = plt.fill_between([years[0], years[1]], [YLIM[0], YLIM[0]], [YLIM[1], YLIM[1]], facecolor=years_colors[idx], alpha=.2)
    fig_name = 'NLCI_gray_period_' + str(idx) + '.png'
    fig.set_size_inches(w=7,h=5)
    fig.savefig(fig_name, dpi=300)
    os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)
    c.remove()
