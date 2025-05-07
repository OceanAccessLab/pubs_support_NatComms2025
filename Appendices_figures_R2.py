'''
Script to generate Appendix Figures of the Climate fisheries paper
(trends and levels during climate phases).

This script generates the "ecosystem_data.csv" data file.

This code is static and not updated since re-submission 2
Frederic.Cyr@mi.mun.ca
2025-01-17
'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn as sn
import statsmodels.api as sm

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

years_colors = [
['green'],
['red'],
['green'],
['red'],
['green'],
['red'],
['green'],
['red'],
['green']]

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

## Load climate index
nlci = pd.read_csv('/home/cyrf0006/AZMP/state_reports/reporting_2023/operation_files/NL_climate_index.csv')
nlci.set_index('Year', inplace=True)
# year index vs indices
yindex = nlci.index
iindex = np.arange(len(nlci))
index = pd.Series(iindex, index=yindex)
yndex = pd.Series(yindex, index=iindex)

## Load new Capelin index (2024)
df_cap3 = pd.read_excel('/home/cyrf0006/data/capelin/spring_acoustic_biomass_index_-_mariano_-_updated.xlsx', index_col='year')
df_cap3 = df_cap3['median.biomass.kt']
df_cap3 = df_cap3.interpolate().dropna()

## Load cod biomass data (Schijns et al. 2021)
# Load relative biomass
df_cod = pd.read_csv('/home/cyrf0006/research/keynote_capelin/fig_S11D_data.csv', index_col='year')
Redline = df_cod.RedLine
Redline = Redline.iloc[Redline.index>=1950]*10
# Load catches (Fig 1A)
df_cod_catch = pd.read_csv('/home/cyrf0006/research/keynote_capelin/NCod_Catch.csv', index_col='yr')
df_cod_catch = df_cod_catch.ct
df_cod_catch = df_cod_catch.iloc[df_cod_catch.index>=1950]/1000 #%in kt

# Load Mariano's biomass density (t/km2)
df_bio = pd.read_excel(open('/home/cyrf0006/research/keynote_capelin/biomass_density-clean.xlsx', 'rb'), sheet_name='data_only')
df_bio.set_index('Year', inplace=True)
df_bio_ave = df_bio['Median'].dropna()

# Load Groundfish 
df_ncam = pd.read_csv('/home/cyrf0006/data/NCAM/multispic_process_error.csv', index_col='year')
df_cod = df_ncam[df_ncam.species=='Atlantic Cod']
df_had = df_ncam[df_ncam.species=='Haddock']
df_hak = df_ncam[df_ncam.species=='White Hake']
df_pla = df_ncam[df_ncam.species=='American Plaice']
df_red = df_ncam[df_ncam.species=='Redfish spp.']
df_ska = df_ncam[df_ncam.species=='Skate spp.']
df_wit = df_ncam[df_ncam.species=='Witch Flounder']
df_yel = df_ncam[df_ncam.species=='Yellowtail Flounder']
df_tur = df_ncam[df_ncam.species=='Greenland Halibut']
df_wol = df_ncam[df_ncam.species=='Wolffish spp.']
# drop 3Ps
df_ncam = df_ncam[df_ncam.region!='3Ps']
# average all years
df_ncam = df_ncam.groupby('year').sum()['delta_biomass_kt']

# Load Primary Production
df_PP = pd.read_csv('/home/cyrf0006/research/keynote_capelin/PP/cmems_PP.csv')
df_PP.set_index('time', inplace=True)
df_PP.drop(2024, inplace=True)

# Load Cal fin Abundance
df_cal_ab = pd.read_csv('/home/cyrf0006/research/PeopleStuff/BelangerStuff/CALFIN_abundance.csv', index_col='Year')
df_cal_ab = df_cal_ab['Mean abundance (log10 ind. m-2)']

##  ---- Save data for outreach ---- ##
df_PP = df_PP.squeeze()
df_PP.index.name = 'Year'
df_PP.name = 'primary_prod_mgC_m-3_d-1'
df_cal_ab.name = 'calfin_density_log10_ind_m-2'
df_bio_ave.name = 'multispecies_biomass_density_kt_km-2'
df_cap3.name = 'capelin_biomass_index_kt'
df_cap3.index.name = 'Year'
df_ncam.name = 'delta_groundfish_biomass_kt'
df_ncam.index.name = 'Year'
df_cod_catch.name = 'Cod_catches_kt'
df_cod_catch.index.name = 'Year'
NLCI = nlci.squeeze()
NLCI.name = 'NLCI'

df_data = pd.concat([NLCI, df_PP,df_cal_ab,df_cap3, df_bio_ave, df_ncam, df_cod_catch], axis=1)
df_data.sort_index(inplace=True)
df_data.to_csv('ecosystem_data.csv', float_format='%.2f')


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
df_PP.plot( ax=ax2, color='Green', linewidth=3, alpha=.9, zorder=200)
ax2.legend(['Net Primary Production'])
plt.ylabel(r'PP ($\rm mgC\,m^{-3}\,d^{-1}$)', color='green')
plt.title('Global ocean biogeochemistry hindcast')
# add mean 
PPmean = np.array(df_PP).mean()
PPstd = np.array(df_PP).std()
ax2.plot([df_PP.index[0], df_PP.index[-1]], [PPmean, PPmean], linestyle=':', color='green', linewidth=2, alpha=0.5)

for years in years_list:
    PP_tmp = df_PP[(df_PP.index>=years[0]) & (df_PP.index<=years[1])]
    if len(PP_tmp)>0:
        ax2.plot([PP_tmp.index[0], PP_tmp.index[-1]], [PP_tmp.mean(), PP_tmp.mean()], linestyle='--', color='green', linewidth=2)
        # add mean +- CI shade
        PPmean = np.array(PP_tmp).mean().round(2)
        PPstd = np.array(PP_tmp).std().round(2)
        ax2.fill_between([PP_tmp.index[0]-.5, PP_tmp.index[-1]+.5], [PPmean - 1.28*PPstd/np.sqrt(len(PP_tmp))], [PPmean + 1.28*PPstd/np.sqrt(len(PP_tmp))], facecolor='green', alpha=.3)
        # print values
        yyyy = index.loc[(index>=years[0]) & (index<=years[1])]
        yyyy = [yyyy.index[0], yyyy.index[-1]]
        # 80% CI:
        print('PP ' + str(yyyy) + ':' + str(PPmean.round(2)) + ' +- ' +  str((1.28*PPstd/np.sqrt(len(PP_tmp))).round(2)))
        print('  -> ' + str((PP_tmp.mean() - df_PP.mean()).round(2)) + ' (' + str(((PP_tmp.mean() - df_PP.mean())/df_PP.std()).round(2)) + ')')
        print(' ')
# Red/Green shades    
for idx, years in enumerate(years_list):
    c = ax.fill_between([years[0], years[1]], [NLCI_XLIMS[0], NLCI_XLIMS[0]], [NLCI_XLIMS[1], NLCI_XLIMS[1]], facecolor=years_colors[idx], alpha=.1) 
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
    plt.plot([years[0], years[0]], [NLCI_XLIMS[0], NLCI_XLIMS[1]], '--k')
ax.set_ylim([NLCI_XLIMS[0], NLCI_XLIMS[1]])
ax2 = ax.twinx()
df_cal_ab.index = index.loc[df_cal_ab.index]
df_cal_ab.plot(ax=ax2, color='tab:brown', linewidth=3, alpha=.9, zorder=200)
ax2.legend(['Calfin'])
plt.ylabel(r'$\rm log_{10}(ind\,m^{-2})$', color='tab:brown')
plt.title('2J3KLNO Calanus finmarchicus density')
ax2.set_ylim([8.5, 9.5])
# add mean
ZPmean = np.array(df_cal_ab).mean()
ZPstd = np.array(df_cal_ab).std()
ax2.plot([df_cal_ab.index[0], df_cal_ab.index[-1]], [ZPmean, ZPmean], linestyle=':', color='brown', linewidth=2, alpha=0.5)

for years in years_list:
    cal_tmp = df_cal_ab[(df_cal_ab.index>=years[0]) & (df_cal_ab.index<=years[1])]
    if len(cal_tmp)>0:
        ax2.plot([cal_tmp.index[0], cal_tmp.index[-1]], [cal_tmp.mean(), cal_tmp.mean()], linestyle='--', color='tab:brown', linewidth=2)
        # add mean CI shade
        ZPmean = np.array(cal_tmp).mean()
        ZPstd = np.array(cal_tmp).std()
        ax2.fill_between([cal_tmp.index[0]-.5, cal_tmp.index[-1]+.5], [ZPmean - 1.28*ZPstd/np.sqrt(len(cal_tmp))], [ZPmean + 1.28*ZPstd/np.sqrt(len(cal_tmp))], facecolor='tab:brown', alpha=.3)
        # print values
        yyyy = index.loc[(index>=years[0]) & (index<=years[1])]
        yyyy = [yyyy.index[0], yyyy.index[-1]]
        # 80% CI:
        print('Calfin ' + str(yyyy) + ':' + str(ZPmean.round(2)) + ' +- ' +  str((1.28*ZPstd/np.sqrt(len(cal_tmp))).round(2)))
        print('  -> ' + str(((cal_tmp.mean() - df_cal_ab.mean())).round(2)) + ' (' + str(((cal_tmp.mean() - df_cal_ab.mean())/df_cal_ab.std()).round(2)) + ')')
        print(' ')        
# Red/Green shades    
for idx, years in enumerate(years_list):
    c = ax.fill_between([years[0], years[1]], [NLCI_XLIMS[0], NLCI_XLIMS[0]], [NLCI_XLIMS[1], NLCI_XLIMS[1]], facecolor=years_colors[idx], alpha=.1)
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
    plt.plot([years[0], years[0]], [NLCI_XLIMS[0], NLCI_XLIMS[1]], '--k')
ax.set_ylim([NLCI_XLIMS[0], NLCI_XLIMS[1]])
ax2 = ax.twinx()
df_bio_ave.index = index.loc[df_bio_ave.index]
df_bio_ave.plot(ax=ax2, color='red', linewidth=3, alpha=.9, zorder=200)
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
# Red/Green shades    
for idx, years in enumerate(years_list):
    c = ax.fill_between([years[0], years[1]], [NLCI_XLIMS[0], NLCI_XLIMS[0]], [NLCI_XLIMS[1], NLCI_XLIMS[1]], facecolor=years_colors[idx], alpha=.1)
# Set xlim
ax.set_xlim([index.loc[1975], index.loc[2023]])
# Save fig
fig.set_size_inches(w=8, h=6)
fig_name = 'trawl_ruptures.png'
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
    plt.plot([years[0], years[0]], [NLCI_XLIMS[0], NLCI_XLIMS[1]], '--k')
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
        sn.regplot(x=x, y=y, ci=90, color='tab:blue')
         # print values
        yyyy = index.loc[(index>=years[0]) & (index<=years[1])]
        yyyy = [yyyy.index[0], yyyy.index[-1]]
        print('capelin ' + str(yyyy) + ':' + str(summary['mean_slope'].round(2)) + ' +- ' + str(summary['slope_SE'].round(2)))
        print('  -> [' + str((summary['mean_slope'] - summary['slope_SE']).round(2)) + ' , ' + str((summary['mean_slope'] + summary['slope_SE']).round(2)) + ']')
        print('  -> conf. int.: ' + str(summary['slope_CI']))
        print(' ')
# Red/Green shades    
for idx, years in enumerate(years_list):
    c = ax.fill_between([years[0], years[1]], [NLCI_XLIMS[0], NLCI_XLIMS[0]], [NLCI_XLIMS[1], NLCI_XLIMS[1]], facecolor=years_colors[idx], alpha=.1)
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
    plt.plot([years[0], years[0]], [NLCI_XLIMS[0], NLCI_XLIMS[1]], '--k')
ax.set_ylim([NLCI_XLIMS[0], NLCI_XLIMS[1]])
ax2 = ax.twinx()
df_ncam.index = index.loc[df_ncam.index]
df_ncam.plot(ax=ax2, color='tab:orange', linewidth=3, alpha=.9, zorder=200)
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
# Red/Green shades    
for idx, years in enumerate(years_list):
    c = ax.fill_between([years[0], years[1]], [NLCI_XLIMS[0], NLCI_XLIMS[0]], [NLCI_XLIMS[1], NLCI_XLIMS[1]], facecolor=years_colors[idx], alpha=.1)
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
    plt.plot([years[0], years[0]], [NLCI_XLIMS[0], NLCI_XLIMS[1]], '--k')
ax.set_ylim([NLCI_XLIMS[0], NLCI_XLIMS[1]])
ax2 = ax.twinx()
df_cod_catch.index = index.loc[df_cod_catch.index]
df_cod_catch.plot(ax=ax2, color='darkgoldenrod', linewidth=3, alpha=.9, zorder=200)
ax2.legend(['Cod'])
plt.ylabel(r'Catches ($\rm kt$)', color='darkgoldenrod')
plt.title('Cod catches re-construction (Schjins et al.,2021)')
#ax2.set_ylim([0, 6000])
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
# Red/Green shades    
for idx, years in enumerate(years_list):
    c = ax.fill_between([years[0], years[1]], [NLCI_XLIMS[0], NLCI_XLIMS[0]], [NLCI_XLIMS[1], NLCI_XLIMS[1]], facecolor=years_colors[idx], alpha=.1)
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
    c = plt.fill_between([years[0], years[1]], [YLIM[0], YLIM[0]], [YLIM[1], YLIM[1]], facecolor=years_colors[idx], alpha=.2)
    fig_name = 'NLCI_gray_period_' + str(idx) + '.png'
    fig.set_size_inches(w=7,h=5)
    fig.savefig(fig_name, dpi=200)
    os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)
    c.remove()

