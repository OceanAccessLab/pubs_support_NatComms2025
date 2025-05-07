# -*- coding: utf-8 -*-
'''
Script to generate Figure 2 of the Climate fisheries paper
(climate index with climate period).

This code is static and not updated since re-submission 3
Frederic.Cyr@mi.mun.ca
2025-05-07

'''

import numpy as  np
import matplotlib.pyplot as plt
import pandas as pd
import os
import unicodedata
from matplotlib.colors import from_levels_and_colors
import cmocean as cmo
import seaborn as sn

# Adjust fontsize/weight
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 7}
plt.rc('font', **font)

YEAR_MIN = 1950
YEAR_MAX = 2023
#clim_year = [1981, 2010]
clim_year = [1991, 2020]
width = 0.7

# Load climate index
climate_index = pd.read_csv('NL_climate_index_all_fields.csv')
climate_index = climate_index.set_index('Year')
climate_index_sc = climate_index.copy() # for scorecards at top

# Mean index.
climate_index_mean = climate_index.mean(axis=1)

## restrict time series and normalize for plots.
climate_index = climate_index[climate_index.index<YEAR_MAX+1]
climate_index_norm = climate_index.divide((10 - climate_index.isna().sum(axis=1)).values, axis=0)
climate_index_norm_ns = climate_index_sc.divide((10 - climate_index_sc.isna().sum(axis=1)).values, axis=0)

# Breaking points based on the NLCI:
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

#### ----- Climate index with simple Scorecards ---- ####
# Build the colormap - Stack
from matplotlib.colors import from_levels_and_colors
YlGn = plt.cm.YlGn(np.linspace(0,1, num=12))
YlGn = YlGn[4:,]
cmap_stack, norm_stack = from_levels_and_colors(np.arange(0,7), YlGn, extend='both') 
# Build the colormap - Scorecard
vmin = -3.49
vmax = 3.49
midpoint = 0
levels = np.linspace(vmin, vmax, 15)
midp = np.mean(np.c_[levels[:-1], levels[1:]], axis=1)
colvals = np.interp(midp, [vmin, midpoint, vmax], [-1, 0., 1])
normal = plt.Normalize(-3.49, 3.49)
reds = plt.cm.Reds(np.linspace(0,1, num=7))
blues = plt.cm.Blues_r(np.linspace(0,1, num=7))
whites = [(1,1,1,1)]*2
colors = np.vstack((blues[0:-1,:], whites, reds[1:,:]))
colors = np.concatenate([[colors[0,:]], colors, [colors[-1,:]]], 0)
cmap, norm = from_levels_and_colors(levels, colors, extend='both')
cmap_r, norm_r = from_levels_and_colors(levels, np.flipud(colors), extend='both')

# Plot
fig, ax = plt.subplots(nrows=1, ncols=1)
n = 5 # xtick every n years
ax = climate_index_norm.plot(kind='bar', stacked=True, cmap='nipy_spectral', zorder=10)
ticks = ax.xaxis.get_ticklocs()
ticklabels = [l.get_text() for l in ax.xaxis.get_ticklabels()]
ax.xaxis.set_ticks(ticks[::n])
ax.xaxis.set_ticklabels(ticklabels[::n])
# Add regimes
for idx, years in enumerate(years_list):
    c = plt.fill_between([years[0], years[1]], [-1.55, -1.55], [1.55, 1.55], facecolor=years_colors[idx], alpha=.2, zorder=-1)
plt.grid('on')
ax.set_xlabel(r'')
# legend inside:
ax.legend(loc='upper left', fontsize=6)
ax.set_ylim([-1.55, 1.55])

colors = cmap(normal(np.nansum(climate_index_norm.values, axis=1)))
nlci_text = np.nansum(climate_index_norm.values, axis=1).round(1).astype('str')
if nlci_text[0] == '0.0':
    nlci_text[0] = 'nan'
the_table = ax.table(cellText=[nlci_text],
        rowLabels=['NLCI '],
        colLabels=None,
        cellColours = [colors],
        cellLoc = 'center', rowLoc = 'center',
        loc='bottom', bbox=[0, -0.14, 1, 0.05])
the_table.auto_set_font_size (False)
the_table.set_fontsize(5.5)

for key, cell in the_table.get_celld().items():
    cell_text = cell.get_text().get_text() 
    if key[1] == -1:
        cell.set_linewidth(0)
        cell.set_fontsize(7)
    elif cell_text=='nan':
        cell.set_linewidth(0.1)
        cell._set_facecolor('darkgray')
        cell._text.set_color('darkgray')
        cell.set_fontsize(0)
    else:
        cell.set_linewidth(0.1)
        cell._text.set_rotation(90)

## Save Fig.
fig = ax.get_figure()
fig.set_size_inches(w=7.09,h=5.3)
fig_name = 'NL_climate_index_ms.png'
fig.savefig(fig_name, dpi=300)
os.system('convert -trim -bordercolor White -border 10x10 ' + fig_name + ' ' + fig_name)
plt.savefig('NL_climate_index_ms.pdf', format='pdf')
