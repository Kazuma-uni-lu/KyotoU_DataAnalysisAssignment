from IPython.display import display,Markdown #,HTML
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import pandas as pd
import parse_data

def display_title(s, pref='Figure', num=1, center=False):
    ctag = 'center' if center else 'p'
    s    = f'<{ctag}><span style="font-size: 1.2em;"><b>{pref} {num}</b>: {s}</span></{ctag}>'
    if pref=='Figure':
        s = f'{s}<br><br>'
    else:
        s = f'<br><br>{s}'
    display( Markdown(s) )


df = parse_data.parse()
#df.describe()

def central(x, print_output=True):
    x0     = np.mean( x )
    x1     = np.median( x )
    x2     = stats.mode( x ).mode
    return x0, x1, x2


def dispersion(x, print_output=True):
    y0 = np.std( x ) # standard deviation
    y1 = np.min( x )  # minimum
    y2 = np.max( x )  # maximum
    y3 = y2 - y1      # range
    y4 = np.percentile( x, 25 ) # 25th percentile (i.e., lower quartile)
    y5 = np.percentile( x, 75 ) # 75th percentile (i.e., upper quartile)
    y6 = y5 - y4 # inter-quartile range
    return y0,y1,y2,y3,y4,y5,y6
def display_central_tendency_table(num=1):
    display_title('Central tendency summary statistics.', pref='Table', num=num, center=False)
    df_central = df.apply(lambda x: central(x), axis=0)
    round_dict = {'quality': 3, 'acidity': 3, 'density': 6, 'sugar': 3}
    df_central = df_central.round( round_dict )
    row_labels = 'mean', 'median', 'mode'
    df_central.index = row_labels
    display( df_central )

#display_central_tendency_table(num=1)
def display_dispersion_table(num=1):
    display_title('Dispersion summary statistics.', pref='Table', num=num, center=False)
    round_dict            = {'quality': 3, 'acidity': 3, 'density': 6, 'sugar': 3}
    df_dispersion         = df.apply(lambda x: dispersion(x), axis=0).round( round_dict )
    row_labels_dispersion = 'st.dev.', 'min', 'max', 'range', '25th', '75th', 'IQR'
    df_dispersion.index   = row_labels_dispersion
    display( df_dispersion )

#display_dispersion_table(num=2)
y    = df['Bike Count']
hour = df['Hour']
temp = df['Temperature']
humd = df['Humidity']
wind = df['Wind Speed']
visi = df['Visibility']
solr = df['Solar Radiation']
rain = df['Rainfall']
snow = df['Snowfall']
hday = df['Holiday']
fday = df['Functioning Day']
month = df['Month']
#fig, axs = plt.subplots(2, 6, figsize=(18, 6), tight_layout=True)
#axs = axs.flatten()
#axs[0].scatter(hour, y, alpha=0.5, color='black' )
#axs[1].scatter(temp, y, alpha=0.5, color='r' )
#axs[2].scatter(humd, y, alpha=0.5, color='b' )
#axs[3].scatter(wind, y, alpha=0.5, color='green')
#axs[4].scatter(visi, y, alpha=0.5, color='purple')
#axs[5].scatter(solr, y, alpha=0.5, color='orange')
#axs[6].scatter(rain, y, alpha=0.5, color='cyan')
#axs[7].scatter(snow, y, alpha=0.5, color='brown')
#axs[8].scatter(hday, y, alpha=0.5, color='magenta')
#axs[9].scatter(fday, y, alpha=0.5, color='gray')
#axs[10].scatter(month, y, alpha=0.5, color='gold')
#xlabels = 'Hour', 'Temperature', 'Humidity', 'Wind Speed', 'Visibility', 'Solar Radiation', 'Rainfall', 'Snowfall', 'Holiday', 'Functioning Day', 'Month'
#[ax.set_xlabel(s) for ax,s in zip(axs,xlabels)]
#axs[0].set_ylabel('Bikes')
#[ax.set_yticklabels([])  for ax in axs[1:]]
#
#
#axs[8].set_xticks([0, 1])
#axs[9].set_xticks([0, 1])
#axs[10].set_xticks(np.arange(1, 13, 1))
#plt.show()
def corrcoeff(x, y):
    r = np.corrcoef(x, y)[0,1]
    return r
def plot_regression_line(ax, x, y, **kwargs):
    a,b   = np.polyfit(x, y, deg=1)
    x0,x1 = min(x), max(x)
    y0,y1 = a*x0 + b, a*x1 + b
    ax.plot([x0,x1], [y0,y1], **kwargs)
def plot_descriptive():    
    fig, axs = plt.subplots(2, 6, figsize=(18, 6), tight_layout=True)
    axs = axs.flatten()
    ivs     = [hour, temp, humd, wind, visi, solr, rain, snow, hday, fday, month]
    colors  = 'black' , 'r' , 'b' , 'green', 'purple', 'orange', 'cyan', 'brown', 'magenta', 'gray', 'gold'
    for ax,x,c in zip(axs, ivs, colors):
        ax.scatter( x, y, alpha=0.5, color=c )
        plot_regression_line(ax, x, y, color='k', ls='-', lw=2)
        r   = corrcoeff(x, y)
        ax.text(0.7, 0.3, f'r = {r:.3f}', color=c, transform=ax.transAxes, bbox=dict(color='0.8', alpha=0.7))
    xlabels = 'Hour', 'Temperature', 'Humidity', 'Wind Speed', 'Visibility', 'Solar Radiation', 'Rainfall', 'Snowfall', 'Holiday', 'Functioning Day', 'Month'
    [ax.set_xlabel(s) for ax,s in zip(axs,xlabels)]
    axs[0].set_ylabel('Bikes')
    axs[6].set_ylabel('Bikes')
    axs[8].set_xticks([0, 1])
    axs[9].set_xticks([0, 1])
    axs[10].set_xticks(np.arange(1, 13, 1))
    panel_labels = 'a b c d e f g h i j k'.split()
    for ax, label in zip(axs, panel_labels):
        ax.text(
            0.02, 0.95, f'({label})',
            transform=ax.transAxes,
            fontsize=12,
            fontweight='bold',
            va='top'
        )
    plt.show()
    display_title('Correlations amongst main variables.', pref='Figure', num=1)

plot_descriptive()

