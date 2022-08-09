from functions.Data_Processing import *
from functions.Machine_Learning import *

os.chdir(r'C:\Users\rapon\Documents\UNI\Tesi Magistrale\Python\tesiRaponi')
low_P = 0.60
high_P = 0.75

TOTAL = True
if TOTAL:
    data = pd.read_csv('Data/Processed/Corsini2021/Corsini2021_Processed_ON.csv')
    # data = pd.read_csv('Data/Processed/Corsini2021/Acc/Corsini2021_Acc02.csv')
    rel_pwr = data['PwrTOT_rel'].values
    LowP_data = filter_data(data, None, None, 0, low_P)
    HighP_data = filter_data(data, None, None, high_P, 1)
    low_perc = len(LowP_data) / len(data) * 100
    high_perc = len(HighP_data) / len(data) * 100
    text_lowP = 'P < ' + str(low_P) + ' = ' "{:.1f}".format(low_perc) + '%'
    text_highP = 'P > ' + str(high_P) + ' = ' "{:.1f}".format(high_perc) + '%'

    # This is  the colormap I'd like to use.
    cm = plt.cm.get_cmap('RdYlBu_r')

    fig, ax = plt.subplots(figsize=(18, 9))

    _, bins, patches = ax.hist(rel_pwr, bins=200, density=False, range=(0.4, 0.9))
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))

    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(0.4, 0.9, 0.05))

    locs = ax.get_yticks()
    ax.set_yticks(locs, np.round(locs / len(rel_pwr) * 100, 1))
    ax.set_ylabel('Percentuale [%]')
    ax.axvline(x=low_P, color='black', linestyle='dashed', linewidth=1, alpha = 0.5)
    ax.text(0.63, len(data)*0.046, text_lowP, fontsize=13, color='blue', ha='left')
    ax.text(0.63, len(data)*0.05, text_highP, fontsize=13, color='red', ha='left')

    ax.axvline(x=high_P, color='black', linestyle='dashed', linewidth=1, alpha = 0.5)
    plt.show()
