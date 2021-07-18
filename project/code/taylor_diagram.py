import skill_metrics as sm
import numpy as np
import matplotlib.pyplot as plt

def taylor_diagram(std_obs, metrics, title, savename):

    keys = metrics.keys()
    columns = metrics[next(iter(keys))].columns.values
    n_horizons = len(columns)

    STD = [std_obs] + [metrics[k].loc['std',c] for k in keys for c in columns]
    RMSD = [0] + [metrics[k].loc['rmsd',c] for k in keys for c in columns]
    CORR = [1] + [metrics[k].loc['corr_coef',c] for k in keys for c in columns]

    STD = np.array(STD)
    RMSD = np.array(RMSD)
    CORR = np.array(CORR)
    
    markerLabel = ['O'] + [f'{k}-{h}' for k in keys for h in range(1,n_horizons+1)]

    plt.figure(figsize=(15, 8))
    plt.rc('font', size=14)

    sm.taylor_diagram(STD, RMSD, CORR, markerobs = 'o', styleobs = '-', 
                    titleobs = 'obs', colcor = 'k', 
                    widthcor= 0.5, widthrms = 1.3, colrms = 'g', 
                    markerLabel = markerLabel,
                    alpha=0.7, checkStats='on', tickrmsangle=135, 
                    markersize=10, colobs='#fc03df')
    plt.title(title, y=1.05)
    plt.savefig(f"images/{savename}.jpg", dpi=300)