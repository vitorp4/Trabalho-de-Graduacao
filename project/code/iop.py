import matplotlib.pyplot as plt

def iop(metrics, stat, central, model, savename):

    p = metrics['P']
    m = metrics['M']
    n_horizons = p.shape[1]

    iop = []
    for h in range(1,n_horizons+1):
        pers_stat = p.loc[stat,f't+{h}']
        model_stat = m.loc[stat,f't+{h}']
        iop.append(100*(pers_stat-model_stat)/pers_stat)

    plt.figure(figsize=(8, 2.5))
    plt.plot(range(1,n_horizons+1), iop, marker='x', ms=5, label=model)
    plt.xticks(range(1,n_horizons+1))
    plt.grid(axis='y', ls='-.')
    plt.ylabel(f"IoP (%) {stat}")
    plt.xlabel('Horizonte')
    plt.legend()
    plt.title(f"Central {central}")
    plt.savefig(f"images/{savename}.jpg", dpi=300, bbox_inches='tight')