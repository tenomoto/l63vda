import numpy as np
import matplotlib.pyplot as plt
from l63vda import dt, nstop, miss


cycles = [1, 2, 3, 5, 10, 30]
x = np.linspace(0, dt * nstop, nstop+1) 

nx, ny = 2, 3
off1, off2 = 40, 60
fig, ax = plt.subplots(nx, ny, figsize=[12, 9])
wtrue = np.load("m01c000.npy")
wtrue[:, 1] += off1
wtrue[:, 2] += off2
wobs = np.load("o01.npy")
wobs[:, 1] += off1
wobs[:, 2] += off2

k = 0
for ncyc in cycles:
    i, j = k // ny, k % ny
    k += 1
    w = np.load(f"m01c{ncyc:03}.npy")
    w[:, 1] += off1
    w[:, 2] += off2
    ax[i, j].plot(x, wtrue, c="gray", linestyle="--", label=["true","",""])
    ax[i, j].plot(x, w, label=[f"cycle {ncyc:02}","",""])
    ax[i, j].set_prop_cycle(None)
    ax[i, j].scatter(x, wobs[:,0], marker="+", label="obs")
    ax[i, j].scatter(x, wobs[:,1], marker="+")
    ax[i, j].scatter(x, wobs[:,2], marker="+")
    ax[i, j].set_xlim([0.0, 2.0])
    ax[i, j].set_ylim([-40, 160])
    ax[i, j].legend(loc="upper right")
fig.tight_layout()
fig.savefig("cycle.png", bbox_inches="tight", dpi=300)
fig.savefig("cycle.pdf", bbox_inches="tight")
#plt.show()
