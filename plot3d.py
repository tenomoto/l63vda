import numpy as np
import matplotlib.pyplot as plt


dt = 0.01
nstop = 500
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
for nexp in range(1,3):
    fname = f"m{nexp:02}c000.npy"
    w = np.load(fname)
    ax.scatter(*w[0, :], c="red")
    ax.plot(*w.transpose(), label=f"nexp{nexp:02}")
ax.legend()
fig.savefig("control3d.png", bbox_inches="tight", dpi=300)
fig.savefig("control3d.pdf", bbox_inches="tight")
plt.show()

