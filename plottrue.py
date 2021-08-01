import numpy as np
import matplotlib.pyplot as plt


dt = 0.01
nstop = 500
x = np.linspace(0, dt*nstop, nstop+1)
fname = "m01c000.npy"
w = np.load(fname)
off1, off2 = 40, 50
w[:, 1] += off1
w[:, 2] += off2
fig, ax = plt.subplots()
ax.plot(x, w)
plt.gca().set_prop_cycle(None)
fname = "m02c000.npy"
w = np.load(fname)
w[:, 1] += off1
w[:, 2] += off2
ax.plot(x, w, linestyle="--")
fig.savefig("control.png", bbox_inches="tight", dpi=300)
fig.savefig("control.pdf", bbox_inches="tight")
#plt.show()

