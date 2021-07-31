import numpy as np
import matplotlib.pyplot as plt


fname = "m01c000.npy"
x = np.load(fname)
x[:, 1] += 10
x[:, 2] += 20
fig, ax = plt.subplots()
ax.plot(x)
plt.gca().set_prop_cycle(None)
fname = "m02c000.npy"
x = np.load(fname)
x[:, 1] += 10
x[:, 2] += 20
ax.plot(x, linestyle="--")
fig.savefig("control.png", bbox_inches="tight", dpi=300)
fig.savefig("control.pdf", bbox_inches="tight")
#plt.show()

