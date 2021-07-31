import numpy as np
import matplotlib.pyplot as plt
from l63vda import nstop, calc_cost


costc = calc_cost(1, 0, nstop)
cost = np.loadtxt("j01i02.txt").reshape([100, 2])

fig, ax = plt.subplots()
ax.semilogy(cost[:, 0], cost[:, 1])
ax.plot(cost[:, 0], [costc]*cost[:, 0].size, c="gray", linestyle="--")
fig.savefig("cost.png", bbox_inches="tight", dpi=300)
fig.savefig("cost.pdf", bbox_inches="tight")
#plt.show()

