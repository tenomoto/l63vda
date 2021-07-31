import numpy as np
import matplotlib.pyplot as plt


grad = np.loadtxt("g01.txt")
grad = grad.reshape([grad.size//2, 2])

fig, ax = plt.subplots()
ax.plot(grad[:, 0], grad[:, 1])
fig.savefig("grad.png", bbox_inches="tight", dpi=300)
fig.savefig("grad.pdf", bbox_inches="tight")
#plt.show()

