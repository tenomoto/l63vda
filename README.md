This repository contains the Python version of the Lorenz-63 model in Huang and Yang (1996).
The code was used in [the 25-th data assimilation summer school in 2021](http://jmsfmml.or.jp/j/event/summerschool/25.html) held online in Japan.

# Jupyter notebooks

Note that initial condition files need to be created manually in Binder.

* Variational data assimilation
 - original version generating files: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tenomoto/l63vda/HEAD?filepath=l63vda.ipynb) [nbviewer](https://nbviewer.jupyter.org/github/tenomoto/l63vda/blob/master/l63vda.ipynb)
 - modified version without files: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tenomoto/l63vda/HEAD?filepath=l63vda_mem.ipynb) [nbviewer](https://nbviewer.jupyter.org/github/tenomoto/l63vda/blob/master/l63vda_mem.ipynb)
* Ensemble Kalman filter with perturbed observations [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tenomoto/l63vda/HEAD?filepath=l63po.ipynb) [nbviewer](https://nbviewer.jupyter.org/github/tenomoto/l63vda/blob/master/l63po.ipynb)
* Neural network [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tenomoto/l63vda/HEAD?filepath=l63nn.ipynb) [nbviewer](https://nbviewer.jupyter.org/github/tenomoto/l63vda/blob/master/l63nn.ipynb)
* Autmatic differentiation with [JAX](https://jax.readthedocs.io/en/latest/index.html) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tenomoto/l63vda/HEAD?filepath=l63jax.ipynb) [nbviewer](https://nbviewer.jupyter.org/github/tenomoto/l63vda/blob/master/l63jax.ipynb)

# References

* Huang, X.-Y. and X. Yang, 1996: Variational data assimilation with the Lorenz model. [Technical Report 26](http://hirlam.org/index.php/publications-54/hirlam-technical-reports-a/doc_view/1317-hirlam-technical-report-no-26), HIRAM, April 1996, 44 pp.
