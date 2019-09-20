
# Network Inference Toolkit

A toolkit for inferring regularization partial correlation networks

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

You'll need numpy and sklearn to run 

### Installing

This can be installed using pip

```
pip install numpy
pip install sklearn
pip install nitk
```

To test if the installation was successful, create a new Python script 
```
from nitk import scaled_lasso
import numpy as np
from sklearn.datasets import make_sparse_spd_matrix

p = 50
n = 20
K = make_sparse_spd_matrix(p)
C = np.linalg.inv(K)
X = np.random.multivariate_normal(np.zeros(p), C, n)
sli = scaled_lasso.ScaledLassoInference()
sli.fit(X)
print(sli.precision_)
print(np.count_nonzero(sli.precision_))
```
See if that works 



## Authors

* **Tristan Millington**

## License

This project is licensed under the GNU GPL - see the [LICENSE.md](LICENSE.md) file for details

## Implemented:
*SPACE - Partial Correlation Estimation by Joint Sparse Regression Models  by Peng, Wang and Zhu - https://doi.org/10.1198/jasa.2009.0126
*SCIO - Fast and adaptive sparse precision matrix estimation in high dimensions - Liu and Luo - https://doi.org/10.1016/j.jmva.2014.11.005
*CLIME - A Constrained L1 Minimization Approach to Sparse Precision Matrix Estimation - Cai, Liu and Luo - https://doi.org/10.1198/jasa.2011.tm10155
*DTrace - Sparse precision matrix estimation via lasso penalized D-trace loss - Zou and Zhang - https://doi.org/10.1093/biomet/ast059
*Correlation Permutation - Estimates a sparse correlation matrix by permuting the dataset repeatedly to get a p-value to see if
the correlation between two variables is just as likely to occur through noise 
*Scaled Lasso - "Sparse Matrix Inversion with Scaled Lasso" by Sun and Zhang - http://www.jmlr.org/papers/volume14/sun13a/sun13a.pdf

