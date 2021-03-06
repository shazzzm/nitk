
# Network Inference Toolkit

A toolkit for inferring regularized partial correlation networks

## Getting Started

### Prerequisites

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

## Testing

If you're interested in running the tests they can be found in the /tests/ folder and are to
be run with nose2. 

If the authors have provided an implementation, we test the python version against that. This
usually requires that you have R installed and the appropriate packages. We then use rpy2 to 
interface between R and Python. 

## Authors

* **Tristan Millington**

## License

This project is licensed under the GNU GPL - see the [LICENSE.md](LICENSE.md) file for details

## Implemented:
* SPACE - Partial Correlation Estimation by Joint Sparse Regression Models  by Peng, Wang and Zhu - https://doi.org/10.1198/jasa.2009.0126
* SCIO - Fast and adaptive sparse precision matrix estimation in high dimensions - Liu and Luo - https://doi.org/10.1016/j.jmva.2014.11.005
* CLIME - A Constrained L1 Minimization Approach to Sparse Precision Matrix Estimation - Cai, Liu and Luo - https://doi.org/10.1198/jasa.2011.tm10155
* DTrace - Sparse precision matrix estimation via lasso penalized D-trace loss - Zou and Zhang - https://doi.org/10.1093/biomet/ast059
* Correlation Permutation - Estimates a sparse correlation matrix by permuting the dataset repeatedly to get a p-value to see if the correlation between two variables is just as likely to occur through noise 
* Scaled Lasso - "Sparse Matrix Inversion with Scaled Lasso" by Sun and Zhang - http://www.jmlr.org/papers/volume14/sun13a/sun13a.pdf
* Neighbourhood Selection - "High-dimensional graphs and variable selection with the Lasso" - https://projecteuclid.org/euclid.aos/1152540754#info

