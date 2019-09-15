import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
from sklearn.preprocessing import StandardScaler, normalize
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, TimeSeriesSplit
from scipy.stats import norm
import sklearn.datasets
import sklearn.linear_model as lm
import scipy

class SPACE():
    """
    Estimates a partial correlation matrix using the SPACE method
    proposed by Peng et al
    
    See
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2770199/
    for more information

    Attributes
    ----------
    l1_reg : float 
        Lasso regularization parameter
    l2_reg : float
        L2 regularization parameter
    partial_correlation : p by p array
        Estimated partial correlation matrix
    precision_ : p by p array
        Estimated precision matrix
    sig_ : p by 1 array
        Estimated noise in the problem (inverse of the diagonal of the 
        precision matrix)
    weight_ : p by 1 array 
        Weight to give various nodes - currently not implemented
    iter_ : int (default 2)
        Number of iterations to run the method for (2/3 is usually sufficient)
    solver_: 'c' or 'python' (default 'c')
        Whether to use the C code provided by the authors or a Python solver
        for the problem (please note the Python solver can be very slow and use
        a lot of memory)
    verbose : bool
        Whether to dump more information to the console
    """
    def __init__(self, l1_reg, l2_reg=0, sig=None, weight=None, iter=2, solver='c', verbose=False):
        self.sig_ = sig
        self.weight_ = weight
        self.partial_correlation_ = None
        self.precision_ = None
        self.iter_ = iter
        self.solver_ = solver
        self.verbose_ = verbose

        # This is mostly magic so we can call the C code the authors have 
        # kindly provided
        self.l1_reg_ = l1_reg
        self.l2_reg_ = l2_reg
        self.l1_reg_ctype_ = ctypes.c_float(self.l1_reg_)
        self.l2_reg_ctype_ = ctypes.c_float(self.l2_reg_)

        self.lib_ = ctypes.CDLL("jsrm.so")   
        self.fun_ = self.lib.JSRM
        self.fun.restype_ = None
        self.fun.argtypes_ = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float), 
            ctypes.POINTER(ctypes.c_float), ndpointer(ctypes.c_float),ndpointer(ctypes.c_float), ctypes.POINTER(ctypes.c_int), 
            ctypes.POINTER(ctypes.c_int), ndpointer(ctypes.c_float)]


    def _run_jsrm(self, X):
        """
        Calls the JSRM c code and returns the precision matrix it estimates out
        Parameters
        ----------
        X : array_like
            n by p matrix - data matrix

        Returns
        -------
        p by p precision matrix estimated by the JSRM model
        """
        X = X.copy()
        n, p = X.shape
        n_in = ctypes.c_int(n)
        p_in = ctypes.c_int(p)
        sigma_sr = np.sqrt(self.sig_).astype(np.float32)
        n_iter = ctypes.c_int(500)
        iter_count = ctypes.c_int(0)
        beta = np.zeros(p**2, dtype=np.float32)
        n_iter_out = ctypes.c_int(0)

        X = X.astype(np.float32)
        X_in = X.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        sigma_sr_in = sigma_sr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        beta_out = beta.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        space_prec = np.reshape(beta, (p, p))
        self.fun(ctypes.byref(n_in), ctypes.byref(p_in), ctypes.byref(self.l1_reg_ctype), ctypes.byref(self.l2_reg_ctype), X, sigma_sr, ctypes.byref(n_iter), ctypes.byref(n_iter_out), beta)
        space_prec = np.reshape(beta, (p, p))

        return space_prec

    def fit(self, X):
        """
        Solves a scaled lasso problem for X and y
        Parameters
        ----------
        X : array_like
            n by p matrix - data matrix
        y : array_like
            n by 1 matrix - produced values

        Returns
        -------
        """
        n, p = X.shape
        self.sig_ = np.ones(p, dtype=np.float32)

        for i in range(self.iter_):
            if self.solver_ == 'c':
                self.partial_correlation_ = self.run_jsrm(X)
            elif self.solver_ == 'python':
                lasso = lm.Lasso(self.l1_reg/(X_inp.shape[0]))
                lasso.fit(X_inp, y_inp)
                self.partial_correlation_[ind] = lasso.coef
                self.partial_correlation_[ind] += self.partial_correlation_[ind]
            np.fill_diagonal(self.partial_correlation_, 1)
            ind = np.triu_indices(p)
            coef = self.partial_correlation_[ind]
            self.precision_ = self.create_precision_matrix_estimate(coef)
            self.sig_ = self.calculate_noise(X, beta)   

    def calculate_noise (self, X):
        """
        Calculates the noise for this estimate of the off-diagonal
        of the precision matrix 
        Parameters
        ----------
        X : array_like
            n by p matrix - data matrix
        beta : array_like
            p by p vector - estimate of the off-diagonal of the precision matrix  

        Returns
        -------
        p by 1 vector containing the noise for each variable
        """
        n,p = X.shape
        np.fill_diagonal(beta, 0)
        X_hat = X @ self.precision_
        residue = X - X_hat
        result = np.power(residue, 2).mean(axis=0)
        return np.reciprocal(result)

    def create_precision_matrix_estimate(self, coef):
        """
        Creates the estimate of the precision matrix given the 
        off diagonal estimates (coef)
        Parameters
        ----------
        coef : array_like
            p by p matrix containing the estimates of the off-diagonal 
            values 

        Returns
        -------
        p by 1 vector containing the noise for each variable
        """
        p = self.sig_.shape[0]
        result = np.zeros((p, p), dtype=np.float32)
        ind = np.triu_indices(p)

        result[ind] = coef
        result = result + result.T
        reciprocal_diag_sig_sqrt = np.diag(np.reciprocal(np.sqrt(self.sig_)))
        diag_sig_sqrt = np.diag(np.sqrt(self.sig_))
        result = reciprocal_diag_sig_sqrt @ result @ diag_sig_sqrt
        result = result.T 
        return result.astype(np.float32)

    def build_input(self, X):
        """
        Builds two large matrices new_X and new_Y that we can put into
        a normal lasso solver to solve the SPACE problem. WARNING: These
        could be very large for anything other than trivial problems

        Parameters
        ----------
        X : array_like
            n by p data matrix

        Returns
        -------
        (new_X, new_y) - X and y matrices to input into a lasso solver
        """
        n, p = X.shape
        new_n = n * p
        new_p = int(p * (p - 1)/2)
        #new_X = np.zeros((new_n, new_p))
        new_X = scipy.sparse.dok_matrix((new_n, new_p))
        #new_X = sp.sparse.lil_matrix((new_n, new_p))
        indices = np.arange(p)

        if self.sig_ is None:
            self.sig_ = np.ones(p)

        x = 0
        for i in range(p):
            for j in range(i+1, p):
                #if i == j:
                #    continue
                new_col = np.zeros((n*p, 1))
                new_col[i*n:(i+1)*n] = np.sqrt(self.sig_[j]/self.sig_[i]) * X[:, j].reshape((n, 1))
                new_col[j*n:(j+1)*n] = np.sqrt(self.sig_[i]/self.sig_[j]) * X[:, i].reshape((n, 1))

                new_X[:, x] = new_col
                x += 1
        new_y = X.flatten(order='F')

        return new_X, new_y

class SPACE_BIC():
    """
    Estimates a set of partial correlation matrices using the SPACE method
    proposed by Peng et al and selects the one with the lowest BIC
    
    See
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2770199/
    for more information

    Attributes
    ----------
    alpha_ : float 
        Lasso regularization parameter that gives the lowest BIC
    l2_reg : float
        L2 regularization parameter
    partial_correlation : p by p array
        Estimated partial correlation matrix with the lowest BIC
    sig_ : p by 1 array
        Estimated noise in the problem (inverse of the diagonal of the 
        precision matrix)
    alphas_ : array_like
        Array of l1 regularization values to run
    verbose : bool
        Whether to dump more information to the console
    """
    def __init__(self, verbose=False, l2_reg=0, alphas=None):
        self.verbose = verbose
        self.outputs_ = None
        self.precision_ = None
        self.l2_reg = l2_reg
        self.alpha_ = None
        
        if alphas is None:
            self.alphas_ = np.linspace(1, 200)
        else:
            self.alphas_ = alphas

    def _run(self, X, l1_reg, l2_reg=0):
        """
        Runs the SPACE problem and for the specified X and 
        regularization parameters

        Parameters
        ----------
        X : array_like
            n by p data matrix

        l1_reg : float
            L1 regularization parameter

        l2_reg : float
            L2 regularization parameter

        Returns
        -------
        (precision matrix, noise vector) - tuple containing a p by p precision
        matrix estimate and a p by 1 noise vector 
        """
        if self.verbose:
            print("Running %s" % l1_reg)
        s = SPACE(l1_reg, l2_reg)
        s.fit(X)
        return s.precision_, s.sig_

    def fit(self, X):
        """
        Fits the SPACE problem to X and finds the precision 
        matrix with the lowest BIC

        Parameters
        ----------
        X : array_like
            n by p data matrix

        Returns
        -------
        """
        n, p = X.shape

        rerun = True

        while rerun:
            if self.verbose:
                print("Lambda limits are from %s to %s" % (self.alphas_[0], self.alphas_[-1]))

            outputs = Parallel(n_jobs=4)(delayed(self._run)(X, l, self.l2_reg) for l in self.alphas_)
            bics = []
            for prec, sig in outputs:
                error = self.bic(X, prec, sig)
                bics.append(error)
            bics = np.array(bics)
            min_err_i = np.argmin(bics)
            best_l = self.alphas_[min_err_i]
            rerun = False
            if best_l == self.alphas_[0]:
                rerun = True
                print("WARNING: lambda is at the minimum value. It might be worth rerunning with a different set of alphas")
                min_l = min_l*0.1
                max_l = max_l*0.1
            
            if best_l == self.alphas_[-1]:
                rerun = True
                print("WARNING: lambda is at the maximum value. It might be worth rerunning with a different set of alphas")
                min_l = min_l*10
                max_l = max_l*10

            self.alphas_ = np.linspace(min_l, max_l)


        if self.verbose:
            print("Best lambda is at %s" % best_l)
            #print(bics)
        self.alpha_ = best_l
        self.precision_ = outputs[min_err_i][0] 
        self.outputs_ = outputs       

    def bic(self, X, prec, sig):
        """
        Calculates the BIC for the given SPACE solution

        Parameters
        ----------
        X : array_like
            n by p data matrix
        prec : array_like
            p by p estimate of the precision matrix
        sig : array_like
            p by 1 estimate of the noise

        Returns
        -------
        float - BIC of the problem
        """
        n, p = X.shape
        total_bic = 0
        indices = np.arange(p)
        for i in range(p):
            rss_i = 0
            predict = 0
            vec_rss_i = 0
            for j in range(p):
                if i == j:
                    continue
                predict += prec[i, j] * np.sqrt(sig[j] / sig[i]) * X[:, j]     
            residual = np.power(X[:, i] - predict, 2).sum()

            k = np.count_nonzero(prec[indices!=i, i])
            total_bic += n * np.log(residual) + np.log(n) * k

        return total_bic

if __name__=="__main__":
    # A trivial example to show the methods
    n = 10 
    p = 5
    P = sklearn.datasets.make_sparse_spd_matrix(dim=p, alpha=0.7, smallest_coef=.4, largest_coef=.7, norm_diag=True)
    C = np.linalg.inv(P)
    X = np.random.multivariate_normal(np.zeros(p), C, n)
    ss = StandardScaler()
    X = ss.fit_transform(X)
    S = np.cov(X.T)
    off_diag_ind = ~np.eye(p, dtype=bool)
    max_l = n*np.abs(S[off_diag_ind]).max()
    space = SPACE(max_l)
    space.fit(X)
    print(space.precision_)
    print(space.python_solve(X))

    space_bic = SPACE_BIC_Python(verbose=True)
    space_bic.fit(X)
    print(space_bic.precision_)

    space_bic = SPACE_BIC(verbose=True)
    space_bic.fit(X)
    print(space_bic.precision_)
    print(space_bic.alpha_)

    prec = space_r.run(X, max_l)
    print(prec[0])