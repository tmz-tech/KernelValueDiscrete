#!/usr/bin/env python
# coding: utf-8

#import pandas as pd
#import numpy as np
#from sklearn.base import BaseEstimator

from . import pd, np, BaseEstimator

class KDE(BaseEstimator):
    """
    Kernel Density Estimation (KDE) with Gaussian kernel.

    Parameters
    ----------
    bandwidth : float, optional, default=1.0
        The bandwidth of the kernel, controlling the smoothness of the density estimate.

    Attributes
    ----------
    X_ : array-like, shape (n_samples, n_features)
        The data used for fitting the KDE model.
    """
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth

    def fit(self, X):
        """
        Fit the model using the given data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.X_ = X
        return self

    def score(self, X):
        """
        Calculate the log-likelihood of the data under the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        score : float
            The sum of the log-likelihoods of the KDE density estimates.
        """
        # Use the sum of the log-likelihoods of the KDE density estimates as the score.
        dens = self.gaussian_kernel_matrix(X, self.bandwidth)
        return np.sum(np.log(dens + 1e-10)) # Add a small value to avoid log(0).

    def gaussian_kernel_matrix(self, data_matrix, h_x):
        """
        Calculate the density estimates for the data using Gaussian kernels.

        Parameters
        ----------
        data_matrix : array-like, shape (n_samples, n_features)
            The input data.
        h_x : float
            The bandwidth for the kernel.

        Returns
        -------
        dens : array-like, shape (n_samples,)
            The density estimates for each sample.
        """
        dens = np.apply_along_axis(self.gaussian_kernel, 1, data_matrix, h_x)
        return dens

    def gaussian_kernel(self, x, h_x):
        """
        Apply the Gaussian kernel to a data point.

        Parameters
        ----------
        x : array-like, shape (n_features,)
            The data point to estimate the density for.
        h_x : float
            The bandwidth for the kernel.

        Returns
        -------
        density : float
            The estimated density for the data point.
        """
        # Broadcasting to subtract x from each row of self.X_
        u = (x - self.X_) / h_x
        # Gaussian kernel, applied elementwise
        K = np.exp(-0.5 * np.sum(u**2, axis=1)) / ((2 * np.pi)**(self.X_.shape[1] / 2) * h_x**self.X_.shape[1])
        density = np.mean(K)
        return density



class KDE2(BaseEstimator):
    """
    Kernel Density Estimation (KDE) with Gaussian kernel.

    Parameters
    ----------
    bandwidth : float, optional, default=1.0
        The bandwidth of the kernel.

    Attributes
    ----------
    X_ : array-like, shape (n_samples, n_features)
        The data used for fitting the KDE model.
    """
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth

    def fit(self, X):
        """
        Fit the model using the given data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.X_ = X
        return self

    def score(self, X):
        """
        Calculate the log-likelihood of the data under the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        score : float
            The sum of the log-likelihoods of the KDE density estimates.
        """
        # Use the sum of the log-likelihoods of the KDE density estimates as the score.
        dens = self.gaussian_kernel_matrix(X, self.bandwidth)
        return np.sum(np.log(dens + 1e-10))  # Add a small value to avoid log(0).

    def gaussian_kernel_matrix(self, data_matrix, h_x):
        """
        Calculate the density estimates for the data using Gaussian kernels.

        Parameters
        ----------
        data_matrix : array-like, shape (n_samples, n_features)
            The input data.
        h_x : float
            The bandwidth for the kernel.

        Returns
        -------
        dens : array-like, shape (n_samples,)
            The density estimates for each sample.
        """
        dens = np.apply_along_axis(self.gaussian_kernel, 1, data_matrix, h_x)
        return dens

    def gaussian_kernel(self, x, h_x):
        """
        Apply the Gaussian kernel to a data point.

        Parameters
        ----------
        x : array-like, shape (n_features,)
            The data point to estimate the density for.
        h_x : float
            The bandwidth for the kernel.

        Returns
        -------
        density : float
            The estimated density for the data point.
        """
        u = (x - self.X_) / h_x
        K = np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
        density = np.mean(np.prod(K/h_x, axis=1))
        return density


class est_policy(BaseEstimator):
    
    """
    Estimator for policy using kernel density estimation (KDE) with Gaussian kernel.
    
    Parameters
    ----------
    bandwidth : float, optional, default=1.0
        The bandwidth of the kernel.
    
    Attributes
    ----------
    X_ : array-like, shape (n_samples, n_features)
        The feature matrix used for fitting the model.
    A_ : array-like, shape (n_samples,)
        The action vector corresponding to the feature matrix.
    """
    
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth

    def fit(self, X, A):
        """
        Fit the model using the given feature matrix and action vector.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        A : array-like, shape (n_samples,)
            The action vector.
        
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.X_ = X
        self.A_ = A
        return self
    
    def indicator(self, vec, a):
        """
        Create an indicator vector where elements are True if they match action 'a'.
        
        Parameters
        ----------
        vec : array-like, shape (n_samples,)
            The vector to compare against.
        a : scalar
            The action to compare.
        
        Returns
        -------
        ind : array-like, shape (n_samples,)
            Indicator vector.
        """
        ind = vec == a
        return ind

    def pi_est(self, x, a, h_x):
        """
        Estimate the policy probability for a given 'x' and 'a'.
        
        Parameters
        ----------
        x : array-like, shape (n_features,)
            The data point to estimate the density for.
        a : scalar
            The action for which the policy probability is estimated.
        h_x : float
            The bandwidth for the kernel.
        
        Returns
        -------
        prob : float
            The estimated policy probability.
        """
        u = (x - self.X_) / h_x
        K= np.exp(-0.5*u**2)/np.sqrt(2 * np.pi)
        ind_A = self.indicator(self.A_, a)
        prob_num = np.mean(ind_A*np.prod(K/h_x, axis=1))
        prob_denom = np.mean(np.prod(K/h_x, axis=1))
        prob = prob_num/prob_denom
        return prob
    
    def __call__(self, data_matrix, a):
        """
        Apply pi_est across all rows in the data matrix.
        
        Parameters
        ----------
        data_matrix : array-like, shape (n_samples, n_features)
            The input data.
        a : scalar
            The action for which the policy probability is estimated.
        h_x : float
            The bandwidth for the kernel.
        
        Returns
        -------
        pi_est_vec : array-like, shape (n_samples,)
            Vector of estimated policy probabilities for each sample.
        """
        pi_est_vec = np.apply_along_axis(self.pi_est, 1, data_matrix, a, self.bandwidth)
        return pi_est_vec
    
    
class est_policy2(BaseEstimator):
    
    """
    Estimator for policy using kernel density estimation (KDE) with Gaussian kernel.
    
    Parameters
    ----------
    bandwidth : float, optional, default=1.0
        The bandwidth of the kernel.
    
    Attributes
    ----------
    X_ : array-like, shape (n_samples, n_features)
        The feature matrix used for fitting the model.
    A_ : array-like, shape (n_samples,)
        The action vector corresponding to the feature matrix.
    """
    
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth

    def fit(self, X, A):
        """
        Fit the model using the given feature matrix and action vector.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        A : array-like, shape (n_samples,)
            The action vector.
        
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.X_ = X
        self.A_ = A
        return self
    
    def indicator(self, vec, a):
        """
        Create an indicator vector where elements are True if they match action 'a'.
        
        Parameters
        ----------
        vec : array-like, shape (n_samples,)
            The vector to compare against.
        a : scalar
            The action to compare.
        
        Returns
        -------
        ind : array-like, shape (n_samples,)
            Indicator vector.
        """
        ind = vec == a
        return ind

    def pi_est(self, x, a, h_x):
        """
        Estimate the policy probability for a given 'x' and 'a'.
        
        Parameters
        ----------
        x : array-like, shape (n_features,)
            The data point to estimate the density for.
        a : scalar
            The action for which the policy probability is estimated.
        h_x : float
            The bandwidth for the kernel.
        
        Returns
        -------
        prob : float
            The estimated policy probability.
            
        se : float
            The approximated standard errors for each estimated policy probability.
        """
        u = (x - self.X_) / h_x
        K= np.exp(-0.5*u**2)/np.sqrt(2 * np.pi)
        ind_A = self.indicator(self.A_, a)
        prob_num = np.mean(ind_A*np.prod(K/h_x, axis=1))
        prob_denom = np.mean(np.prod(K/h_x, axis=1))
        prob = prob_num/prob_denom
        
        # Calculation of the approximation of standard error
        se = np.sqrt((np.pi**(self.X_.shape[1]/2)*prob * (1 - prob))/(prob_denom*self.X_.shape[0]*h_x**self.X_.shape[1]))
        
        return prob, se
    
    def __call__(self, data_matrix, a):
        """
        Apply pi_est across all rows in the data matrix.
        
        Parameters
        ----------
        data_matrix : array-like, shape (n_samples, n_features)
            The input data.
        a : scalar
            The action for which the policy probability is estimated.
        h_x : float
            The bandwidth for the kernel.
        
        Returns
        -------
        pi_est_vec : array-like, shape (n_samples,)
            Vector of estimated policy probabilities for each sample.
        se_vec : array-like, shape (n_samples, 2)
            Vector of standard errors for each estimated probability.
        """
        
        results = np.apply_along_axis(self.pi_est, 1, data_matrix, a, self.bandwidth)
        pi_est_vec, se_vec = results[:, 0], results[:, 1]
        
        return pi_est_vec, se_vec
    
    
    


class est_policyData(BaseEstimator):
    
    """
    Estimator for policy using kernel density estimation (KDE) with Gaussian kernel.
    
    Parameters
    ----------
    bandwidth : float, optional, default=1.0
        The bandwidth of the kernel.
    
    Attributes
    ----------
    X_ : array-like, shape (n_samples, n_features)
        The feature matrix used for fitting the model.
    A_ : array-like, shape (n_samples,)
        The action vector corresponding to the feature matrix.
    """
    
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth

    def fit(self, X, A):
        """
        Fit the model using the given feature matrix and action vector.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        A : array-like, shape (n_samples,)
            The action vector.
        
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.X_ = X
        self.A_ = A
        return self
    
    def indicator(self, vec, a):
        """
        Create an indicator vector where elements are True if they match action 'a'.
        
        Parameters
        ----------
        vec : array-like, shape (n_samples,)
            The vector to compare against.
        a : scalar
            The action to compare.
        
        Returns
        -------
        ind : array-like, shape (n_samples,)
            Indicator vector.
        """
        ind = vec == a
        return ind

    def pi_est(self, x, a, h_x):
        """
        Estimate the policy probability for a given 'x' and 'a'.
        
        Parameters
        ----------
        x : array-like, shape (n_features,)
            The data point to estimate the density for.
        a : scalar
            The action for which the policy probability is estimated.
        h_x : float
            The bandwidth for the kernel.
        
        Returns
        -------
        prob : float
            The estimated policy probability.
        """
        
        # Compute the normalized pairwise differences for the state and action
        u_x = (x - self.X_) / h_x # Difference between input state and fitted states
        
        # Apply the Gaussian kernel to the state component
        K_x = np.exp(-0.5 * np.sum(u_x**2, axis=1)) / ((2 * np.pi)**(self.X_.shape[1] / 2) * h_x**self.X_.shape[1])
        
        # Create an indicator vector for samples corresponding to action 'a'
        ind_A = self.indicator(self.A_, a)
        
        # Combine the state and action kernels
        K = K_x*ind_A
        
        # Compute the weighted sum of response values (numerator) and the sum of kernel weights (denominator)
        prob_num = np.sum(K) # Weighted sum of responses
        
        prob_denom = np.sum(K_x) + 1e-10
        
        prob = prob_num/prob_denom
        
        return prob
    
    def __call__(self, data_matrix, a_vec):
        """
        Apply pi_est across all rows in the data matrix with action vector.

        Parameters
        ----------
        data_matrix : array-like, shape (n_samples, n_features)
            The input data.
        a_vec : array-like, shape (n_samples,)
            The action vector for which the policy probability is estimated.

        Returns
        -------
        pi_est_vec : array-like, shape (n_samples,)
            Vector of estimated policy probabilities for each sample.
        """
        # Ensure that data_matrix and a_vec have the same length
        assert data_matrix.shape[0] == a_vec.shape[0], "data_matrix and a_vec must have the same length"

        # Apply pi_est function row by row, passing the corresponding element of a_vec
        pi_est_vec = np.array([self.pi_est(data_matrix[i], a_vec[i], self.bandwidth) for i in range(len(data_matrix))])

        return pi_est_vec 
    

class est_policyData2(BaseEstimator):
    
    """
    Estimator for policy using kernel density estimation (KDE) with Gaussian kernel.
    
    Parameters
    ----------
    bandwidth : float, optional, default=1.0
        The bandwidth of the kernel.
    
    Attributes
    ----------
    X_ : array-like, shape (n_samples, n_features)
        The feature matrix used for fitting the model.
    A_ : array-like, shape (n_samples,)
        The action vector corresponding to the feature matrix.
    """
    
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth

    def fit(self, X, A):
        """
        Fit the model using the given feature matrix and action vector.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        A : array-like, shape (n_samples,)
            The action vector.
        
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.X_ = X
        self.A_ = A
        return self
    
    def indicator(self, vec, a):
        """
        Create an indicator vector where elements are True if they match action 'a'.
        
        Parameters
        ----------
        vec : array-like, shape (n_samples,)
            The vector to compare against.
        a : scalar
            The action to compare.
        
        Returns
        -------
        ind : array-like, shape (n_samples,)
            Indicator vector.
        """
        ind = vec == a
        return ind

    def pi_est(self, x, a, h_x):
        """
        Estimate the policy probability for a given 'x' and 'a'.
        
        Parameters
        ----------
        x : array-like, shape (n_features,)
            The data point to estimate the density for.
        a : scalar
            The action for which the policy probability is estimated.
        h_x : float
            The bandwidth for the kernel.
        
        Returns
        -------
        prob : float
            The estimated policy probability.
        """
        u = (x - self.X_) / h_x
        K= np.exp(-0.5*u**2)/np.sqrt(2 * np.pi)
        ind_A = self.indicator(self.A_, a)
        prob_num = np.mean(ind_A*np.prod(K/h_x, axis=1))
        prob_denom = np.mean(np.prod(K/h_x, axis=1))
        prob = prob_num/prob_denom
        return prob
    
    def __call__(self, data_matrix, a_vec):
        """
        Apply pi_est across all rows in the data matrix with action vector.

        Parameters
        ----------
        data_matrix : array-like, shape (n_samples, n_features)
            The input data.
        a_vec : array-like, shape (n_samples,)
            The action vector for which the policy probability is estimated.

        Returns
        -------
        pi_est_vec : array-like, shape (n_samples,)
            Vector of estimated policy probabilities for each sample.
        """
        # Ensure that data_matrix and a_vec have the same length
        assert data_matrix.shape[0] == a_vec.shape[0], "data_matrix and a_vec must have the same length"

        # Apply pi_est function row by row, passing the corresponding element of a_vec
        pi_est_vec = np.array([self.pi_est(data_matrix[i], a_vec[i], self.bandwidth) for i in range(len(data_matrix))])

        return pi_est_vec 
    
    
    

    
    


class est_r_pi(BaseEstimator):
    """
    Custom estimator using Nadaraya-Watson kernel regression for estimating a response function.

    Parameters
    ----------
    bandwidth : float, default=1.0
        The bandwidth parameter for the Gaussian kernel, which determines the smoothness of the estimate.

    alpha : float, optional, default=0.1
        The regularization parameter to improve numerical stability and prevent underflow in density calculations.

    Attributes
    ----------
    X_ : ndarray of shape (n_samples, n_features)
        The input data used for fitting the model.
        
    R_ : ndarray of shape (n_samples,)
        The response values associated with the input data X_.
    """
    def __init__(self, bandwidth=1.0, alpha=0.1):
        self.bandwidth = bandwidth # Initialize the bandwidth parameter for the kernel
        self.alpha = alpha # Initialize the regularization parameter for numerical stability

    def fit(self, X, R):
        """
        Fit the model using input data X and responses R.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.
        R : ndarray of shape (n_samples,)
            The response values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.X_ = X # Store input data
        self.R_ = R # Store response values
        return self

    def nw_est(self, x, h_x):
        """
        Estimate the function value at a given point using Nadaraya-Watson kernel regression.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            The point at which to estimate the function value.

        h_x : float or ndarray of shape (n_features,)
            The bandwidth parameter(s) for the kernel, controlling the smoothness of the estimate.

        Returns
        -------
        nw : float
            The estimated value at point x based on the weighted average of the response values.
        """
        
        # Calculate the pairwise differences between `x` (current state) and the fitted states `self.X_`
        u_x = (x - self.X_) / h_x # Normalized difference between the input and stored states
        
        # Compute the Gaussian kernel weights for the current state
        # K_x: Kernel function applied to the squared pairwise differences
        K_x = np.exp(-0.5 * np.sum(u_x**2, axis=1)) / ((2 * np.pi)**(self.X_.shape[1] / 2) * h_x**self.X_.shape[1])
        
        K_x += self.alpha # Add regularization to improve stability and avoid numerical issues
    
        # Calculate the numerator (weighted sum of R_) and denominator (sum of weights)
        nw_num = np.sum(self.R_ * K_x)
        
        nw_denom = np.sum(K_x) + 1e-10 # Adding a small value to avoid division by zero 
        
        # Estimate the function value at x by taking the ratio of the weighted sum and the sum of weights
        nw = nw_num/nw_denom
            
            
        return nw
    
    def __call__(self, data_matrix):
        """
        Apply Nadaraya-Watson kernel regression to each row in the data matrix.

        Parameters
        ----------
        data_matrix : ndarray of shape (n_samples, n_features)
            The data points at which to estimate the function values.
        h_x : float or ndarray of shape (n_features,)
            The bandwidth parameter(s) for the kernel.

        Returns
        -------
        nw_est_vec : ndarray of shape (n_samples,)
            The estimated values for each row in the data matrix.
        """
        # Apply the nw_est method to each row in data_matrix
        nw_est_vec = np.apply_along_axis(self.nw_est, 1, data_matrix, self.bandwidth)
        return nw_est_vec
    

class est_r_pi_w(BaseEstimator):
    """
    Custom estimator using Nadaraya-Watson kernel regression with a window parameter.

    Parameters
    ----------
    bandwidth : float, default=1.0
        The bandwidth parameter for the Gaussian kernel, controlling the smoothness of the estimate.

    alpha : float, optional, default=0.1
        The regularization parameter to enhance numerical stability and prevent underflow in density calculations.

    Attributes
    ----------
    X_ : ndarray of shape (n_samples, n_features)
        The input data used for fitting the model.
        
    R_ : ndarray of shape (n_samples,)
        The response values associated with the input data X_.

    w : int
        The window parameter that determines the lag between the response and the input data. It should be set during model fitting.

    R_w : ndarray of shape (n_samples - w,)
        The truncated response values aligned with the window parameter, used for estimating the current state based on previous observations.

    X_w : ndarray of shape (n_samples - w, n_features)
        The truncated input data aligned with the window parameter, excluding the last w samples to maintain consistency with R_w.
    """
    def __init__(self, bandwidth=1.0, alpha = 0.1):
        self.bandwidth = bandwidth # Initialize the bandwidth parameter for the kernel
        self.alpha = alpha # Initialize the regularization parameter for numerical stability

    def fit(self, X, R, w):
        """
        Fit the model using input data X, responses R, and a window parameter w.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.
        R : ndarray of shape (n_samples,)
            The response values.
        w : int
            The window parameter determining the lag between X and R.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.X_ = X # Store input data
        self.R_ = R # Store response values
        self.w =w # Store the window parameter
        
        # Adjust input and response data according to the window parameter
        self.R_w = self.R_[w:] # Truncated response values
        self.X_w = self.X_[:-w] # Truncated input data
        return self
    
    def nw_est(self, x, h_x):
        """
        Estimate the function value at a given point using Nadaraya-Watson kernel regression.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            The point at which to estimate the function value.
        h_x : float or ndarray of shape (n_features,)
            The bandwidth parameter(s) for the kernel.

        Returns
        -------
        nw : float
            The estimated value at point x.
        """
        # Compute the normalized pairwise distances between input x and the original training data X_
        u_x = (x - self.X_) / h_x # Scale the differences by bandwidth
        # Compute the normalized pairwise distances between input x and the truncated data X_w
        u_x_w = (x - self.X_w) / h_x # Scale the differences with truncated data

        # Apply the Gaussian kernel to the pairwise differences for original data
        K_x = np.exp(-0.5 * np.sum(u_x**2, axis=1)) / ((2 * np.pi)**(self.X_.shape[1] / 2) * h_x**self.X_.shape[1])
        
        # Apply the Gaussian kernel to the pairwise differences for truncated data
        K_x_w = np.exp(-0.5 * np.sum(u_x_w**2, axis=1)) / ((2 * np.pi)**(self.X_w.shape[1] / 2) * h_x**self.X_w.shape[1])
        
        # Add regularization to avoid numerical issues
        K_x += self.alpha
        K_x_w += self.alpha #* (len(K_x_w)/len(K_x))

        # Compute the numerator: the weighted sum of truncated response values
        nw_num = np.sum(self.R_w * K_x_w) # Element-wise multiplication of response values with kernel weights
        
        nw_denom = np.sum(K_x) + 1e-10 # Add a small value to avoid division by zero
        
        # Calculate the Nadaraya-Watson estimate
        nw = (nw_num/nw_denom)* (len(K_x)/len(K_x_w))
        
        
        return nw
        
    
    
    def __call__(self, data_matrix):
        """
        Apply Nadaraya-Watson kernel regression to each row in the data matrix.

        Parameters
        ----------
        data_matrix : ndarray of shape (n_samples, n_features)
            The data points at which to estimate the function values.
        h_x : float or ndarray of shape (n_features,)
            The bandwidth parameter(s) for the kernel.

        Returns
        -------
        nw_est_vec : ndarray of shape (n_samples,)
            The estimated values for each row in the data matrix.
        """
        # Apply the nw_est method to each row in data_matrix
        nw_est_vec = np.apply_along_axis(self.nw_est, 1, data_matrix, self.bandwidth)
        
        return nw_est_vec
    
    
    
class est_f_sa_pi(BaseEstimator):
    """
    Custom estimator using Nadaraya-Watson kernel regression.

    Parameters
    ----------
    bandwidth : float, default=1.0
        The bandwidth parameter for the Gaussian kernel.

    Attributes
    ----------
    X_ : ndarray of shape (n_samples, n_features)
        The input data used for fitting.
    A_ : ndarray of shape (n_samples,)
        The action labels corresponding to each sample in X_.
    """
    def __init__(self, bandwidth=1.0, alpha = 0.1):
        # Initialize the estimator with a given bandwidth parameter for the kernel
        self.bandwidth = bandwidth
        self.alpha = alpha

    def fit(self, X, A):
        """
        Fit the model using input data X, action labels A, and responses R.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.
        A : ndarray of shape (n_samples,)
            The action labels corresponding to the data.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Store the input data (X), action labels (A), and response values (R) as instance attributes
        self.X_ = X  # Input data
        self.A_ = A  # Action labels
        return self
    
    def indicator(self, vec, a):
        """
        Create an indicator vector where elements are True if they match the specified action 'a'.
        
        Parameters
        ----------
        vec : array-like, shape (n_samples,)
            The vector to compare against (action labels).
        a : scalar
            The action to compare (the specific action to filter).
        
        Returns
        -------
        ind : array-like, shape (n_samples,)
            Indicator vector where True corresponds to matching the action.
        """
        # Create a boolean array where True indicates that the action in vec matches action 'a'
        ind = vec == a
        return ind

    def f_sa_pi_est(self, x, a, h_x):
        """
        Estimate the function value at a given point using Nadaraya-Watson kernel regression.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            The point at which to estimate the function value.
        a : scalar
            The action label to use for conditioning the estimate.
        h_x : float or ndarray of shape (n_features,)
            The bandwidth parameter(s) for the Gaussian kernel.

        Returns
        -------
        nw : float
            The estimated value at the point x, conditioned on action 'a'.
        """
        # Calculate the scaled differences (u) between x and each point in the training data X_
        #u = (x - self.X_) / h_x
        # Apply the Gaussian kernel function to the scaled differences (element-wise exponentiation)
        #K = np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
        # Create an indicator vector for samples corresponding to action 'a'
        #ind_A = self.indicator(self.A_, a)
        
        # Compute the numerator of the Nadaraya-Watson estimate:
        # - Multiply responses (R_) by the kernel weights (K) and the indicator for action 'a'
        # - Take the mean of these weighted responses
        #nw_num = np.mean(self.R_ * np.prod(K / h_x, axis=1) * ind_A)
        
        # Compute the denominator of the Nadaraya-Watson estimate:
        # - Multiply the kernel weights by the indicator for action 'a'
        # - Take the mean of these values to normalize the estimate
        #nw_denom = np.mean(ind_A * np.prod(K / h_x, axis=1))  # Add small constant to avoid division by zero
        
        # Compute the normalized pairwise differences for the state and action
        u_x = (x - self.X_) / h_x # Difference between input state and fitted states
        
        # Apply the Gaussian kernel to the state component
        K_x = np.exp(-0.5 * np.sum(u_x**2, axis=1)) / ((2 * np.pi)**(self.X_.shape[1] / 2) * h_x**self.X_.shape[1])
        
        # Create an indicator vector for samples corresponding to action 'a'
        ind_A = self.indicator(self.A_, a)
        
        # Combine the state and action kernels
        K = K_x*ind_A
        #K+= self.alpha
        
        f_sa_pi = np.mean(K) # Sum of kernel weights (marginal density)
        
                      
            
        return f_sa_pi
    
    def __call__(self, data_matrix, a_vec):
        """
        Apply Nadaraya-Watson kernel regression across all rows in the data matrix with action vector.

        Parameters
        ----------
        data_matrix : ndarray of shape (n_samples, n_features)
            The data points at which to estimate the function values.
        a_vec : array-like, shape (n_samples,)
            The action vector for which the policy probability is estimated.
            
        Returns
        -------
        nw_est_vec : ndarray of shape (n_samples,)
            The estimated values for each row in the data matrix.
        """
        # Ensure that data_matrix and a_vec have the same length
        assert data_matrix.shape[0] == a_vec.shape[0], "data_matrix and a_vec must have the same length"
        
        # Apply the nw_est method row by row, passing the corresponding element of a_vec
        f_sa_pi_est_vec = np.array([self.f_sa_pi_est(data_matrix[i], a_vec[i], self.bandwidth) for i in range(len(data_matrix))])
        return f_sa_pi_est_vec

    
    
    
class est_r_sa(BaseEstimator):
    """
    Custom estimator using Nadaraya-Watson kernel regression.

    Parameters
    ----------
    bandwidth : float, default=1.0
        The bandwidth parameter for the Gaussian kernel, which controls the smoothness of the density estimation.

    alpha : float, default=0.1
        The regularization parameter to prevent overfitting and improve numerical stability.
        It adds a small positive value to the kernel weights, ensuring that the computed densities remain finite
        and stable during evaluation, especially in regions where the weight of data points may be very low.

    Attributes
    ----------
    X_ : ndarray of shape (n_samples, n_features)
        The input data used for fitting, representing the features of each sample.
        
    A_ : ndarray of shape (n_samples,)
        The action labels corresponding to each sample in X_, indicating the actions taken for each input sample.
        
    R_ : ndarray of shape (n_samples,)
        The response values associated with X_, representing the outcomes we want to estimate or predict.
    """
    def __init__(self, bandwidth=1.0, alpha = 0.1):
        # Initialize the estimator with the specified bandwidth and regularization parameter
        self.bandwidth = bandwidth # Set the bandwidth for the kernel, influencing the smoothness of estimates
        self.alpha = alpha # Set the regularization parameter to prevent overfitting and improve numerical stability.

    def fit(self, X, A, R):
        """
        Fit the model using input data X, action labels A, and responses R.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.
        A : ndarray of shape (n_samples,)
            The action labels corresponding to the data.
        R : ndarray of shape (n_samples,)
            The response values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Store the input data (X), action labels (A), and response values (R) as instance attributes
        self.X_ = X  # Input data
        self.A_ = A  # Action labels
        self.R_ = R  # Response values
        return self
    
    def indicator(self, vec, a):
        """
        Create an indicator vector where elements are True if they match the specified action 'a'.
        
        Parameters
        ----------
        vec : array-like, shape (n_samples,)
            The vector to compare against (action labels).
        a : scalar
            The action to compare (the specific action to filter).
        
        Returns
        -------
        ind : array-like, shape (n_samples,)
            Indicator vector where True corresponds to matching the action.
        """
        # Create a boolean array where True indicates that the action in vec matches action 'a'
        ind = vec == a
        return ind

    def nw_est(self, x, a, h_x):
        """
        Estimate the function value at a given point using Nadaraya-Watson kernel regression.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            The point at which to estimate the function value.
        a : scalar
            The action label to use for conditioning the estimate.
        h_x : float or ndarray of shape (n_features,)
            The bandwidth parameter(s) for the Gaussian kernel.

        Returns
        -------
        nw : float
            The estimated value at the point x, conditioned on action 'a'.
        """
        
        # Compute the normalized pairwise differences for the state and action
        u_x = (x - self.X_) / h_x # Difference between input state and fitted states
        
        # Apply the Gaussian kernel to the state component
        K_x = np.exp(-0.5 * np.sum(u_x**2, axis=1)) / ((2 * np.pi)**(self.X_.shape[1] / 2) * h_x**self.X_.shape[1])
        
        # Create an indicator vector for samples corresponding to action 'a'
        ind_A = self.indicator(self.A_, a) # This vector is 1 for samples where action matches 'a', 0 otherwise
        
        # Combine the state kernel weights with the indicator for the specified action
        K = K_x*ind_A # Element-wise multiplication to apply action condition
        #K += self.alpha # Adding self.alpha ensures stability and prevents overfitting
        
        # Calculate the weighted sum of response values (numerator)
        nw_num = np.sum(self.R_ * K)  # Response values weighted by kernel weights

        # Calculate the sum of kernel weights (denominator) 
        # A small constant (1e-10) is added to avoid numerical instability
        nw_denom = np.sum(K) + 1e-10
        
        # Calculate the final estimate by dividing the numerator by the denominator
        nw = nw_num/nw_denom
            
        return nw
    
    def __call__(self, data_matrix, a):
        """
        Apply Nadaraya-Watson kernel regression to each row in the data matrix.

        Parameters
        ----------
        data_matrix : ndarray of shape (n_samples, n_features)
            The data points at which to estimate the function values.
        a : scalar
            The action label to use for conditioning the estimate.

        Returns
        -------
        nw_est_vec : ndarray of shape (n_samples,)
            The estimated values for each row in the data matrix.
        """
        # Apply the nw_est method to each row in data_matrix, estimating the function value for action 'a'
        nw_est_vec = np.apply_along_axis(self.nw_est, 1, data_matrix, a, self.bandwidth)
        
        return nw_est_vec


    
class est_r_saData(BaseEstimator):
    """
    Custom estimator using Nadaraya-Watson kernel regression.

    Parameters
    ----------
    bandwidth : float, default=1.0
        The bandwidth parameter for the Gaussian kernel, which controls the smoothness of the density estimation.

    alpha : float, default=0.1
        The regularization parameter to prevent overfitting and improve numerical stability.
        It adds a small positive value to the kernel weights, ensuring that the computed densities remain finite
        and stable during evaluation, especially in regions where the weight of data points may be very low.

    Attributes
    ----------
    X_ : ndarray of shape (n_samples, n_features)
        The input data used for fitting, representing the features of each sample.
        
    A_ : ndarray of shape (n_samples,)
        The action labels corresponding to each sample in X_, indicating the actions taken for each input sample.
        
    R_ : ndarray of shape (n_samples,)
        The response values associated with X_, representing the outcomes we want to estimate or predict.
    """
    def __init__(self, bandwidth=1.0, alpha = 0.1):
        # Initialize the estimator with the specified bandwidth and regularization parameter
        self.bandwidth = bandwidth # Set the bandwidth for the kernel, influencing the smoothness of estimates
        self.alpha = alpha # Set the regularization parameter to prevent overfitting and improve numerical stability.

    def fit(self, X, A, R):
        """
        Fit the model using input data X, action labels A, and responses R.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.
        A : ndarray of shape (n_samples,)
            The action labels corresponding to the data.
        R : ndarray of shape (n_samples,)
            The response values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Store the input data (X), action labels (A), and response values (R) as instance attributes
        self.X_ = X  # Input data
        self.A_ = A  # Action labels
        self.R_ = R  # Response values
        return self
    
    def indicator(self, vec, a):
        """
        Create an indicator vector where elements are True if they match the specified action 'a'.
        
        Parameters
        ----------
        vec : array-like, shape (n_samples,)
            The vector to compare against (action labels).
        a : scalar
            The action to compare (the specific action to filter).
        
        Returns
        -------
        ind : array-like, shape (n_samples,)
            Indicator vector where True corresponds to matching the action.
        """
        # Create a boolean array where True indicates that the action in vec matches action 'a'
        ind = vec == a
        return ind

    def nw_est(self, x, a, h_x):
        """
        Estimate the function value at a given point using Nadaraya-Watson kernel regression.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            The point at which to estimate the function value.
        a : scalar
            The action label to use for conditioning the estimate.
        h_x : float or ndarray of shape (n_features,)
            The bandwidth parameter(s) for the Gaussian kernel.

        Returns
        -------
        nw : float
            The estimated value at the point x, conditioned on action 'a'.
        """
        # Compute the normalized pairwise differences for the state and action
        u_x = (x - self.X_) / h_x # Difference between input state and fitted states
        
        # Apply the Gaussian kernel to the state component
        K_x = np.exp(-0.5 * np.sum(u_x**2, axis=1)) / ((2 * np.pi)**(self.X_.shape[1] / 2) * h_x**self.X_.shape[1])
        
        # Create an indicator vector for samples corresponding to action 'a'
        ind_A = self.indicator(self.A_, a) # This vector is 1 for samples where action matches 'a', 0 otherwise
        
        # Combine the state kernel weights with the indicator for the specified action
        K = K_x*ind_A # Element-wise multiplication to apply action condition
        #K += self.alpha # Adding self.alpha ensures stability and prevents overfitting
        
        # Calculate the weighted sum of response values (numerator)
        nw_num = np.sum(self.R_ * K)  # Response values weighted by kernel weights

        # Calculate the sum of kernel weights (denominator) 
        # A small constant (1e-10) is added to avoid numerical instability
        nw_denom = np.sum(K) + 1e-10
        
        # Calculate the final estimate by dividing the numerator by the denominator
        nw = nw_num/nw_denom
        
  
            
        return nw
    
    def __call__(self, data_matrix, a_vec):
        """
        Apply Nadaraya-Watson kernel regression across all rows in the data matrix with action vector.

        Parameters
        ----------
        data_matrix : ndarray of shape (n_samples, n_features)
            The data points at which to estimate the function values.
        a_vec : array-like, shape (n_samples,)
            The action vector for which the policy probability is estimated.
            
        Returns
        -------
        nw_est_vec : ndarray of shape (n_samples,)
            The estimated values for each row in the data matrix.
        """
        # Ensure that data_matrix and a_vec have the same length
        assert data_matrix.shape[0] == a_vec.shape[0], "data_matrix and a_vec must have the same length"
        
        # Apply the nw_est method row by row, passing the corresponding element of a_vec
        nw_est_vec = np.array([self.nw_est(data_matrix[i], a_vec[i], self.bandwidth) for i in range(len(data_matrix))])
        return nw_est_vec
    
    
class est_r_pi_sa_w(BaseEstimator):
    """
    Custom estimator using Nadaraya-Watson kernel regression with a window parameter.
    
    Parameters
    ----------
    bandwidth : float, default=1.0
        The bandwidth parameter for the Gaussian kernel, which controls the smoothness
        of the kernel density estimation by adjusting how much influence nearby points have.

    alpha : float, default=0.1
        The regularization parameter added to the kernel weights to ensure numerical stability,
        especially when there are very small or zero kernel values.

    Attributes
    ----------
    X_ : ndarray of shape (n_samples, n_features)
        The input data used for fitting (e.g., state variables or features). This data
        represents the independent variables in the model.
    
    A_ : ndarray of shape (n_samples,)
        The action values associated with each sample in X_. These could be either
        discrete (e.g., action labels) or continuous, depending on the problem setup.
    
    R_ : ndarray of shape (n_samples,)
        The response values (e.g., rewards or outcomes) associated with each sample
        in X_ and A_. These are the dependent variables the model aims to estimate.
    
    w : int
        The window parameter that defines the lag between the response variable and
        the input data. This lag is useful for time-dependent models where the response
        is expected to be influenced by a previous state.
    
    R_w : ndarray of shape (n_samples-w,)
        The truncated response values aligned with the window parameter. This array
        contains only the response values that have corresponding lagged input data.
    
    X_w : ndarray of shape (n_samples-w, n_features)
        The truncated input data aligned with the window parameter. This ensures that
        each row in X_w corresponds to a response value in R_w.
    
    A_w : ndarray of shape (n_samples-w,)
        The truncated action data aligned with the window parameter. Each action in
        A_w aligns with the corresponding state in X_w and response in R_w.
    """

    def __init__(self, bandwidth=1.0, alpha = 0.1):
        # Initialize the estimator with the specified bandwidth and regularization parameter
        self.bandwidth = bandwidth # Set the bandwidth for the kernel, influencing the smoothness of estimates
        self.alpha = alpha # Set the regularization parameter to prevent overfitting and improve numerical stability.

    def fit(self, X, A, R, w):
        """
        Fit the model using input data X, action data A, responses R, and a window parameter w.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data (e.g., states or features).
        A : ndarray of shape (n_samples,)
            The action data associated with each input (e.g., discrete or continuous actions).
        R : ndarray of shape (n_samples,)
            The response values (e.g., rewards or outcomes).
        w : int
            The window parameter determining the lag between X and R.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Store input, action, and response data
        self.X_ = X  # Store the input data (states/features)
        self.A_ = A  # Store the action data corresponding to each state
        self.R_ = R  # Store the response values (rewards)
        self.w = w   # Store window parameter (lag between X and R)

        # Adjust input, action, and response data based on the window parameter
        self.R_w = self.R_[w:]      # Truncate response values by skipping the first 'w' values
        self.X_w = self.X_[:-w]     # Truncate input data to align with the window-adjusted responses
        self.A_w = self.A_[:-w]     # Truncate action data to align with the window-adjusted inputs
        return self  # Return the fitted estimator

    def indicator(self, vec, a):
        """
        Create an indicator vector where elements are True if they match action 'a'.
        
        Parameters
        ----------
        vec : array-like, shape (n_samples,)
            The vector to compare against.
        a : scalar
            The action to compare.
        
        Returns
        -------
        ind : array-like, shape (n_samples,)
            Indicator vector that is True where 'vec' equals 'a'.
        """
        # Create a boolean vector where elements equal to action 'a' are marked as True
        ind = vec == a
        return ind

    def nw_est(self, x, a, h_x):
        """
        Estimate the function value at a given point using Nadaraya-Watson kernel regression.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            The point at which to estimate the function value.
        a : scalar
            The action for which to estimate the value.
        h_x : float or ndarray of shape (n_features,)
            The bandwidth parameter(s) for the kernel.

        Returns
        -------
        nw : float
            The estimated value at point x for action 'a'.
        """
        
        # Compute the normalized pairwise distances between input x and the original training data X_ and A_
        u_x = (x - self.X_) / h_x # Scale the differences by bandwidth for state
        # Compute the normalized pairwise distances between input x and the truncated data X_w and A_w
        u_x_w = (x - self.X_w) / h_x # Scale the differences by bandwidth for truncated state

        # Apply the Gaussian kernel to the pairwise differences for the full data
        K_x = np.exp(-0.5 * np.sum(u_x**2, axis=1)) / ((2 * np.pi)**(self.X_.shape[1] / 2) * h_x**self.X_.shape[1])
        
        # Create an indicator for the current action 'a' in the full action data (A_)
        ind_A = self.indicator(self.A_, a)
        
        # Apply the Gaussian kernel to the pairwise differences for the truncated data
        K_x_w = np.exp(-0.5 * np.sum(u_x_w**2, axis=1)) / ((2 * np.pi)**(self.X_w.shape[1] / 2) * h_x**self.X_w.shape[1])
        
        # Create an indicator for the current action 'a' in the truncated action data (A_w)
        ind_A_w = self.indicator(self.A_w, a)
        
        # Combine the kernel weights with action indicator for full data and add alpha for numerical stability
        K = K_x*ind_A
        #K+= self.alpha # Regularization term for stability in case of small or zero weights
        
        # Combine the kernel weights with action indicator for truncated data and add alpha for stability
        K_w = K_x_w*ind_A_w
        #K_w+= self.alpha # Regularization term for stability in case of small or zero weights
        
        # Compute the numerator: the weighted sum of truncated response values
        nw_num = np.sum(self.R_w * K_w) # Element-wise multiplication of response values with kernel weights
        
        # Compute the denominator as the sum of kernel weights, with a small constant to avoid division by zero
        nw_denom = np.sum(K) + 1e-10
        
        # Compute the final estimate as the ratio of the weighted response sum to the sum of kernel weights
        nw = (nw_num/nw_denom)* (len(K)/len(K_w))
        
                      
        
        return nw_num

    def __call__(self, data_matrix, a):
        """
        Apply Nadaraya-Watson kernel regression to each row in the data matrix.

        Parameters
        ----------
        data_matrix : ndarray of shape (n_samples, n_features)
            The data points at which to estimate the function values.
        a : scalar
            The action for which to estimate the values.

        Returns
        -------
        nw_est_vec : ndarray of shape (n_samples,)
            The estimated values for each row in the data matrix, for action 'a'.
        """
        # Apply the Nadaraya-Watson estimation method to each row of the data matrix
        nw_est_vec = np.apply_along_axis(self.nw_est, 1, data_matrix, a, self.bandwidth)
        
        return nw_est_vec  # Return the estimated values for each data point and action

    
    
class est_r_pi_sa_wData(BaseEstimator):
    """
    Custom estimator using Nadaraya-Watson kernel regression with a window parameter.
    
    Parameters
    ----------
    bandwidth : float, default=1.0
        The bandwidth parameter for the Gaussian kernel, which controls the smoothness
        of the kernel density estimation by adjusting how much influence nearby points have.

    alpha : float, default=0.1
        The regularization parameter added to the kernel weights to ensure numerical stability,
        especially when there are very small or zero kernel values.

    Attributes
    ----------
    X_ : ndarray of shape (n_samples, n_features)
        The input data used for fitting (e.g., state variables or features). This data
        represents the independent variables in the model.
    
    A_ : ndarray of shape (n_samples,)
        The action values associated with each sample in X_. These could be either
        discrete (e.g., action labels) or continuous, depending on the problem setup.
    
    R_ : ndarray of shape (n_samples,)
        The response values (e.g., rewards or outcomes) associated with each sample
        in X_ and A_. These are the dependent variables the model aims to estimate.
    
    w : int
        The window parameter that defines the lag between the response variable and
        the input data. This lag is useful for time-dependent models where the response
        is expected to be influenced by a previous state.
    
    R_w : ndarray of shape (n_samples-w,)
        The truncated response values aligned with the window parameter. This array
        contains only the response values that have corresponding lagged input data.
    
    X_w : ndarray of shape (n_samples-w, n_features)
        The truncated input data aligned with the window parameter. This ensures that
        each row in X_w corresponds to a response value in R_w.
    
    A_w : ndarray of shape (n_samples-w,)
        The truncated action data aligned with the window parameter. Each action in
        A_w aligns with the corresponding state in X_w and response in R_w.
    """

    def __init__(self, bandwidth=1.0, alpha = 0.1):
        # Initialize the estimator with the specified bandwidth and regularization parameter
        self.bandwidth = bandwidth # Set the bandwidth for the kernel, influencing the smoothness of estimates
        self.alpha = alpha # Set the regularization parameter to prevent overfitting and improve numerical stability.

    def fit(self, X, A, R, w):
        """
        Fit the model using input data X, action data A, responses R, and a window parameter w.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data (e.g., states or features).
        A : ndarray of shape (n_samples,)
            The action data associated with each input (e.g., discrete or continuous actions).
        R : ndarray of shape (n_samples,)
            The response values (e.g., rewards or outcomes).
        w : int
            The window parameter determining the lag between X and R.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Store input, action, and response data
        self.X_ = X  # Store the input data (states/features)
        self.A_ = A  # Store the action data corresponding to each state
        self.R_ = R  # Store the response values (rewards)
        self.w = w   # Store window parameter (lag between X and R)

        # Adjust input, action, and response data based on the window parameter
        self.R_w = self.R_[w:]      # Truncate response values by skipping the first 'w' values
        self.X_w = self.X_[:-w]     # Truncate input data to align with the window-adjusted responses
        self.A_w = self.A_[:-w]     # Truncate action data to align with the window-adjusted inputs
        return self  # Return the fitted estimator

    def indicator(self, vec, a):
        """
        Create an indicator vector where elements are True if they match action 'a'.
        
        Parameters
        ----------
        vec : array-like, shape (n_samples,)
            The vector to compare against.
        a : scalar
            The action to compare.
        
        Returns
        -------
        ind : array-like, shape (n_samples,)
            Indicator vector that is True where 'vec' equals 'a'.
        """
        # Create a boolean vector where elements equal to action 'a' are marked as True
        ind = vec == a
        return ind

    def nw_est(self, x, a, h_x):
        """
        Estimate the function value at a given point using Nadaraya-Watson kernel regression.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            The point at which to estimate the function value.
        a : scalar
            The action for which to estimate the value.
        h_x : float or ndarray of shape (n_features,)
            The bandwidth parameter(s) for the kernel.

        Returns
        -------
        nw : float
            The estimated value at point x for action 'a'.
        """
        
        
        # Compute the normalized pairwise distances between input x and the original training data X_ and A_
        u_x = (x - self.X_) / h_x  # Scale the differences by bandwidth for state
        # Compute the normalized pairwise distances between input x and the truncated data X_w and A_w
        u_x_w = (x - self.X_w) / h_x # Scale the differences by bandwidth for truncated state

        # Apply the Gaussian kernel to the pairwise differences for the full data
        K_x = np.exp(-0.5 * np.sum(u_x**2, axis=1)) / ((2 * np.pi)**(self.X_.shape[1] / 2) * h_x**self.X_.shape[1])
        
        # Create an indicator for the current action 'a' in the full action data (A_)
        ind_A = self.indicator(self.A_, a)
        
        # Apply the Gaussian kernel to the pairwise differences for the truncated data
        K_x_w = np.exp(-0.5 * np.sum(u_x_w**2, axis=1)) / ((2 * np.pi)**(self.X_w.shape[1] / 2) * h_x**self.X_w.shape[1])
        
        # Create an indicator for the current action 'a' in the truncated action data (A_w)
        ind_A_w = self.indicator(self.A_w, a)
        
        # Combine the kernel weights with action indicator for full data and add alpha for numerical stability
        K = K_x*ind_A
        #K+= self.alpha # Regularization term for stability in case of small or zero weights
        
        # Combine the kernel weights with action indicator for truncated data and add alpha for stability
        K_w = K_x_w*ind_A_w
        #K_w+= self.alpha # Regularization term for stability in case of small or zero weights
        
        # Compute the numerator: the weighted sum of truncated response values
        nw_num = np.sum(self.R_w * K_w)   # Element-wise multiplication of response values with kernel weights
        
        # Compute the denominator as the sum of kernel weights, with a small constant to avoid division by zero
        nw_denom = np.sum(K) + 1e-10
        
        # Compute the final estimate as the ratio of the weighted response sum to the sum of kernel weights
        nw = (nw_num/nw_denom) * (len(K)/len(K_w))
        
        
        
        return nw

    def __call__(self, data_matrix, a_vec):
        """
        Apply Nadaraya-Watson kernel regression across all rows in the data matrix with action vector.

        Parameters
        ----------
        data_matrix : ndarray of shape (n_samples, n_features)
            The data points at which to estimate the function values.
        a_vec : array-like, shape (n_samples,)
            The action vector for which the policy probability is estimated.

        Returns
        -------
        nw_est_vec : ndarray of shape (n_samples,)
            The estimated values for each row in the data matrix, for action 'a'.
        """
        # Ensure that data_matrix and a_vec have the same length
        assert data_matrix.shape[0] == a_vec.shape[0], "data_matrix and a_vec must have the same length"
        
        # Apply the Nadaraya-Watson estimation method to each row of the data matrix
        nw_est_vec = np.array([self.nw_est(data_matrix[i], a_vec[i], self.bandwidth) for i in range(len(data_matrix))])
        
        return nw_est_vec  # Return the estimated values for each data point and action
