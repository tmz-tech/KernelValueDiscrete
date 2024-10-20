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
    Custom estimator using Nadaraya-Watson kernel regression.

    Parameters
    ----------
    bandwidth : float, default=1.0
        The bandwidth parameter for the Gaussian kernel.

    Attributes
    ----------
    X_ : ndarray of shape (n_samples, n_features)
        The input data used for fitting.
    R_ : ndarray of shape (n_samples,)
        The response values associated with X_.
    """
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth

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
            The bandwidth parameter(s) for the kernel.

        Returns
        -------
        nw : float
            The estimated value at point x.
        """
        # Calculate the u values as the scaled distance from x to each point in X_
        u = (x - self.X_) / h_x
        # Gaussian kernel values
        K= np.exp(-0.5*u**2)/np.sqrt(2 * np.pi)
        # Calculate the numerator (weighted sum of R_) and denominator (sum of weights)
        nw_num = np.mean(self.R_*np.prod(K/h_x, axis=1))
        nw_denom = np.mean(np.prod(K/h_x, axis=1))
        # Return the Nadaraya-Watson estimate
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
        The bandwidth parameter for the Gaussian kernel.

    Attributes
    ----------
    X_ : ndarray of shape (n_samples, n_features)
        The input data used for fitting.
    R_ : ndarray of shape (n_samples,)
        The response values associated with X_.
    w : int
        The window parameter that determines the lag between the response and the input data.
    R_w : ndarray of shape (n_samples-w,)
        The truncated response values aligned with the window parameter.
    X_w : ndarray of shape (n_samples-w, n_features)
        The truncated input data aligned with the window parameter.
    """
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth

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
        # Calculate the kernel values for the numerator using truncated X_w and R_w
        u_num = (x - self.X_w) / h_x
        K_num= np.exp(-0.5*u_num**2)/np.sqrt(2 * np.pi)
        nw_num = np.mean(self.R_w*np.prod(K_num/h_x, axis=1))
        # Calculate the kernel values for the denominator using the full dataset X_
        u_denom = (x - self.X_) / h_x
        K_denom= np.exp(-0.5*u_denom**2)/np.sqrt(2 * np.pi)
        nw_denom = np.mean(np.prod(K_denom/h_x, axis=1))
        # Return the Nadaraya-Watson estimate as the ratio of the numerator to the denominator
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
    
    
    
    

    
    
    
class est_r_sa(BaseEstimator):
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
    R_ : ndarray of shape (n_samples,)
        The response values associated with X_.
    """
    def __init__(self, bandwidth=1.0):
        # Initialize the estimator with a given bandwidth parameter for the kernel
        self.bandwidth = bandwidth

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
        # Calculate the scaled differences (u) between x and each point in the training data X_
        u = (x - self.X_) / h_x
        # Apply the Gaussian kernel function to the scaled differences (element-wise exponentiation)
        K = np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
        # Create an indicator vector for samples corresponding to action 'a'
        ind_A = self.indicator(self.A_, a)
        
        # Compute the numerator of the Nadaraya-Watson estimate:
        # - Multiply responses (R_) by the kernel weights (K) and the indicator for action 'a'
        # - Take the mean of these weighted responses
        nw_num = np.mean(self.R_ * np.prod(K / h_x, axis=1) * ind_A)
        
        # Compute the denominator of the Nadaraya-Watson estimate:
        # - Multiply the kernel weights by the indicator for action 'a'
        # - Take the mean of these values to normalize the estimate
        nw_denom = np.mean(ind_A * np.prod(K / h_x, axis=1)) + 1e-10  # Add small constant to avoid division by zero
        
        # Calculate the Nadaraya-Watson estimate as the ratio of the numerator to the denominator
        nw = nw_num / nw_denom
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
        The bandwidth parameter for the Gaussian kernel.

    Attributes
    ----------
    X_ : ndarray of shape (n_samples, n_features)
        The input data used for fitting.
    A_ : ndarray of shape (n_samples,)
        The action labels corresponding to each sample in X_.
    R_ : ndarray of shape (n_samples,)
        The response values associated with X_.
    """
    def __init__(self, bandwidth=1.0):
        # Initialize the estimator with a given bandwidth parameter for the kernel
        self.bandwidth = bandwidth

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
        # Calculate the scaled differences (u) between x and each point in the training data X_
        u = (x - self.X_) / h_x
        # Apply the Gaussian kernel function to the scaled differences (element-wise exponentiation)
        K = np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
        # Create an indicator vector for samples corresponding to action 'a'
        ind_A = self.indicator(self.A_, a)
        
        # Compute the numerator of the Nadaraya-Watson estimate:
        # - Multiply responses (R_) by the kernel weights (K) and the indicator for action 'a'
        # - Take the mean of these weighted responses
        nw_num = np.mean(self.R_ * np.prod(K / h_x, axis=1) * ind_A)
        
        # Compute the denominator of the Nadaraya-Watson estimate:
        # - Multiply the kernel weights by the indicator for action 'a'
        # - Take the mean of these values to normalize the estimate
        nw_denom = np.mean(ind_A * np.prod(K / h_x, axis=1)) + 1e-10  # Add small constant to avoid division by zero
        
        # Calculate the Nadaraya-Watson estimate as the ratio of the numerator to the denominator
        nw = nw_num / nw_denom
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
        The bandwidth parameter for the Gaussian kernel.

    Attributes
    ----------
    X_ : ndarray of shape (n_samples, n_features)
        The input data used for fitting (e.g., state variables or features).
    A_ : ndarray of shape (n_samples,)
        The action values associated with X_ (discrete or continuous actions).
    R_ : ndarray of shape (n_samples,)
        The response values (e.g., rewards) associated with X_ and A_.
    w : int
        The window parameter that determines the lag between the response and the input data.
    R_w : ndarray of shape (n_samples-w,)
        The truncated response values aligned with the window parameter.
    X_w : ndarray of shape (n_samples-w, n_features)
        The truncated input data aligned with the window parameter.
    A_w : ndarray of shape (n_samples-w,)
        The truncated action data aligned with the window parameter.
    """

    def __init__(self, bandwidth=1.0):
        # Initialize the estimator with the specified bandwidth
        self.bandwidth = bandwidth

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
        # Compute the scaled distances (u_num) for the numerator using truncated input data (X_w)
        u_num = (x - self.X_w) / h_x
        
        # Compute the Gaussian kernel values for the numerator
        K_num = np.exp(-0.5 * u_num ** 2) / np.sqrt(2 * np.pi)
        
        # Create an indicator for the current action 'a' in the truncated action data (A_w)
        ind_A_w = self.indicator(self.A_w, a)
        
        # Calculate the numerator of the Nadaraya-Watson estimator: weighted sum of responses (R_w)
        nw_num = np.mean(self.R_w * np.prod(K_num / h_x, axis=1) * ind_A_w)
        
        # Compute the scaled distances (u_denom) for the denominator using the full input data (X_)
        u_denom = (x - self.X_) / h_x
        
        # Compute the Gaussian kernel values for the denominator
        K_denom = np.exp(-0.5 * u_denom ** 2) / np.sqrt(2 * np.pi)
        
        # Create an indicator for the current action 'a' in the full action data (A_)
        ind_A = self.indicator(self.A_, a)
        
        # Calculate the denominator of the Nadaraya-Watson estimator: sum of kernel values
        nw_denom = np.mean(np.prod(K_denom / h_x, axis=1) * ind_A) + 1e-10  # Small value to avoid division by zero
        
        # Return the Nadaraya-Watson estimate as the ratio of the numerator to the denominator
        nw = nw_num / nw_denom
        
        return nw

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
        The bandwidth parameter for the Gaussian kernel.

    Attributes
    ----------
    X_ : ndarray of shape (n_samples, n_features)
        The input data used for fitting (e.g., state variables or features).
    A_ : ndarray of shape (n_samples,)
        The action values associated with X_ (discrete or continuous actions).
    R_ : ndarray of shape (n_samples,)
        The response values (e.g., rewards) associated with X_ and A_.
    w : int
        The window parameter that determines the lag between the response and the input data.
    R_w : ndarray of shape (n_samples-w,)
        The truncated response values aligned with the window parameter.
    X_w : ndarray of shape (n_samples-w, n_features)
        The truncated input data aligned with the window parameter.
    A_w : ndarray of shape (n_samples-w,)
        The truncated action data aligned with the window parameter.
    """

    def __init__(self, bandwidth=1.0):
        # Initialize the estimator with the specified bandwidth
        self.bandwidth = bandwidth

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
        # Compute the scaled distances (u_num) for the numerator using truncated input data (X_w)
        u_num = (x - self.X_w) / h_x
        
        # Compute the Gaussian kernel values for the numerator
        K_num = np.exp(-0.5 * u_num ** 2) / np.sqrt(2 * np.pi)
        
        # Create an indicator for the current action 'a' in the truncated action data (A_w)
        ind_A_w = self.indicator(self.A_w, a)
        
        # Calculate the numerator of the Nadaraya-Watson estimator: weighted sum of responses (R_w)
        nw_num = np.mean(self.R_w * np.prod(K_num / h_x, axis=1) * ind_A_w)
        
        # Compute the scaled distances (u_denom) for the denominator using the full input data (X_)
        u_denom = (x - self.X_) / h_x
        
        # Compute the Gaussian kernel values for the denominator
        K_denom = np.exp(-0.5 * u_denom ** 2) / np.sqrt(2 * np.pi)
        
        # Create an indicator for the current action 'a' in the full action data (A_)
        ind_A = self.indicator(self.A_, a)
        
        # Calculate the denominator of the Nadaraya-Watson estimator: sum of kernel values
        nw_denom = np.mean(np.prod(K_denom / h_x, axis=1) * ind_A) + 1e-10  # Small value to avoid division by zero
        
        # Return the Nadaraya-Watson estimate as the ratio of the numerator to the denominator
        nw = nw_num / nw_denom
        
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
