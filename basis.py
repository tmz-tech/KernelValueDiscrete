#!/usr/bin/env python
# coding: utf-8


#import pandas as pd
#import numpy as np
#from sklearn.base import BaseEstimator
from itertools import product

from . import pd, np, BaseEstimator


class ChebyshevBasis(BaseEstimator):
    def __init__(self, max_order:int =3):
        """
        Initialize the ChebyshevBasis estimator with a maximum polynomial order.
        
        Parameters:
        max_order (int): The maximum order of Chebyshev polynomials to include.
        """
        self.max_order = max_order
        
    def __call__(self, data:np.ndarray):
        """
        Compute Chebyshev polynomial basis functions for the given data.
        
        Parameters:
        data (np.ndarray): Input data with shape (n_samples, n_features).
        
        Returns:
        tuple: A DataFrame of basis functions and a dictionary mapping column names to polynomial orders for each sample.
        """
        d = data.shape[1]
        self.orders = self.generate_orders(self.max_order, d)
        weights = self.compute_weights(data)
        basis_df,basis_dict = self.compute_basis_functions(data, self.orders)
        
        # Compute the weighted basis functions
        #sqrt_weights = np.sqrt(weights)  # Square root of weights for scaling
        #basis_sqrt_weights = basis_df.values * sqrt_weights[:, np.newaxis]  # Scale basis functions
        #basis_sqrt_weights_df = pd.DataFrame(basis_sqrt_weights, columns=basis_df.columns)  # Create DataFrame
        
        
        return basis_df, basis_dict
        
    def generate_orders(self, n:int, d:int):
        """
        Generate a list of order combinations for each feature.
        
        Parameters:
        max_order (int): Maximum order of the polynomial.
        n_features (int): Number of features in the data.
        
        Returns:
        list: List of order combinations.
        """
        return [list(range(n + 1)) for _ in range(d)]
    
    def chebyshev_polynomials(self, x, order):
        """
        Compute Chebyshev polynomials of a given order.
        
        Parameters:
        x (np.ndarray): Input data for polynomial computation.
        order (int): The order of the Chebyshev polynomial.
        
        Returns:
        np.ndarray: Array of Chebyshev polynomials.
        """
        T = [np.ones_like(x), x]  # T0, T1
        for n in range(2, order + 1):
            T.append(2 * x * T[n-1] - T[n-2])
        return np.array(T)
    
    def compute_basis_functions(self, x, orders):
        """
        Compute the basis functions for all combinations of polynomial orders.
        
        Parameters:
        x (np.ndarray): Input data with shape (n_samples, n_features).
        orders (list): List of order combinations for each feature.
        
        Returns:
        tuple: A DataFrame of basis functions and a dictionary mapping column names to polynomial orders.
        """
        basis_dict = {}
        basis_values = []
        col_index = 0
        for combo in product(*orders):
            product_term = np.ones(x.shape[0])
            for l, order in enumerate(combo):
                T = self.chebyshev_polynomials(x[:, l], order)
                product_term *= T[order]
            column_name = f'Basis_{col_index}'
            basis_dict[column_name] = combo
            basis_values.append(product_term)
            col_index += 1
        basis_df = pd.DataFrame(np.column_stack(basis_values))
        basis_df.columns = [f'Basis_{i}' for i in range(basis_df.shape[1])]
        return basis_df, basis_dict
    
    
    def compute_weights(self, data: np.ndarray, epsilon: float = 1e-10):
        """
        Compute the weights corresponding to the Chebyshev basis functions.

        Parameters:
        data (np.ndarray): Input data with shape (n_samples, n_features).
        epsilon (float): Small constant to prevent division by zero.

        Returns:
        np.ndarray: Array of weights for each sample.
        """
        # Calculate weights for each dimension and combine them all at once
        weights = np.prod(1 / np.sqrt(np.maximum(1 - data ** 2, epsilon)), axis=1)
        return weights
    
    
    
class HermiteBasis(BaseEstimator):
    def __init__(self, max_order:int = 3):
        """
        Initialize the HermiteBasis estimator with a maximum polynomial order.
        
        Parameters:
        max_order (int): The maximum order of Hermite polynomials to include.
        """
        self.max_order = max_order
        
    def __call__(self, data: np.ndarray):
        """
        Compute Hermite polynomial basis functions for the given data.
        
        Parameters:
        data (np.ndarray): Input data with shape (n_samples, n_features).
        
        Returns:
        tuple: A DataFrame of basis functions and a dictionary mapping column names to polynomial orders.
        """
        d = data.shape[1]
        self.orders = self.generate_orders(self.max_order, d)
        weights = self.compute_weights(data)  # Compute weights
        basis_df, basis_dict = self.compute_basis_functions(data, self.orders)
        
        # Compute the weighted basis functions
        #sqrt_weights = np.sqrt(weights)  # Square root of weights for scaling
        #basis_sqrt_weights = basis_df.values * sqrt_weights[:, np.newaxis]  # Scale basis functions
        #basis_sqrt_weights_df = pd.DataFrame(basis_sqrt_weights, columns=basis_df.columns)  # Create DataFrame
        
        return basis_df, basis_dict
        
    def generate_orders(self, n: int, d: int):
        """
        Generate a list of order combinations for each feature.
        
        Parameters:
        n (int): Maximum order of the polynomial.
        d (int): Number of features in the data.
        
        Returns:
        list: List of order combinations.
        """
        return [list(range(n + 1)) for _ in range(d)]
    
    def hermite_polynomials(self, x, order):
        """
        Compute Hermite polynomials of a given order.
        
        Parameters:
        x (np.ndarray): Input data for polynomial computation.
        order (int): The order of the Hermite polynomial.
        
        Returns:
        np.ndarray: Array of Hermite polynomials.
        """
        H = [np.ones_like(x), 2 * x]  # H0, H1
        for n in range(2, order + 1):
            H.append(2 * x * H[n - 1] - 2 * (n - 1) * H[n - 2])
        return np.array(H)
    
    def compute_basis_functions(self, x, orders):
        """
        Compute the basis functions for all combinations of polynomial orders.
        
        Parameters:
        x (np.ndarray): Input data with shape (n_samples, n_features).
        orders (list): List of order combinations for each feature.
        
        Returns:
        tuple: A DataFrame of basis functions and a dictionary mapping column names to polynomial orders.
        """
        basis_dict = {}
        basis_values = []
        col_index = 0
        for combo in product(*orders):
            product_term = np.ones(x.shape[0])
            for l, order in enumerate(combo):
                H = self.hermite_polynomials(x[:, l], order)
                product_term *= H[order]
            column_name = f'Basis_{col_index}'
            basis_dict[column_name] = combo
            basis_values.append(product_term)
            col_index += 1
        basis_df = pd.DataFrame(np.column_stack(basis_values))
        basis_df.columns = [f'Basis_{i}' for i in range(basis_df.shape[1])]
        return basis_df, basis_dict
    
    def compute_weights(self, data: np.ndarray, epsilon: float = 1e-10):
        """
        Compute the weights corresponding to the Hermite basis functions.

        Parameters:
        data (np.ndarray): Input data with shape (n_samples, n_features).
        epsilon (float): Small constant to prevent negative values in exponentiation.

        Returns:
        np.ndarray: Array of weights for each sample.
        """
        
        # Calculate weights using the Gaussian weight function for each dimension
        # Sum of squares for each sample
        sum_of_squares = np.sum(data**2, axis=1)
        # Calculate weights
        weights = np.exp(-sum_of_squares)
        # Clip weights to prevent numerical issues
        weights = np.clip(weights, epsilon, None)  # Ensure weights are not too small
      
        return weights



# In[ ]:


class BasisNextExpect(BaseEstimator):
    def __init__(self, bandwidth=1.0):
        """
        Initialize the class with a specified bandwidth for the kernel density estimation.

        Parameters
        ----------
        bandwidth : float, default=1.0
            The bandwidth parameter that controls the width of the Gaussian kernel.
        """
        self.bandwidth = bandwidth

    def fit(self, X, X_next):
        """
        Fit the model by storing the current and next state data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The current state data.
        X_next : array-like, shape (n_samples, n_features)
            The next state data.
        
        Returns
        -------
        self : object
            Returns self with stored data.
        """
        self.X_ = X # Store the current state data in the instance
        self.X_next = X_next # Store the next state data in the instance
        return self
    
    def __call__(self, data_matrix, basis):
        """
        Apply the basis_next_expect function to each row of the data matrix.

        Parameters
        ----------
        data_matrix : ndarray of shape (n_samples, n_features)
            The data points at which to estimate the function values.
        h_x : float or ndarray of shape (n_features,)
            The bandwidth parameter(s) for the kernel.
        
        Returns
        -------
        BNE_vec : ndarray of shape (n_samples,)
            The estimated values for each row in the data matrix.
        """
        # Apply the basis_next_expect method to each row in data_matrix
        BNE_vec = np.apply_along_axis(self.basis_next_expect, 1, data_matrix, self.bandwidth, basis)
        return BNE_vec
    
    def basis_next_expect(self, x, h_x, basis):
        """
        Calculate the conditional density using KDE and basis functions.

        Parameters
        ----------
        x : array-like, shape (p,)
            The conditioning variable (current state).
        h_x : float or array-like, shape (p,)
            The bandwidth parameter(s) for the kernel.
        basis : object
            The basis object used for computing basis functions.
        
        Returns
        -------
        BNE : ndarray
            The conditional expectation value(s) for the given state.
        """
        # Compute the normalized difference between x and the current state data X_
        u = (x - self.X_) / h_x
        # Calculate Gaussian kernel values based on the normalized differences
        K= np.exp(-0.5*u**2)/np.sqrt(2 * np.pi)
        
        # Create an array filled with the first element of x and append it to the next state data
        #add_x = np.full(self.X_next.shape, x[:-1])
        add_x = np.tile(x[:-1], (self.X_next.shape[0], 1))
        state_next_ = np.hstack((self.X_next, add_x))
        # Compute the basis functions for the extended next state
        basis_next_df, _ = basis.compute_basis_functions(state_next_, basis.orders)
        
        
        # Compute the numerator: the weighted sum of the basis functions with kernel values
        BNE_num = np.mean(basis_next_df.values * np.prod(K/h_x, axis=1)[:, np.newaxis], axis=0)
        # Compute the denominator: the sum of the kernel values
        BNE_denom = np.mean(np.prod(K/h_x, axis=1))
        
        # Ensure that the denominator is not zero to avoid division by zero
        if BNE_denom == 0:
            raise ValueError("Denominator in conditional density calculation is zero.")
            
        # Compute the conditional expectation by dividing the numerator by the denominator
        BNE = BNE_num/BNE_denom
        return BNE
    
    

class BasisNextSAExpect(BaseEstimator):
    """
    Custom estimator for calculating the conditional expectation of next state 
    using Nadaraya-Watson kernel regression and basis functions.

    Parameters
    ----------
    bandwidth : float, default=1.0
        The bandwidth parameter that controls the width of the Gaussian kernel.
    """
    def __init__(self, bandwidth=1.0):
        """
        Initialize the class with a specified bandwidth for kernel density estimation.

        Parameters
        ----------
        bandwidth : float, default=1.0
            The bandwidth parameter for the Gaussian kernel.
        """
        self.bandwidth = bandwidth

    def fit(self, X, A, X_next):
        """
        Fit the model by storing the current state, action, and next state data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The current state data.
        A : array-like, shape (n_samples,)
            The action labels corresponding to each sample in X.
        X_next : array-like, shape (n_samples, n_features)
            The next state data.

        Returns
        -------
        self : object
            Returns the fitted estimator with stored data.
        """
        self.X_ = X  # Store current state data in the instance
        self.A_ = A  # Store action labels in the instance
        self.X_next = X_next  # Store next state data in the instance
        return self

    def __call__(self, data_matrix, a, basis):
        """
        Apply the basis_next_expect function to each row of the data matrix.

        Parameters
        ----------
        data_matrix : ndarray of shape (n_samples, n_features)
            The data points at which to estimate the conditional expectation.
        a : scalar
            The action label to use for conditioning.
        basis : object
            The basis object for computing basis functions.

        Returns
        -------
        BNE_vec : ndarray of shape (n_samples,)
            The estimated values for each row in the data matrix.
        """
        # Apply the basis_next_expect function to each row in data_matrix
        BNE_vec = np.apply_along_axis(self.basis_next_expect, 1, data_matrix, a, self.bandwidth, basis)
        return BNE_vec
    
    def indicator(self, vec, a):
        """
        Create an indicator vector where elements are True if they match action 'a'.

        Parameters
        ----------
        vec : array-like, shape (n_samples,)
            The vector to compare against (action labels).
        a : scalar
            The action to compare.

        Returns
        -------
        ind : array-like, shape (n_samples,)
            Indicator vector that is True where 'vec' equals 'a'.
        """
        # Create a boolean vector where elements are True if action 'a' matches
        ind = vec == a
        return ind

    def basis_next_expect(self, x, a, h_x, basis):
        """
        Calculate the conditional expectation using Nadaraya-Watson kernel regression 
        and basis functions for the next state.

        Parameters
        ----------
        x : array-like, shape (p,)
            The current state (conditioning variable).
        a : scalar
            The action label to condition the expectation on.
        h_x : float or array-like, shape (p,)
            The bandwidth parameter(s) for the Gaussian kernel.
        basis : object
            The basis object used to compute basis functions.

        Returns
        -------
        BNE : ndarray
            The conditional expectation value(s) for the given state.
        """
        # Compute the normalized differences between the current state x and each point in X_
        u = (x - self.X_) / h_x
        # Calculate Gaussian kernel values based on the normalized differences
        K = np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
        
        # Create an array filled with the first element of x and append it to the next state data
        #add_x = np.full(self.X_next.shape, x[0])  # Broadcast x[0] to match the shape of X_next
        add_x = np.tile(x[:-1], (self.X_next.shape[0], 1))
        state_next_ = np.hstack((self.X_next, add_x))  # Concatenate next state data with add_x

        # Compute the basis functions for the extended next state
        basis_next_df, _ = basis.compute_basis_functions(state_next_, basis.orders)
        
        # Create an indicator vector for samples corresponding to action 'a'
        ind_A = self.indicator(self.A_, a)
        
        # Compute the numerator: the weighted sum of the basis functions using the kernel values
        BNE_num = np.mean(basis_next_df.values * np.prod(K / h_x, axis=1)[:, np.newaxis] * ind_A[:, np.newaxis], axis=0)
        # Compute the denominator: the sum of the kernel values for action 'a'
        BNE_denom = np.mean(ind_A * np.prod(K / h_x, axis=1)) + 1e-10  # Small constant to prevent division by zero
        
        # Ensure that the denominator is not zero to avoid division by zero errors
        if BNE_denom == 0:
            raise ValueError("Denominator in conditional density calculation is zero.")
        
        # Compute the conditional expectation by dividing the numerator by the denominator
        BNE = BNE_num / BNE_denom
        return BNE

    

class BasisNextSAExpectData(BaseEstimator):
    """
    Custom estimator for calculating the conditional expectation of next state 
    using Nadaraya-Watson kernel regression and basis functions.

    Parameters
    ----------
    bandwidth : float, default=1.0
        The bandwidth parameter that controls the width of the Gaussian kernel.
    """
    def __init__(self, bandwidth=1.0):
        """
        Initialize the class with a specified bandwidth for kernel density estimation.

        Parameters
        ----------
        bandwidth : float, default=1.0
            The bandwidth parameter for the Gaussian kernel.
        """
        self.bandwidth = bandwidth

    def fit(self, X, A, X_next):
        """
        Fit the model by storing the current state, action, and next state data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The current state data.
        A : array-like, shape (n_samples,)
            The action labels corresponding to each sample in X.
        X_next : array-like, shape (n_samples, n_features)
            The next state data.

        Returns
        -------
        self : object
            Returns the fitted estimator with stored data.
        """
        self.X_ = X  # Store current state data in the instance
        self.A_ = A  # Store action labels in the instance
        self.X_next = X_next  # Store next state data in the instance
        return self

    def __call__(self, data_matrix, a_vec, basis):
        """
        Apply the basis_next_expect function to each row of the data matrix.

        Parameters
        ----------
        data_matrix : ndarray of shape (n_samples, n_features)
            The data points at which to estimate the conditional expectation.
        a_vec : array-like, shape (n_samples,)
            The action vector for which the policy probability is estimated.
        basis : object
            The basis object for computing basis functions.

        Returns
        -------
        BNE_vec : ndarray of shape (n_samples,)
            The estimated values for each row in the data matrix, for action 'a'.
        """
        
        # Ensure that data_matrix and a_vec have the same length
        assert data_matrix.shape[0] == a_vec.shape[0], "data_matrix and a_vec must have the same length"
        
        # Apply the basis_next_expect function to each row of the data matrix
        BNE_vec = np.array([self.basis_next_expect(data_matrix[i], a_vec[i], self.bandwidth, basis) for i in range(len(data_matrix))])
       
        
        return BNE_vec
    
    def indicator(self, vec, a):
        """
        Create an indicator vector where elements are True if they match action 'a'.

        Parameters
        ----------
        vec : array-like, shape (n_samples,)
            The vector to compare against (action labels).
        a : scalar
            The action to compare.

        Returns
        -------
        ind : array-like, shape (n_samples,)
            Indicator vector that is True where 'vec' equals 'a'.
        """
        # Create a boolean vector where elements are True if action 'a' matches
        ind = vec == a
        return ind

    def basis_next_expect(self, x, a, h_x, basis):
        """
        Calculate the conditional expectation using Nadaraya-Watson kernel regression 
        and basis functions for the next state.

        Parameters
        ----------
        x : array-like, shape (p,)
            The current state (conditioning variable).
        a : scalar
            The action label to condition the expectation on.
        h_x : float or array-like, shape (p,)
            The bandwidth parameter(s) for the Gaussian kernel.
        basis : object
            The basis object used to compute basis functions.

        Returns
        -------
        BNE : ndarray
            The conditional expectation value(s) for the given state.
        """
        # Compute the normalized differences between the current state x and each point in X_
        u = (x - self.X_) / h_x
        # Calculate Gaussian kernel values based on the normalized differences
        K = np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
        
        # Create an array filled with the first element of x and append it to the next state data
        #add_x = np.full(self.X_next.shape, x[0])  # Broadcast x[0] to match the shape of X_next
        add_x = np.tile(x[:-1], (self.X_next.shape[0], 1))
        state_next_ = np.hstack((self.X_next, add_x))  # Concatenate next state data with add_x

        # Compute the basis functions for the extended next state
        basis_next_df, _ = basis.compute_basis_functions(state_next_, basis.orders)
        
        # Create an indicator vector for samples corresponding to action 'a'
        ind_A = self.indicator(self.A_, a)
        
        # Compute the numerator: the weighted sum of the basis functions using the kernel values
        BNE_num = np.mean(basis_next_df.values * np.prod(K / h_x, axis=1)[:, np.newaxis] * ind_A[:, np.newaxis], axis=0)
        # Compute the denominator: the sum of the kernel values for action 'a'
        BNE_denom = np.mean(ind_A * np.prod(K / h_x, axis=1)) + 1e-10  # Small constant to prevent division by zero
        
        # Ensure that the denominator is not zero to avoid division by zero errors
        if BNE_denom == 0:
            raise ValueError("Denominator in conditional density calculation is zero.")
        
        # Compute the conditional expectation by dividing the numerator by the denominator
        BNE = BNE_num / BNE_denom
        return BNE
    
