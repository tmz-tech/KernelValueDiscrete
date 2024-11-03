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
    """
    Custom estimator using Nadaraya-Watson kernel regression to calculate the next expectation
    of basis functions conditioned on state variables.

    Parameters
    ----------
    bandwidth : float, default=1.0
        The bandwidth parameter that controls the width of the Gaussian kernel, affecting the
        smoothness of the kernel density estimation.
    alpha : float, default=0.1
        The regularization term added to the kernel weights to ensure numerical stability.
        Adding a small positive value helps prevent extremely small or zero weights, which
        could otherwise result in unstable or biased estimates.
    """

    def __init__(self, bandwidth=1.0, alpha=0.1):
        # Initialize the estimator with specified bandwidth and regularization alpha
        self.bandwidth = bandwidth
        self.alpha = alpha

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
        Calculate the conditional expectation of basis functions for the next state, 
        conditioned on the current state, using Kernel Density Estimation (KDE).

        Parameters
        ----------
        x : array-like, shape (p,)
            The conditioning variable (current state) for which the conditional expectation is computed.
        h_x : float or array-like, shape (p,)
            The bandwidth parameter(s) controlling the smoothness of the Gaussian kernel for each dimension.
        basis : object
            An object representing the basis function model, providing a method to compute basis 
            functions based on the given orders and input data.

        Returns
        -------
        BNE : ndarray
            The conditional expectation value(s) of the basis functions for the next state, given the current state `x`.
        """

        # Extend each row of the next state data by appending the relevant components of the current state `x` (excluding the last element).
        # This creates a combined dataset that includes both the current state information (up to the second-to-last element of `x`) 
        # and the data of the next state to facilitate the basis function computation for the next state given `x`.
        add_x = np.tile(x[:-1], (self.X_next.shape[0], 1))
        state_next_ = np.hstack((self.X_next, add_x)) # Concatenate current state (up to last element) with each row of next state data
        
        # Compute the basis functions for the concatenated next state data, using the basis function model.
        # basis_next_df contains the evaluated basis functions for each extended next state.
        basis_next_df, _ = basis.compute_basis_functions(state_next_, basis.orders)
        
        # Calculate the pairwise differences between `x` (current state) and the fitted states `self.X_`
        u_x = (x - self.X_) / h_x # Normalized difference between the input and stored states
        
        # Compute Gaussian kernel weights for the current state, `x`, based on stored states `self.X_`.
        # This results in `K_x`, which indicates the weight of each stored state relative to `x`.
        K_x = np.exp(-0.5 * np.sum(u_x**2, axis=1)) / ((2 * np.pi)**(self.X_.shape[1] / 2) * h_x**self.X_.shape[1])
        K_x += self.alpha # Regularize the kernel weights by adding `self.alpha`.

        # Compute the numerator as the weighted sum of basis function values for the next state.
        # Perform element-wise multiplication of basis function values with kernel weights.
        BNE_num = np.sum(basis_next_df.values * K_x[:, np.newaxis], axis=0)

        # Calculate the denominator as the sum of kernel weights.
        # Add a small constant `1e-10` to prevent division by zero and ensure numerical stability.
        BNE_denom = np.sum(K_x) + 1e-10
        
        # Compute the conditional expectation by dividing the weighted sum of basis functions by the total kernel weights.
        BNE = BNE_num/BNE_denom
        
        return BNE
    
    

class BasisNextSAExpect(BaseEstimator):
    """
    Custom estimator for calculating the conditional expectation of the next state 
    and action values based on Nadaraya-Watson kernel regression and basis functions.
    This estimator is used to compute the expected values of basis functions given the 
    current state-action pair and leverages a Gaussian kernel for smoothing.

    Parameters
    ----------
    bandwidth : float, default=1.0
        The bandwidth parameter that controls the width of the Gaussian kernel, 
        affecting the smoothness of the kernel density estimation.
    alpha : float, default=0.1
        The regularization term added to the kernel weights to ensure numerical stability.
        Adding a small positive value helps prevent extremely small or zero weights, which
        could otherwise result in unstable or biased estimates.
    """
    
    def __init__(self, bandwidth=1.0, alpha = 0.1):
        """
        Initialize the BasisNextSAExpect estimator with specified parameters.

        Parameters
        ----------
        bandwidth : float, default=1.0
            The bandwidth parameter for the Gaussian kernel, affecting the influence 
            range of each data point in the kernel density estimation.
        alpha : float, default=0.1
            Regularization term added to the kernel weights to avoid issues such as 
            division by zero during kernel-based computations.
        """
        self.bandwidth = bandwidth
        self.alpha = alpha

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
        Calculate the conditional expectation of the basis functions for the next state
        given the current state-action pair, using Nadaraya-Watson kernel regression.

        Parameters
        ----------
        x : array-like, shape (p,)
            The current state (conditioning variable).
        a : scalar
            The action label to condition the expectation on.
        h_x : float or array-like, shape (p,)
            The bandwidth parameter(s) for the Gaussian kernel.
        basis : object
            The basis object used to compute basis functions, providing the basis functions 
            and their computation method.

        Returns
        -------
        BNE : ndarray
            The estimated conditional expectation value(s) for the given state-action pair.
        """

        # Create an array by tiling the current state's first elements, used to extend the next state data for conditioning
        add_x = np.tile(x[:-1], (self.X_next.shape[0], 1)) # Repeat elements of `x` to match next state data rows
        state_next_ = np.hstack((self.X_next, add_x)) # Concatenate current and next state data
        
        # Compute the basis functions for the extended next state data
        basis_next_df, _ = basis.compute_basis_functions(state_next_, basis.orders)
        
        # Compute the normalized pairwise distances between the input state-action pair and the stored training data
        u_x = (x - self.X_) / h_x # Scale differences for states by bandwidth
        
        # Apply the Gaussian kernel to the scaled differences (state)
        K_x = np.exp(-0.5 * np.sum(u_x**2, axis=1)) / ((2 * np.pi)**(self.X_.shape[1] / 2) * h_x**self.X_.shape[1])
        
        # Create an indicator vector for samples corresponding to action 'a'
        ind_A = self.indicator(self.A_, a) # Binary indicator array for action `a`
        
        # Multiply the Gaussian kernel weights by the action indicator for joint state-action smoothing
        K= K_x * ind_A
        #K+=self.alpha # Adding self.alpha ensures regularization to prevent overfitting

        # Compute the numerator: weighted sum of the basis function values for the next state
        BNE_num = np.sum(basis_next_df.values * K[:, np.newaxis], axis=0) # Element-wise multiplication of kernel weights
        
        # Calculate the denominator as the sum of kernel weights
        # A small constant (1e-10) is included to avoid division by zero
        BNE_denom = np.sum(K) + 1e-10
        
        # Calculate the conditional expectation by dividing the weighted sum of basis functions by the kernel weights
        BNE = BNE_num/BNE_denom
        
        
        
        
        
        
                      
            
        return BNE

    

class BasisNextSAExpectData(BaseEstimator):
    """
    Custom estimator for calculating the conditional expectation of the next state 
    and action values based on Nadaraya-Watson kernel regression and basis functions.
    This estimator is used to compute the expected values of basis functions given the 
    current state-action pair and leverages a Gaussian kernel for smoothing.

    Parameters
    ----------
    bandwidth : float, default=1.0
        The bandwidth parameter that controls the width of the Gaussian kernel, 
        affecting the smoothness of the kernel density estimation.
    alpha : float, default=0.1
        The regularization term added to the kernel weights to ensure numerical stability.
        Adding a small positive value helps prevent extremely small or zero weights, which
        could otherwise result in unstable or biased estimates.
    """
    
    def __init__(self, bandwidth=1.0, alpha = 0.1):
        """
        Initialize the BasisNextSAExpect estimator with specified parameters.

        Parameters
        ----------
        bandwidth : float, default=1.0
            The bandwidth parameter for the Gaussian kernel, affecting the influence 
            range of each data point in the kernel density estimation.
        alpha : float, default=0.1
            Regularization term added to the kernel weights to avoid issues such as 
            division by zero during kernel-based computations.
        """
        self.bandwidth = bandwidth
        self.alpha = alpha

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
        Calculate the conditional expectation of the basis functions for the next state
        given the current state-action pair, using Nadaraya-Watson kernel regression.

        Parameters
        ----------
        x : array-like, shape (p,)
            The current state (conditioning variable).
        a : scalar
            The action label to condition the expectation on.
        h_x : float or array-like, shape (p,)
            The bandwidth parameter(s) for the Gaussian kernel.
        basis : object
            The basis object used to compute basis functions, providing the basis functions 
            and their computation method.

        Returns
        -------
        BNE : ndarray
            The estimated conditional expectation value(s) for the given state-action pair.
        """

        # Create an array by tiling the current state's first elements, used to extend the next state data for conditioning
        add_x = np.tile(x[:-1], (self.X_next.shape[0], 1)) # Repeat elements of `x` to match next state data rows
        state_next_ = np.hstack((self.X_next, add_x)) # Concatenate current and next state data
        
        # Compute the basis functions for the extended next state data
        basis_next_df, _ = basis.compute_basis_functions(state_next_, basis.orders)
        
        # Compute the normalized pairwise distances between the input state-action pair and the stored training data
        u_x = (x - self.X_) / h_x # Scale differences for states by bandwidth
        
        # Apply the Gaussian kernel to the scaled differences (state)
        K_x = np.exp(-0.5 * np.sum(u_x**2, axis=1)) / ((2 * np.pi)**(self.X_.shape[1] / 2) * h_x**self.X_.shape[1])
        
        # Create an indicator vector for samples corresponding to action 'a'
        ind_A = self.indicator(self.A_, a) # Binary indicator array for action `a`
        
        # Multiply the Gaussian kernel weights by the action indicator for joint state-action smoothing
        K= K_x * ind_A
        #K+=self.alpha # Adding self.alpha ensures regularization to prevent overfitting

        # Compute the numerator: weighted sum of the basis function values for the next state
        BNE_num = np.sum(basis_next_df.values * K[:, np.newaxis], axis=0) # Element-wise multiplication of kernel weights
        
        # Calculate the denominator as the sum of kernel weights
        # A small constant (1e-10) is included to avoid division by zero
        BNE_denom = np.sum(K) + 1e-10
        
        
        # Calculate the conditional expectation by dividing the weighted sum of basis functions by the kernel weights
        BNE = BNE_num/BNE_denom
        
            
        return BNE
    
