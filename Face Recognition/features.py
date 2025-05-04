"""
CS131 - Computer Vision: Foundations and Applications
Author: Donsuk Lee (donlee90@stanford.edu)
Python Version: 3.5+
"""

import numpy as np
import scipy
import scipy.linalg


class PCA(object):
    """Class implementing Principal component analysis (PCA).

    Steps to perform PCA on a matrix of features X:
        1. Fit the training data using method `fit` (either with eigen decomposition of SVD)
        2. Project X into a lower dimensional space using method `transform`
        3. Optionally reconstruct the original X (as best as possible) using method `reconstruct`
    """

    def __init__(self):
        self.W_pca = None
        self.mean = None

    def fit(self, X, method='svd'):
        """Fit the training data X using the chosen method.

        Will store the projection matrix in self.W_pca and the mean of the data in self.mean

        Args:
            X: numpy array of shape (N, D). Each of the N rows represent a data point.
               Each data point contains D features.
            method: Method to solve PCA. Must be one of 'svd' or 'eigen'.
        """
        _, D = X.shape
        

        # YOUR CODE HERE
        # 1. Compute the mean and store it in self.mean
        # 2. Apply either method to `X_centered`
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # 1. Compute the mean of X
        self.mean = np.mean(X, axis=0)
        # 2. Zero-center the data using the computed mean
        X_centered = X - self.mean

        # 3. Compute the principal components using the specified method.
        if method == 'svd':
            vecs, vals = self._svd(X_centered)
            self.W_pca = vecs
        elif method == 'eigen':
            e_vecs, e_vals = self._eigen_decomp(X_centered)
            self.W_pca = e_vecs
        else:
            raise ValueError("Method must be either 'svd' or 'eigen'.")

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # END YOUR CODE

        # Make sure that X_centered has mean zero
        assert np.allclose(X_centered.mean(), 0.0)

        # Make sure that self.mean is set and has the right shape
        assert self.mean is not None and self.mean.shape == (D,)

        # Make sure that self.W_pca is set and has the right shape
        assert self.W_pca is not None and self.W_pca.shape == (D, D)

        # Each column of `self.W_pca` should have norm 1 (each one is an eigenvector)
        for i in range(D):
            assert np.allclose(np.linalg.norm(self.W_pca[:, i]), 1.0)

    def _eigen_decomp(self, X):
        """Performs eigendecompostion of feature covariance matrix.

        Args:
            X: Zero-centered data array, each ROW containing a data point.
               Numpy array of shape (N, D).

        Returns:
            e_vecs: Eigenvectors of covariance matrix of X. Eigenvectors are
                    sorted in descending order of corresponding eigenvalues. Each
                    column contains an eigenvector. Numpy array of shape (D, D).
            e_vals: Eigenvalues of covariance matrix of X. Eigenvalues are
                    sorted in descending order. Numpy array of shape (D,).
        """
        N, D = X.shape
        # YOUR CODE HERE
        # Steps:
        #     1. compute the covariance matrix of X, of shape (D, D)
        #     2. compute the eigenvalues and eigenvectors of the covariance matrix
        #     3. Sort both of them in decreasing order (ex: 1.0 > 0.5 > 0.0 > -0.2 > -1.2)
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # 1. Compute the covariance matrix of X.
        # Using (N-1) as the denominator gives the unbiased sample covariance.
        cov_matrix = np.dot(X.T, X) / (N - 1) if N > 1 else np.dot(X.T, X)
        
        # 2. Compute the eigenvalues and eigenvectors.
        # Since cov_matrix is symmetric, np.linalg.eigh is appropriate.
        eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
        
        # 3. Sort the eigenvalues (and corresponding eigenvectors) in descending order.
        sort_indices = np.argsort(eig_vals)[::-1]   # Indices for descending order
        e_vals = eig_vals[sort_indices]
        e_vecs = eig_vecs[:, sort_indices]

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # END YOUR CODE

        # Check the output shapes
        assert e_vals.shape == (D,)
        assert e_vecs.shape == (D, D)

        return e_vecs, e_vals

    def _svd(self, X):
        """Performs Singular Value Decomposition (SVD) of X.

        Args:
            X: Zero-centered data array, each ROW containing a data point.
                Numpy array of shape (N, D).
        Returns:
            vecs: right singular vectors. Numpy array of shape (D, D)
            vals: singular values. Numpy array of shape (K,) where K = min(N, D)
        """
        N, D = X.shape
        # YOUR CODE HERE
        # Here, compute the SVD of X
        # Make sure to return vecs as the matrix of vectors where each column is a singular vector
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Compute the full SVD so that Vt (and, after transposition, vecs) is of shape (D, D).
        # X = U * diag(s) * Vt
        U, s, Vt = np.linalg.svd(X, full_matrices=True)
        
        # The right singular vectors are the rows of Vt; we take the transpose to have
        # each column be a singular vector.
        vecs = Vt.T
        vals = s
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # END YOUR CODE
        assert vecs.shape == (D, D)
        K = min(N, D)
        assert vals.shape == (K,)

        return vecs, vals

    def transform(self, X, n_components):
        """Center and project X onto a lower dimensional space using self.W_pca.

        Args:
            X: numpy array of shape (N, D). Each row is an example with D features.
            n_components: number of principal components..

        Returns:
            X_proj: numpy array of shape (N, n_components).
        """
        N, _ = X.shape
        # YOUR CODE HERE
        # We need to modify X in two steps:
        #     1. first substract the mean stored during `fit`
        #     2. then project onto a subspace of dimension `n_components` using `self.W_pca`
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

         # 1. First, subtract the mean stored in self.mean (computed in fit)
        X_centered = X - self.mean
        
        # 2. Then project the zero-centered data onto the subspace defined by the 
        # top n_components columns of self.W_pca.
        # Since self.W_pca has shape (D, D), selecting the first n_components columns
        # gives us a projection matrix of shape (D, n_components), and the multiplication
        # results in X_proj of shape (N, n_components)
        X_proj = np.dot(X_centered, self.W_pca[:, :n_components])

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # END YOUR CODE

        assert X_proj.shape == (N, n_components), "X_proj doesn't have the right shape"

        return X_proj

    def reconstruct(self, X_proj):
        """Do the exact opposite of method `transform`: try to reconstruct the original features.

        Given the X_proj of shape (N, n_components) obtained from the output of `transform`,
        we try to reconstruct the original X.

        Args:
            X_proj: numpy array of shape (N, n_components). Each row is an example with D features.

        Returns:
            X: numpy array of shape (N, D).
        """
        N, n_components = X_proj.shape
        

        # YOUR CODE HERE
        # Steps:
        #     1. project back onto the original space of dimension D
        #     2. add the mean that we substracted in `transform`
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Step 1: Project the low-dimensional representation X_proj back to the original D-dimensional space.
        #         Here, self.W_pca[:, :n_components] is the projection matrix used in transform.
        X_reconstructed = np.dot(X_proj, self.W_pca[:, :n_components].T)

        # Step 2: Add back the mean that was subtracted in the transform step to recover the original scale.
        X = X_reconstructed + self.mean

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # END YOUR CODE

        return X


class LDA(object):
    """Class implementing Principal component analysis (LDA).

    Steps to perform LDA on a matrix of features X:
        1. Fit the training data using method `fit`
        2. Project X into a lower dimensional space using method `transform`
    """

    def __init__(self):
        self.W_lda = None

    def fit(self, X, y):
        """Fit the training data `X` using the labels `y`.

        Will store the projection matrix in `self.W_lda`.

        Args:
            X: numpy array of shape (N, D). Each of the N rows represent a data point.
               Each data point contains D features.
            y: numpy array of shape (N,) containing labels of examples in X
        """
        N, D = X.shape

        scatter_between = self._between_class_scatter(X, y)
        scatter_within = self._within_class_scatter(X, y)

        # YOUR CODE HERE
        # Solve generalized eigenvalue problem for matrices `scatter_between` and `scatter_within`
        # Use `scipy.linalg.eig` instead of numpy's eigenvalue solver.
        # Don't forget to sort the values and vectors in descending order.
        # If SVD is used, take the eigen vectors from U

        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Solve the generalized eigenvalue problem for scatter_between and scatter_within
        eigvals, eigvecs = scipy.linalg.eig(scatter_between, scatter_within)

        # Since the matrices are real symmetric, the imaginary parts should be negligible;
        # take only the real parts.
        eigvals = eigvals.real
        eigvecs = eigvecs.real

        # Sort the eigenvalues in descending order and sort eigenvectors accordingly
        sort_indices = np.argsort(eigvals)[::-1]
        e_vecs = eigvecs[:, sort_indices]

        # Normalize each eigenvector (each column) to unit norm.
        for i in range(e_vecs.shape[1]):
            norm = np.linalg.norm(e_vecs[:, i])
            # Avoid division by zero.
            if norm > 0:
                e_vecs[:, i] /= norm

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # END YOUR CODE

        self.W_lda = e_vecs

        # Check that the shape of self.W_lda is correct.
        assert self.W_lda.shape == (D, D)

        # Each column of self.W_lda should have norm 1.
        for i in range(D):
            assert np.allclose(np.linalg.norm(self.W_lda[:, i]), 1.0), f"Column {i} norm is {np.linalg.norm(self.W_lda[:, i])}"

    def _within_class_scatter(self, X, y):
        """Compute the covariance matrix of each class, and sum over the classes.

        For every label i, we have:
            - X_i: matrix of examples with labels i
            - S_i: covariance matrix of X_i (per class covariance matrix for class i)
        The formula for covariance matrix is: X_centered^T X_centered
            where X_centered is the matrix X with mean 0 for each feature.

        Our result `scatter_within` is the sum of all the `S_i`

        Args:
            X: numpy array of shape (N, D) containing N examples with D features each
            y: numpy array of shape (N,), labels of examples in X

        Returns:
            scatter_within: numpy array of shape (D, D), sum of covariance matrices of each label
        """
        _, D = X.shape
        assert X.shape[0] == y.shape[0]
        scatter_within = np.zeros((D, D))

        for i in np.unique(y):
            # YOUR CODE HERE
            # Get the covariance matrix for class i, and add it to scatter_within
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # Extract all examples in class i
            X_i = X[y == i]
            # Compute the mean for class i
            mu_i = np.mean(X_i, axis=0)
            # Center the class data by subtracting the class mean
            X_centered_i = X_i - mu_i
            # Compute the covariance matrix (scatter matrix) for class i
            S_i = np.dot(X_centered_i.T, X_centered_i)
            # Accumulate the scatter matrix for the class into scatter_within
            scatter_within += S_i

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            # END YOUR CODE

        return scatter_within

    def _between_class_scatter(self, X, y):
        """Compute the covariance matrix as if each class is at its mean.

        For every label i, we have:
            - X_i: matrix of examples with labels i
            - mu_i: mean of X_i.

        Args:
            X: numpy array of shape (N, D) containing N examples with D features each
            y: numpy array of shape (N,), labels of examples in X

        Returns:
            scatter_between: numpy array of shape (D, D)
        """
        _, D = X.shape
        assert X.shape[0] == y.shape[0]
        scatter_between = np.zeros((D, D))

        mu = X.mean(axis=0)
        for i in np.unique(y):
            # YOUR CODE HERE
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # Extract examples belonging to class i.
            X_i = X[y == i]
            N_i = X_i.shape[0]  # Number of examples in class i
            
            # Compute the mean of class i.
            mu_i = np.mean(X_i, axis=0)
            
            # Compute the difference between the class mean and the global mean.
            diff = mu_i - mu
            
            # Accumulate the scatter contribution for class i.
            scatter_between += N_i * np.outer(diff, diff)

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            # END YOUR CODE

        return scatter_between

    def transform(self, X, n_components):
        """Project X onto a lower dimensional space using self.W_lda.

        Args:
            X: numpy array of shape (N, D). Each row is an example with D features.
            n_components: number of principal components..

        Returns:
            X_proj: numpy array of shape (N, n_components).
        """
        N, _ = X.shape
        # YOUR CODE HERE
        # project onto a subspace of dimension `n_components` using `self.W_lda`
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        X_proj = np.dot(X, self.W_lda[:, :n_components])

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # END YOUR CODE

        assert X_proj.shape == (N, n_components), "X_proj doesn't have the right shape"

        return X_proj
