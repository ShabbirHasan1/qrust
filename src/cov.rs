//! Robust representation of correlation/covariance matrices

use ndarray::Array2;

/// Representation of a covariance matrix as a pair (Q, s) where Q is an orthogonal matrix
/// (i.e. Q . Q' = I), s is the vector of (positive) diagonal components such that the covariance
/// matrix M is equal to M = Q . diag(s) . Q´
pub struct Covariance {
    /// The orthogonal matrix
    q: Array2<f64>,
}
