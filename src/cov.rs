//! Robust representation of correlation/covariance matrices

// use ndarray::prelude::*;
// use ndarray_linalg::error::LinalgError;
// use ndarray_linalg::{cholesky::CholeskyInplace, Inverse, UPLO};
//
// const VAR_MIN: f64 = 1e-21;
//
// /// Representation of a covariance matrix as a pair (Q, s) where Q is an orthogonal matrix
// /// (i.e. Q . Q' = I), s is the vector of (positive) diagonal components such that the covariance
// /// matrix M is equal to M = Q . diag(s) . Q´
// pub struct Covariance {
//     /// The correlation
//     sqrt_corr: Array2<f64>,
//     /// The unconditional standard deviation of each random variable
//     stdevs: Array1<f64>,
// }
// impl Covariance {
//     /// Creates a new 'identity' covariance
//     pub fn identity(n: usize) -> Self {
//         Self::independant(Array1::ones(n))
//     }
//     /// Variance for independent random variables with given stdevs
//     pub fn independant(stdevs: Array1<f64>) -> Self {
//         Self {
//             sqrt_corr: Array2::eye(stdevs.len()),
//             stdevs,
//         }
//     }
//     pub fn new<T: Into<Array2<f64>>>(data: T) -> Result<Self, LinalgError> {
//         let mut arr = data.into();
//         let stdevs = arr.diag().map(|x| x.max(VAR_MIN).sqrt());
//         stdevs
//             .iter()
//             .zip(arr.rows_mut().into_iter())
//             .for_each(|(&std, mut row)| {
//                 row /= std;
//             });
//         stdevs
//             .iter()
//             .zip(arr.columns_mut().into_iter())
//             .for_each(|(&std, mut col)| {
//                 col /= std;
//             });
//         // `arr` is now the associated correlation matrix
//         Ok(Covariance {
//             sqrt_corr: arr,
//             stdevs,
//         })
//     }
//     /// The dimension of the covariance matrix
//     pub fn dim(&self) -> usize {
//         self.sqrt_corr.shape()[0]
//     }
//     /// The rank for the covariance
//     pub fn rank(&self) -> usize {
//         self.stdevs.len()
//     }
//     /// The inverse of the covariance, which is also a covariance EXCEPT IF
//     /// TODO: include a `Regularize` trait callable from object or error-produced
//     ///       for chaining operations
//     pub fn inv(&self) -> Result<Covariance, LinalgError> {
//         if self.stdevs.iter().all(|x| x.is_normal()) {
//             let inv = 1.0 / &self.stdevs;
//             Ok(Self {
//                 sqrt_corr: self.sqrt_corr.inv()?,
//                 stdevs: inv,
//             })
//         } else {
//             Err(LinalgError::MemoryNotCont)
//         }
//     }
//     /// The variance corresponding to the linear combination of components with this covariance
//     /// using the provided weights, i.e.:
//     ///     `variance = Var(weights/ . x/)`
//     ///     `x/ ~ N(0/, Self//)`
//     /// Or:
//     ///     `variance = weights' . Self . weights`
//     ///
//     pub fn variance(&self, weights: &Array1<f64>) -> f64 {
//         let mut x = self.sqrt_corr.t().dot(weights);
//         x *= &self.stdevs;
//         x.dot(&x)
//     }
//     /// The square-root of the covariance
//     pub fn sqrt(&self) -> Array2<f64> {
//         let mut out = self.sqrt_corr.clone();
//         for (&s, mut col) in self.stdevs.iter().zip(out.columns_mut().into_iter()) {
//             col *= s;
//         }
//         out
//     }
//     /// Representation of the covariance matrix as an array
//     pub fn as_array(&self) -> Array2<f64> {
//         let s = self.sqrt();
//         s.dot(&s.t())
//     }
// }
//
// #[cfg(test)]
// mod tests {
//     use super::*;
//     #[test]
//     fn test_sqrt() {
//         let cov = Covariance::new(vec![[1.0, 0.5], [0.5, 1.0]]).unwrap();
//     }
// }
