//! Miscellaneous linear algebra tools
//! Linear algebra tools
pub use ndarray::prelude::*;

/// Represents the L2 norm, i.e. the square-root of the sum of the squares
///
/// # Arguments
///  * `v` - the vector, of any (finite) dimension
///
/// # Returns
/// The L2 norm of that vector.
pub fn l2_norm<Ix: Dimension>(v: &Array<f64, Ix>) -> f64 {
    v.into_iter().map(|&x| x*x).sum::<f64>().sqrt()
}

/// Newton's method for the inversion of a matrix, starting from a good initial guess
///
/// # Arguments
///  * mat(n, n) - the matrix being inversed
///  * guess(n, n) - the initial guess, modified in-place. Equal to mat^{-1} on exit
///  * max_iter - the maximum number of iterations
///  * norm - A matrix norm to claim convergence
///  * tol - numerical tolerance. Convergence will be declared if the provided norm of the
///          difference between `guess.dot(mat)` and the identity is less than `tol`
///
/// # Returns
///  * Ok((niter, norm)) - the number of iterations and eventual norm of the difference
///  * Err(diff) - the norm of the difference after `max_iter`
pub fn inverse_newton<F>(
    mat: &Array2<f64>,
    guess: &mut Array2<f64>,
    max_iter: usize,
    norm: F,
    tol: f64,
) -> Result<(usize, f64), f64>
    where
        F: Fn(&Array2<f64>) -> f64,
{
    let mut bfr: Array2<f64> = guess.dot(mat);
    bfr.diag_mut().map_inplace(|x| *x -= 1.0);
    let mut diff = norm(&bfr);
    for iter in 0..max_iter {
        if diff < tol {
            return Ok((iter, diff));
        }
        bfr = guess.dot(mat).dot(guess);
        *guess *= 2.0;
        *guess -= &bfr;
        bfr = guess.dot(mat);
        bfr.diag_mut().map_inplace(|x| *x -= 1.0);
        diff = norm(&bfr);
    }
    if diff < tol {
        return Ok((max_iter, diff));
    }
    Err(diff)
}

/// Denman-Beavers iteration for finding the square-root and its inverse
/// for the input square matrix, given a suitable initial guess
///
/// # Arguments
///  * mat(n, n) - the input matrix for which the square root is required
///  * inv(n, n) - the inverse of `mat`
///  * sqrt(n, n) - initial guess for the square-root of `mat`
///  * isqrt(n, n) - initial guess for the inverse of `sqrt`
///  * max_iter - maximum number of Denman-Beavers iterations
///  * inverse_iter - maximum number of iterations for each iterative inversion
///  * norm - the norm used for checking whether convergence has been reached
///  * tol - the numerical tolerance for declaring convergence
///
/// # Returns
///  * Ok(niter, tol_sqrt, tol_inv) - the number of iterations it took to reach convergence, as
///        well as the eventual differences (which should be less than `tol`)
///  * Err(niter, tol_sqrt, tol_inv) - the final norm of the difference `sqrt.dot(sqrt.t()) - mat`
///        and between `sqrt.dot(isqrt) - id` and iteration reached
pub fn sqrt_denman_beavers<F>(
    mat: &Array2<f64>,
    inv: &Array2<f64>,
    sqrt: &mut Array2<f64>,
    isqrt: &mut Array2<f64>,
    max_iter: usize,
    inverse_iter: usize,
    norm: F,
    tol: f64,
) -> Result<(usize, f64, f64), (usize, f64, f64)>
    where
        F: Fn(&Array2<f64>) -> f64,
{
    // TODO: exploit initial guesses
    let n = mat.shape()[0];
    sqrt.assign(mat);
    isqrt.assign(inv);
    let mut bfr = sqrt.dot(&sqrt.t()) - mat;
    let mut diff = norm(&bfr);
    let mut idiff = match inverse_newton(sqrt, isqrt, inverse_iter, &norm, tol) {
        Ok((_, diff)) => diff,
        Err(idiff) => {
            return Err((0, diff, idiff));
        }
    };
    let mut zk = Array2::eye(n);
    let mut zkinv = Array2::eye(n);
    for iter in 0..max_iter {
        if diff < tol {
            return Ok((iter, diff, idiff));
        }
        *sqrt += &zkinv;
        *sqrt /= 2.0;
        zk += &*isqrt;
        zk /= 2.0;
        bfr = sqrt.dot(&sqrt.t());
        bfr -= mat;
        diff = norm(&bfr);
        idiff = inverse_newton(sqrt, isqrt, inverse_iter, &norm, tol)
            .map_err(|inv| (iter + 1, diff, inv))?
            .1;
        if let Err(_) = inverse_newton(&zk, &mut zkinv, inverse_iter, &norm, tol) {
            return Err((iter + 1, diff, idiff));
        }
    }
    if diff < tol {
        Ok((max_iter, diff, idiff))
    } else {
        Err((max_iter, diff, idiff))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::*;

    pub const TOL: f64 = 1e-15;
    pub const MAX_ITER: usize = 100;

    /// Returns a nice 2 x 2 matrix
    fn matrices() -> Vec<Array2<f64>> {
        vec![arr2(&[[1.0, -0.5], [-0.5, 1.0]])]
    }
    #[test]
    fn test_inverse_newton() {
        for mat in matrices() {
            let mut inv = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
            match inverse_newton(&mat, &mut inv, MAX_ITER, &l2_norm, TOL) {
                Ok((niter, tol)) => {
                    assert!(niter <= MAX_ITER);
                    assert!(tol < TOL);
                }
                Err(tol) => {
                    panic!(
                        "Error obtaining inverse after {} iterations: norm of diff is {:.9}",
                        MAX_ITER, tol
                    );
                }
            }
            assert_inverse(&mat, &inv, TOL);
        }
    }
    #[test]
    fn test_sqrt_denman_beavers() {
        for mat in matrices() {
            let mut inv = Array2::eye(2);
            inverse_newton(&mat, &mut inv, MAX_ITER, l2_norm, TOL)
                .expect("Failed to take inverse");
            let mut sqrt = Array2::eye(2);
            let mut isqrt = Array2::eye(2);
            match sqrt_denman_beavers(
                &mat,
                &inv,
                &mut sqrt,
                &mut isqrt,
                MAX_ITER,
                MAX_ITER,
                l2_norm,
                TOL,
            ) {
                Ok((niter, tol_sqrt, tol_inv)) => {
                    assert!(niter <= MAX_ITER);
                    assert!(tol_sqrt < TOL);
                    assert!(tol_inv < TOL);
                }
                Err((iter, diff_sqrt, diff_inv)) => {
                    panic!(
                        "Error obtaining square-root after {} iterations: norm of diff is {:.9} (sqrt) {:.9} (inv)",
                        iter, diff_sqrt, diff_inv
                    );
                }
            }
        }
    }
}
