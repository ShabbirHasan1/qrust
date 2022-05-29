//! Optimisation and root finding

/// The errors which can be raised when doing numerical optimisation
#[derive(Clone, Debug)]
pub enum Error {
    /// A numerical error (typically, division by zero) prevents the algorithm to run,
    /// along with its details
    NumericalError(String),
    /// The maximum number of iterations has been reached
    MaxIterations,
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Error::NumericalError(s) => write!(f, "Numerical error: {}", s),
            Error::MaxIterations => write!(f, "Maximum number of iterations reached"),
        }
    }
}
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}
/// The result of a 1D root-finding algorithm
pub struct RootFinding1d {
    /// The value of the root
    x: f64,
    /// The function evaluation in `x`
    fx: f64,
    /// The number of iterations it took
    niter: usize,
    /// The error which triggered the algorithm to stop, if any
    error: Option<Error>,
}
impl RootFinding1d {
    /// The found root value
    pub fn value(&self) -> f64 {
        self.x
    }
    /// Evaluation of the function at the root
    pub fn fn_eval(&self) -> f64 {
        self.fx
    }
    /// Number of iterations elapsed
    pub fn niter(&self) -> usize {
        self.niter
    }
    /// Whether the root-finding was successful or not
    pub fn success(&self) -> bool {
        self.error.is_none()
    }
    /// The error raised by the root-finding, if any
    pub fn error(&self) -> &Option<Error> {
        &self.error
    }
}

/// Simple unidimensional root-finding using the Newton-Raphson algorithm
///
/// The Newton-Raphson method is a simple iterative method which approximates a root of the
/// function by doing the following iterations:
///
/// x_{i+1} = x_i - f(x_i) / f'(x_i)
///
/// Where f' is the first derivative of the function f.
/// This is a simple and quick method, which converges quadratically to a root,
/// if at least one exists. If the second derivative is easy to compute, `halley`
/// can be faster.
///
/// # Arguments
///  * f - the function for which the zeroes are required, as well as its first derivative
///  * guess - the initial guess from whence the optimisation starts
///  * max_iter - the maximum number of iterations, after which a 'failed' result is returned
///  * tol - the tolerance: if the value of the function is less than `tol` in absolue value,
///          convergence is declared and the present value is returned
///
/// # Returns
/// res - a `RootFinding1d` result storing the value of the root, function at the root, etc
///
/// # See also
///  * `RootFinding1d` - the type of result returned
///  * `halley` - an alternative method for 1d root-finding
pub fn newton_raphson<F>(f: F, guess: f64, max_iter: usize, tol: f64) -> RootFinding1d
    where
        F: Fn(f64) -> (f64, f64),
{
    let mut x = guess;
    let (mut fx, mut dfx) = f(x);
    for niter in 0..max_iter {
        if fx.abs() < tol {
            return RootFinding1d {
                x,
                fx,
                niter,
                error: None,
            };
        }
        if !dfx.is_normal() {
            return RootFinding1d {
                x,
                fx,
                niter,
                error: Some(Error::NumericalError(
                    format!("Newton denominator is subnormal: {dfx}", dfx=dfx)
                )),
            };
        }
        x -= fx / dfx;
        let tup = f(x);
        fx = tup.0;
        dfx = tup.1;
    }
    RootFinding1d {
        x,
        fx,
        niter: max_iter,
        error: Some(Error::MaxIterations),
    }
}
/// Simple unidimensional root-finding using the Halley algorithm
///
/// The Halley method is a simple iterative method which approximates a root of the
/// function by doing the following iterations:
///
/// x_{i+1} = x_i - (2. * f(x_i) * f'(x_i)) / (2 * f'(x_i)^2 - f(x_i) * f''(x_i))
///
/// Where f' is the first derivative of the function f, and f'' its second derivative.
///
/// This is a simple and quick method, which converges cubically to a root,
/// if at least one exists.
///
/// # Arguments
///  * f - the function for which the zeroes are required, returning the function,
///        its first and second derivative at a given point
///  * guess - the initial guess from whence the optimisation starts
///  * max_iter - the maximum number of iterations, after which a 'failed' result is returned
///  * tol - the tolerance: if the value of the function is less than `tol` in absolue value,
///          convergence is declared and the present value is returned
///
/// # Returns
/// res - a `RootFinding1d` result storing the value of the root, function at the root, etc
///
/// # See also
///  * `RootFinding1d` - the type of result returned
///  * `newton_raphson` - an alternative method for 1d root-finding,
///                       which does not require the second derivative
pub fn halley<F>(f: F, guess: f64, max_iter: usize, tol: f64) -> RootFinding1d
    where
        F: Fn(f64) -> (f64, f64, f64),
{
    let mut x = guess;
    let (mut fx, mut dfx, mut d2fx) = f(x);
    for niter in 0..max_iter {
        if fx.abs() < tol {
            return RootFinding1d {
                x,
                fx,
                niter,
                error: None,
            };
        }
        let denom = (2. * dfx * dfx) - (fx * d2fx);
        if !denom.is_normal() {
            return RootFinding1d {
                x,
                fx,
                niter,
                error: Some(Error::NumericalError(
                    format!("Halley denominator is subnormal: {denom}", denom=denom))
                ),
            };
        }
        x -= (2. * fx * dfx) / denom;
        let tup = f(x);
        fx = tup.0;
        dfx = tup.1;
        d2fx = tup.2;
    }
    RootFinding1d {
        x,
        fx,
        niter: max_iter,
        error: Some(Error::MaxIterations),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    /// A simple polynomial: -x^3 + 3x^2 + 2x + 5
    fn poly(x: f64) -> f64 {
        -x * x * x + 3.0 * x * x - 2.0 * x + 5.0
    }
    /// First derivative of the polynomial: -3x^2 + 6x - 2
    fn dpoly(x: f64) -> f64 {
        -3.0 * x * x + 6.0 * x - 2.0
    }
    /// Second derivative of the polynomial: -6x + 6
    fn d2poly(x: f64) -> f64 {
        -6.0 * x + 6.0
    }
    #[test]
    fn newton_poly() {
        let root = newton_raphson(|x| (poly(x), dpoly(x)), 0.0, 100, 1e-12);
        assert!(root.success());
        assert!(poly(root.x) < 1e-12);
    }
    #[test]
    fn halley_poly() {
        let root = halley(|x| (poly(x), dpoly(x), d2poly(x)), 0.0, 100, 1e-12);
        assert!(root.success());
        assert!(poly(root.x) < 1e-12);
    }
    #[test]
    fn newton_nozero() {
        // No real root for x^2 + 1.0
        let root = newton_raphson(|x| (x * x + 1.0, 2. * x), 5.0, 100, 1e-12);
        assert!(!root.success());
    }
    #[test]
    fn halley_nozero() {
        // No real root for x^2 + 1.0
        let root = halley(|x| (x * x + 1.0, 2. * x, 2.0), 5.0, 100, 1e-12);
        assert!(!root.success());
    }
}
