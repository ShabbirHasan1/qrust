//! Definition of a marginal distribution of probability
use std::f64::consts::{FRAC_1_SQRT_2, FRAC_2_SQRT_PI, LN_2, PI, SQRT_2};
use special::Error;
use crate::misc::softplus;
use super::Proba;

/// 1 / (sqrt(2 * pi))
const FRAC_1_SQRT_2_PI: f64 = FRAC_2_SQRT_PI * FRAC_1_SQRT_2 / 2.0;
/// ln(pi)
const LN_PI: f64 = 1.1447298858494001638774761886452324688434600830078125;

/// Represents a one-dimensional marginal distribution of probability over R,
/// with finite first moment.
pub trait Marginal {
    /// The expected value of the marginal distribution.
    ///
    /// # Examples
    /// ```
    /// use qrust::proba::{Uniform01, Marginal};
    ///
    /// let u = Uniform01;
    /// assert_eq!(u.mean(), 0.5);
    /// let shifted = u.shift(1.0);
    /// assert_eq!(shifted.mean(), 1.5);
    /// let scaled = shifted.scale(2.0);
    /// assert_eq!(scaled.mean(), 3.0);
    /// ```
    fn mean(&self) -> f64;
    /// The variance of the distribution.
    ///
    /// Note that this will always be positive, but may be infinite (e.g. for a Cauchy distribution)
    ///
    /// # Examples
    /// ```
    /// use qrust::proba::{StandardNormal, Marginal};
    ///
    /// let n = StandardNormal;
    /// assert_eq!(n.variance(), 1.0);
    /// // Variance is unaffected by simple shifts
    /// let shifted = n.shift(1.0);
    /// assert_eq!(shifted.variance(), 1.0);
    /// // Variance is quadratically affected by scalings
    /// let scaled = shifted.scale(2.0);
    /// assert_eq!(scaled.variance(), 4.0);
    /// ```
    fn variance(&self) -> f64;
    /// The standard-deviation of the distribution, defined as the square-root of the variance.
    ///
    /// This has a default implementation of computing the variance and simply taking the
    /// square-root of the value, but implementors may override this with something more efficient.
    /// Note that this will always be non-negative, but may be infinite (e.g. in the case of a
    /// Cauchy distribution)
    ///
    /// # Examples
    /// ```
    /// use qrust::proba::{StandardNormal, Marginal};
    ///
    /// let n = StandardNormal;
    /// assert_eq!(n.variance(), 1.0);
    /// assert_eq!(n.std(), 1.0);
    /// let s = n.scale(2.0);
    /// assert_eq!(s.variance(), 4.0);
    /// assert_eq!(s.std(), 2.0);
    /// ```
    fn std(&self) -> f64 {
        self.variance().sqrt()
    }
    /// The probability density function.
    ///
    /// Note that this is always defined on floats, i.e. on R. Any distribution with a smaller
    /// domain than R should simply return 0.0 for the pdf outside the domain of definition.
    ///
    /// # Arguments
    /// * x - Value in R for which the PDF is required
    ///
    /// # Returns
    /// * pdf - The density of probability at that point. May be exactly 0.0 for points outside the
    ///         domain of definition of the distribution
    ///
    /// # Examples
    /// ```
    /// use qrust::proba::{Uniform01, Marginal};
    ///
    /// let u = Uniform01;
    /// assert_eq!(u.pdf(-1.0), 0.0);
    /// assert_eq!(u.pdf(0.5), 1.0);
    /// assert_eq!(u.pdf(1.5), 0.0);
    /// ```
    fn pdf(&self, x: f64) -> f64;
    /// The natural logarithm of the PDF at this point
    fn ln_pdf(&self, x: f64) -> f64 {
        self.pdf(x).ln()
    }
    /// The cumulative density function, defined as the integral:
    ///
    ///    $$cdf(x) = \int_{-\infty}^x pdf(u) du$$
    ///
    /// Just like the `pdf`, it should be defined for every float. It is the responsibility of
    /// the implementor to return sensible values outside the domain of definition of the
    /// distribution, using the convention that $p.cdf(x) = P(p < x)$
    ///
    /// # Arguments
    /// * x - Value in R for which the CDF is required
    ///
    /// # Returns
    /// * p - Value in [0; 1] such that $P(D < x) = p$, where $D$ follows that marginal distribution
    ///
    /// # Examples
    /// ```
    /// use qrust::proba::{Uniform01, Marginal};
    ///
    /// let u = Uniform01;
    /// assert_eq!(u.cdf(-1.0), 0.0);
    /// assert_eq!(u.cdf(0.0), 0.0);
    /// assert_eq!(u.cdf(0.5), 0.5);
    /// assert_eq!(u.cdf(1.0), 1.0);
    /// assert_eq!(u.cdf(1.5), 1.0);
    /// ```
    ///
    /// ```
    /// use qrust::proba::{StandardNormal, Marginal, Proba};
    ///
    /// let n = StandardNormal;
    /// assert_eq!(n.cdf(0.0), 0.5);
    /// // The standard normal is a symmetrical distribution
    /// assert!((n.cdf(-1.0).as_f64() + n.cdf(1.0).as_f64() - 1.0).abs() < 1e-15);
    /// // The quantile of the distribution is the inverse of the CDF
    /// let x = n.quantile(Proba::from_f64(0.35));
    /// assert!((n.cdf(x).as_f64() - 0.35).abs() < 1e-15);
    /// ```
    fn cdf(&self, x: f64) -> Proba;
    /// Complement of the CDF at the provided point
    ///
    /// cdf_complement(x) = 1.0 - cdf(x) = \int_{x}^{\infty} pdf(u) du
    ///
    /// # Arguments
    /// * x - Value in R for which the CDF complement is required
    ///
    /// # Returns
    /// * p - Value in [0; 1] such that $P(D >= x) = p$, where $D$ follows that marginal
    ///
    /// # Examples
    /// ```
    /// use qrust::proba::{Uniform01, Marginal};
    ///
    /// let u = Uniform01;
    /// assert_eq!(u.cdf_complement(-1.0), 1.0);
    /// assert_eq!(u.cdf_complement(0.0), 1.0);
    /// assert_eq!(u.cdf_complement(0.5), 0.5);
    /// assert_eq!(u.cdf_complement(1.0), 0.0);
    /// assert_eq!(u.cdf_complement(1.5), 0.0);
    /// ```
    ///
    /// ```
    /// use qrust::proba::{StandardNormal, Marginal, Proba};
    ///
    /// let n = StandardNormal;
    /// assert_eq!(n.cdf_complement(0.0), 0.5);
    /// // The standard normal is a symmetrical distribution
    /// assert!((n.cdf_complement(-1.0).as_f64() + n.cdf_complement(1.0).as_f64() - 1.0).abs() < 1e-15);
    /// // The quantile of the distribution is the inverse of the CDF
    /// let x = n.quantile(Proba::from_f64(0.65));
    /// assert!((n.cdf_complement(x).as_f64() - (1.0 - 0.65)).abs() < 1e-15);
    /// ```
    fn cdf_complement(&self, x: f64) -> Proba {
        self.cdf(x).complement()
    }
    /// The quantile is essentially the inverse of the CDF - it is defined as the number `x` such
    /// that the probability of the distribution being less than `x` is equal to the input `p`.
    ///
    /// In case where multiple such values are defined, the implementor is free to choose whichever
    /// value makes more sense.
    ///
    /// # Arguments
    /// * p - The probability in [0; 1] for which the quantile is required
    ///
    /// # Returns
    /// * x - Value in R such that $P(D < x) = p$, where $D$ follows that marginal distribution
    ///
    /// # Examples
    /// ```
    /// use qrust::proba::{StandardNormal, Marginal, Proba};
    ///
    /// let n = StandardNormal;
    /// // The median is the 50th percentile
    /// assert_eq!(n.quantile(Proba::HALF), n.median());
    /// // The 2.5th percentile is approximatively -1.96 (to two decimal places)
    /// assert!((n.quantile(Proba::from_f64(0.025)) + 1.96).abs() < 1e-2);
    /// ```
    fn quantile(&self, p: Proba) -> f64;
    fn quantile_f64(&self, f: f64) -> Result<f64, <Proba as TryFrom<f64>>::Error> {
        Proba::try_from(f).map(|p| self.quantile(p))
    }
    /// The median is simply an alias for the 50th percentile.
    ///
    /// # Example
    /// ```
    /// use qrust::proba::{Uniform01, StandardNormal, Marginal, Proba};
    ///
    /// let u = Uniform01;
    /// assert_eq!(u.quantile(Proba::HALF), u.median());
    /// assert_eq!(u.median(), 0.5);
    /// let n = StandardNormal;
    /// assert_eq!(n.median(), 0.0);
    /// ```
    fn median(&self) -> f64 {
        self.quantile(Proba::HALF)
    }
    /// Shifts the distribution by a constant amount.
    ///
    /// In other words, if `x` is the marginal of variable $X$, `x.shift(s)` is the marginal of
    /// the variable $X + s$
    ///
    /// # Arguments
    ///  * shift - the absolute shift of the distribution. May be negative or even zero.
    ///
    /// # Returns
    ///  * shifted - The shifted distribution. Note that this will **consume** `self`. If the
    ///              original (unshifted) distribution is still required, it should be cloned by
    ///              the client prior to calling this.
    ///
    /// # Examples
    /// ```
    /// use qrust::proba::{StandardNormal, Marginal};
    ///
    /// let n = StandardNormal;
    /// assert_eq!(n.mean(), 0.0);
    /// assert_eq!(n.variance(), 1.0);
    /// let shifted = n.shift(1.0);
    /// // Expected value is linear in the shift
    /// assert_eq!(shifted.mean(), 1.0);
    /// // Variance is unaffected by shifts
    /// assert_eq!(shifted.variance(), 1.0);
    /// // Shifted variables can be shifted as well
    /// let shifted_again = shifted.shift(3.0);
    /// assert_eq!(shifted_again.mean(), 4.0);
    /// ```
    ///
    /// All the qrustities, cdf, pdf, ... are adapted to the shifted value
    /// ```
    /// use qrust::proba::{Uniform01, Marginal};
    ///
    /// let u = Uniform01;
    /// assert_eq!(u.mean(), 0.5);
    /// assert_eq!(u.cdf(-0.5), 0.0);
    /// assert_eq!(u.cdf(0.5), 0.5);
    /// assert_eq!(u.cdf(1.5), 1.0);
    ///
    /// let s = u.shift(2.3);
    /// assert_eq!(s.mean(), 2.8);
    /// assert_eq!(s.cdf(1.5), 0.0);
    /// assert_eq!(s.cdf(2.8), 0.5);
    /// assert_eq!(s.cdf(3.5), 1.0);
    /// ```
    fn shift(self, shift: f64) -> Shifted<Self>
        where
            Self: Sized,
    {
        Shifted { base: self, shift }
    }
    /// Scales a distribution by a fixed real factor, i.e. if `x` is the marginal of some variable
    /// $X$, then `x.scale(s)` is the marginal of variable $s \cdot X$
    ///
    /// # Arguments
    ///  * scale - the real scale used to scale the variable. Note that this can be negative, or
    ///            even 0.0
    ///
    /// # Returns
    ///  * scaled - the marginal of the scaled variable. Note that this **consumes** `self`, so if
    ///             the (unscaled) marginal is still required, it should be cloned by the client
    ///             prior to calling this.
    ///
    /// # Examples
    /// ```
    /// use qrust::proba::{StandardNormal, Marginal};
    ///
    /// let n = StandardNormal;
    /// assert_eq!(n.mean(), 0.0);
    /// assert_eq!(n.variance(), 1.0);
    ///
    /// let s = n.scale(2.0);
    /// assert_eq!(s.mean(), 0.0);
    /// // Variance is quadratically affected by the scaling
    /// assert_eq!(s.variance(), 4.0);
    /// // Standard deviation is linearly affected by the scaling
    /// assert_eq!(s.std(), 2.0);
    ///
    /// // Scales can be negative
    /// let i = s.scale(-3.0);
    /// // Variance and stdev are always scaled by the *absolute value* of the scale
    /// assert_eq!(i.variance(), 36.0);
    /// assert_eq!(i.std(), 6.0);
    /// ```
    fn scale(self, scale: f64) -> Scaled<Self>
        where
            Self: Sized,
    {
        Scaled { base: self, scale }
    }
}
/// A shifted 1d distribution of probability.
/// If X ~ D, then this is the law of X + shift.
///
/// This is mainly constructed using the `Marginal::shift` method
///
/// # Examples
/// ```
/// use qrust::proba::*;
///
/// let u = uniform(-1.0, 1.0);
/// // Shift U([-1; 1]) -> U([0.0, 2.0])
/// let v = u.shift(1.0);
/// assert_eq!(v.mean(), 1.0);
/// assert_eq!(v.cdf(0.0), 0.0);
/// assert_eq!(v.cdf(1.0), 0.5);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Shifted<D> {
    base: D,
    shift: f64,
}
/// Marginal distribution for the shifted distribution
impl<D: Marginal> Marginal for Shifted<D> {
    fn mean(&self) -> f64 {
        self.base.mean() + self.shift
    }
    fn variance(&self) -> f64 {
        self.base.variance()
    }
    fn std(&self) -> f64 {
        self.base.std()
    }
    fn pdf(&self, x: f64) -> f64 {
        self.base.pdf(x - self.shift)
    }
    fn ln_pdf(&self, x: f64) -> f64 {
        self.base.ln_pdf(x - self.shift)
    }
    fn cdf(&self, x: f64) -> Proba {
        self.base.cdf(x - self.shift)
    }
    fn quantile(&self, p: Proba) -> f64 {
        self.base.quantile(p) + self.shift
    }
    fn median(&self) -> f64 {
        self.base.median() + self.shift
    }
}
/// A scaled 1D distribution of probability.
/// If X ~ D, then this is the law of scale * X
///
/// This is mainly constructed using the `Marginal::scale` method to scale an existing
/// distribution.
///
/// # Examples
/// ```
/// use qrust::proba::*;
///
/// let n = normal(5., 1.);
/// let s = n.scale(2.0);
/// assert_eq!(s.mean(), 10.0);
/// assert_eq!(s.std(), 2.0);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Scaled<D> {
    base: D,
    scale: f64,
}
impl<D: Marginal> Marginal for Scaled<D> {
    fn mean(&self) -> f64 {
        self.base.mean() * self.scale
    }
    fn variance(&self) -> f64 {
        self.base.variance() * self.scale * self.scale
    }
    fn std(&self) -> f64 {
        self.base.std() * self.scale.abs()
    }
    fn pdf(&self, x: f64) -> f64 {
        self.base.pdf(x / self.scale)
    }
    fn ln_pdf(&self, x: f64) -> f64 {
        self.base.ln_pdf(x / self.scale)
    }
    fn cdf(&self, x: f64) -> Proba {
        self.base.cdf(x / self.scale)
    }
    fn quantile(&self, p: Proba) -> f64 {
        self.base.quantile(p) * self.scale
    }
    fn median(&self) -> f64 {
        self.base.median() * self.scale
    }
}
/// A standard continuous uniform [0; 1] random variable
///
/// This is a simple size-0 object which can be cheaply passed around and copied, and serves as the
/// basis for all uniforms variables via scaling and shifting.
///
/// # Examples
/// ```
/// use qrust::proba::{Uniform01, Marginal};
///
/// let u = Uniform01;
/// assert_eq!(u.mean(), 0.5);
/// assert_eq!(u.variance(), 1. / 12.);
/// assert_eq!(u.cdf(-1.0), 0.0);
/// assert_eq!(u.cdf(0.0), 0.0);
/// assert_eq!(u.cdf(0.25), 0.25);
/// assert_eq!(u.cdf(0.5), 0.5);
/// assert_eq!(u.cdf(0.75), 0.75);
/// assert_eq!(u.cdf(1.0), 1.0);
/// assert_eq!(u.cdf(2.5), 1.0);
/// ```
///
/// # See also
///  * `Uniform` - A scaled/shifted version of the uniform distribution
///  * `uniform` - creates a uniform variable on an arbitrary interval [a; b] by shifting and
///                scaling a `Uniform01` standard variable.
#[derive(Clone, Copy, Debug)]
pub struct Uniform01;
impl Marginal for Uniform01 {
    fn mean(&self) -> f64 {
        0.5
    }
    fn variance(&self) -> f64 {
        1. / 12.
    }
    fn std(&self) -> f64 {
        1. / (2. * 3.0_f64.sqrt())
    }
    fn pdf(&self, x: f64) -> f64 {
        if x >= 0.0 && x <= 1.0 {
            1.0
        } else {
            0.0
        }
    }
    fn cdf(&self, x: f64) -> Proba {
        if x < 0.0 {
            Proba::ZERO
        } else if x > 1.0 {
            Proba::ONE
        } else {
            if let Ok(p) = Proba::try_from(x) {
                p
            } else {
                unreachable!("Invalid probability")
            }
        }
    }
    fn quantile(&self, p: Proba) -> f64 {
        p.into()
    }
    fn median(&self) -> f64 {
        0.5
    }
}
pub type Uniform = Shifted<Scaled<Uniform01>>;
impl Uniform {
    pub fn new(low: f64, up: f64) -> Uniform {
        Shifted {
            base: Scaled {
                base: Uniform01,
                scale: (up - low),
            },
            shift: low,
        }
    }
}
/// A standard normal distribution N(0, 1)
///
/// This is a size-0 struct which can be cheaply cloned and copied and serves as the basis for
/// all the normal distributions by scaling and shifting it
///
/// # Examples
/// ```
/// use qrust::proba::{StandardNormal, Marginal};
///
/// let n = StandardNormal;
/// assert_eq!(n.mean(), 0.0);
/// assert_eq!(n.std(), 1.0);
/// ```
///
/// # See also
///  * `normal` - creates a N(mu, sigma) normal variable by shifting and scaling a standard normal
#[derive(Clone, Copy, Debug)]
pub struct StandardNormal;
impl Marginal for StandardNormal {
    fn mean(&self) -> f64 {
        0.0
    }
    fn variance(&self) -> f64 {
        1.0
    }
    fn std(&self) -> f64 {
        1.0
    }
    /// Computes the Probability Density Function of a standard normal distribution
    ///
    /// # Arguments
    ///  * x - the value where the function is evaluated
    ///
    /// # Returns
    ///  * pdf - the PDF at this point, i.e. d_x P(N < x) where N is a standard normal distribution
    fn pdf(&self, x: f64) -> f64 {
        (-(x * x) / 2.0).exp() * FRAC_1_SQRT_2_PI
    }
    fn ln_pdf(&self, x: f64) -> f64 {
        -0.5 * (x * x + LN_2 + LN_PI)
    }
    /// Computes the Cumulative Density Function of a standard normal distribution
    ///
    /// # Arguments
    ///  * x - the value where the function is evaluated
    ///
    /// # Returns
    ///  * cdf - the CDF at this point, i.e. P(N < x) where N is a standard normal distribution
    fn cdf(&self, x: f64) -> Proba {
        Proba::try_from(0.5 * (1. + (x * FRAC_1_SQRT_2).error())).expect("unreachable")
    }
    fn quantile(&self, p: Proba) -> f64 {
        SQRT_2 * (2. * p.as_f64() - 1.).inv_error()
    }
    fn median(&self) -> f64 {
        0.0
    }
}
/// A normal variable
pub type Normal = Shifted<Scaled<StandardNormal>>;
impl Normal {
    pub fn new(mu: f64, sigma: f64) -> Normal {
        Shifted {
            base: Scaled {
                base: StandardNormal,
                scale: sigma,
            },
            shift: mu,
        }
    }
}
/// A log-normal distribution
///
/// If X follows a normal distribution of mean `mu` and stdev `sigma`, then exp{X} follows the
/// _log-normal_ distribution of parameters `mu` and `sigma`.
///
/// # See also
///  * `LogNormal::fit` - fits a log-normal distribution to the provided mean and variance
///  * `lognormal` - creates a new `LogNormal` distribution using the provided parameters
#[derive(Clone, Copy, Debug)]
pub struct LogNormal(Normal);
impl LogNormal {
    /// Creates a new `LogNormal` with the provided mean and variance.
    ///
    /// # Arguments
    ///  * mean - the desired mean of the resulting log-normal variable
    ///  * var - the desired variance of the resulting log-normal variable. This MUST NOT be
    ///          negative, although it can be zero.
    ///
    /// # Returns
    ///  * ln - the log-normal distribution with provided mean and variance
    ///
    /// # Examples
    /// ```
    /// use qrust::proba::{LogNormal, Marginal};
    ///
    /// let ln = LogNormal::fit(1.0, 1.0);
    /// // May have some numerical imprecision from the fit
    /// assert!((ln.mean() - 1.0).abs() < 1e-15);
    /// assert!((ln.variance() - 1.0).abs() < 1e-15);
    /// ```
    pub fn fit(mean: f64, var: f64) -> LogNormal {
        if var < 0.0 {
            panic!("Variance cannot be negative - got {}", var);
        }
        let sigma2 = softplus(var.ln() - 2. * mean.ln());
        let mu = mean.ln() - sigma2 / 2.;
        LogNormal::new(mu, sigma2.sqrt())
    }
    pub fn new(mu: f64, sigma: f64) -> Self {
        LogNormal(normal(mu, sigma))
    }
    #[inline]
    fn mu(&self) -> f64 {
        self.0.mean()
    }
    #[inline]
    fn sigma(&self) -> f64 {
        self.0.std()
    }
    #[inline]
    fn sigma2(&self) -> f64 {
        let s = self.sigma();
        s * s
    }
}
impl Marginal for LogNormal {
    /// Mean of the distribution
    fn mean(&self) -> f64 {
        (self.mu() + self.sigma2() / 2.).exp()
    }
    /// Variance of the log-normal distribution
    fn variance(&self) -> f64 {
        let sigma2 = self.sigma2();
        sigma2.exp_m1() * (2. * self.mu() + sigma2).exp()
    }
    /// Probability density function
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let d = x.ln() - self.mu();
        (-(d * d) / (2. * self.sigma2())).exp() * FRAC_1_SQRT_2_PI / (x * self.sigma())
    }
    /// Log of the PDF
    fn ln_pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let d = x.ln() - self.mu();
        -(d * d) / (2. * self.sigma2()) - x.ln() - self.sigma().ln() - 0.5 * LN_2 - 0.5 * PI.ln()
    }
    /// Cumulative density function
    fn cdf(&self, x: f64) -> Proba {
        if x <= 0.0 {
            return Proba::ZERO;
        }
        Proba::from_f64((((x.ln() - self.mu()) / (SQRT_2 * self.sigma())).error() + 1.) / 2.)
    }
    /// quantile of the distribution
    fn quantile(&self, q: Proba) -> f64 {
        (self.mu() + SQRT_2 * self.sigma() * (2. * q.as_f64() - 1.).inv_error()).exp()
    }
    /// Median of the distribution
    fn median(&self) -> f64 {
        self.mu().exp()
    }
}
/// Creates a new normally distributed distribution with the provided parameters
///
/// The returned variable follows the distribution N(mu, sigma)
///
/// # Arguments
///  * mu - the required mean for the normal distribution
///  * sigma - the standard deviation for the normal distribution
///
/// # Returns
///  * n - the marginal distribution  for the normal distribution of the given parameters
///
/// # Examples
/// ```
/// use qrust::proba::{normal, Marginal, StandardNormal};
///
/// let n1 = normal(1.0, 2.0);
/// // This should be equivalent to 2 * N + 1.0 where N is a standard normal variable
/// let n2 = StandardNormal.scale(2.0).shift(1.0);
/// assert_eq!(n1.mean(), n2.mean());
/// assert_eq!(n1.variance(), n2.variance());
/// assert_eq!(n1.pdf(0.5), n2.pdf(0.5));
/// assert_eq!(n1.cdf(1.5), n2.cdf(1.5));
/// ```
pub fn normal(mu: f64, sigma: f64) -> Normal {
    Normal::new(mu, sigma)
}
/// Creates a new uniform distribution between a and b
///
/// The returned variable follows distribution $U(a, b)$
///
/// # Arguments
///  * a - the lower bound of the domain of definition of the returned distribution
///  * b - the upper bound of the domain of definition of the returned distribution
///
/// # Returns
///  * u - a uniform marginal distribution on [a; b]
///
/// # Examples
/// ```
/// use qrust::proba::{uniform, Marginal};
///
/// let u = uniform(10., 20.);
/// assert_eq!(u.mean(), 15.0);
/// assert_eq!(u.cdf(10.0), 0.0);
/// assert_eq!(u.cdf(17.0), 0.7);
/// assert_eq!(u.cdf(20.0), 1.0);
/// assert_eq!(u.quantile_f64(0.25).unwrap(), 12.5);
/// ```
pub fn uniform(a: f64, b: f64) -> impl Marginal {
    Uniform::new(a, b)
}
/// Creates a log-normal marginal with the provided parameters
///
/// # Arguments
///  * mu - the `mu` parameter, i.e. the mean of the normal variable N such that this follows
///         the same distribution as exp{N}. Note that this is _not_ the mean.
///  * sigma - the `sigma` parameter, i.e. the stdev of the normal variable N such that this
///            follows the same distribution as exp{N}. Note that this is _not_ the stdev of the
///            variable itself
///
/// # Returns
///  * ln - the produced LogNormal marginal distribution, with parameters `mu` and `sigma`.
///
/// # See also
///  * `LogNormal::fit` can be used to fit a log-normal to a desired mean and variance.
///
/// # Examples
/// ```
/// use qrust::proba::{lognormal, Marginal};
///
/// let ln = lognormal(1.0, 2.0);
/// // Mean of the log-normal is exp(mu + sigma^2/2)
/// assert_eq!(ln.mean(), (1.0_f64 + 2.0_f64).exp());
/// // Median is exp(mu)
/// assert_eq!(ln.median(), 1.0_f64.exp());
/// ```
pub fn lognormal(mu: f64, sigma: f64) -> LogNormal {
    LogNormal::new(mu, sigma)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaled_normal() {
        let n = normal(0.4, 1.2);
        assert_eq!(n.std(), 1.2);
        assert_eq!(n.mean(), 0.4);
    }
    #[test]
    fn fit_lognormal() {
        let ln = LogNormal::fit(0.5, 1.2);

        assert!(
            (ln.mean() - 0.5).abs() < 1e-9,
            "Mean was: {} (expected 0.5) ({:?})",
            ln.mean(),
            ln
        );
        assert!(
            (ln.variance() - 1.2).abs() < 1e-9,
            "Variance was: {} (expected 1.2) ({:?})",
            ln.variance(),
            ln
        );
    }
    #[test]
    fn fit_lognormal_big() {
        let ln = LogNormal::fit(8310.2, 100.0);

        assert!(
            (ln.mean() - 8310.2).abs() < 1e-9,
            "Mean was: {} (expected 0.5) ({:?})",
            ln.mean(),
            ln
        );
        assert!(
            (ln.variance() - 100.0).abs() < 1e-9,
            "Variance was: {} (expected 1.2) ({:?})",
            ln.variance(),
            ln
        );
    }
    #[test]
    fn lognormal_median() {
        let ln = LogNormal::new(0.3,  0.8);
        assert_eq!(ln.cdf(0.3_f64.exp()), 0.5);
        assert_eq!(ln.median(), 0.3_f64.exp());
    }
}
