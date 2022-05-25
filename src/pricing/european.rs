//! Pricing of European options with closed-form formulas
//!
//! This contains a set of submodules which implement the various models.
//!
//! As of May 2022, this only contains the Black-Scholes model (for equity) and the
//! Black model (for commodities/FX/...). Note that both of these are extremely
//! similar, and as a consequence, the APIs are essentially interchangeable, when
//! correcting by the correct discount factors and setting dividend rate to 0.0
use crate::proba::{Marginal, StandardNormal};

/// The type of optionality of a vanilla option
#[derive(Clone, Copy, Debug)]
pub enum CallPut {
    Call,
    Put,
}
/// The `d1` term in the usual BS equations
fn d1(f: f64, k: f64, r: f64, q: f64, sigma: f64, ttm: f64) -> f64 {
    ((f / k).ln() + (r - q + sigma * sigma / 2.) * ttm) / (sigma * ttm.sqrt())
}

/// The greeks for an option
pub struct Greeks {
    pv: f64,
    delta: f64,
    vega: f64,
    theta: f64,
    rho: f64,
    gamma: f64,
    vanna: f64,
    vomma: f64,
}
impl Greeks {
    /// The Present-Value of the option
    pub fn pv(&self) -> f64 {
        self.pv
    }
    /// The delta (first derivative w/r to price) of the option
    pub fn delta(&self) -> f64 {
        self.delta
    }
    /// The vega (first derivative w/r to vol) of the option
    pub fn vega(&self) -> f64 {
        self.vega
    }
    /// The theta (first derivative w/r to time) of the option
    pub fn theta(&self) -> f64 {
        self.theta
    }
    /// The rho (first derivative w/r to rates) of the option
    pub fn rho(&self) -> f64 {
        self.rho
    }
    /// The gamma (second derivative w/r to price twice) of the option
    pub fn gamma(&self) -> f64 {
        self.gamma
    }
    /// The vanna (second derivative w/r to price and vol) of the option
    pub fn vanna(&self) -> f64 {
        self.vanna
    }
    /// The vomma (second derivative w/r to vol twice) of the option
    pub fn vomma(&self) -> f64 {
        self.vomma
    }
}

/// Returns the delta and 'delta-bar' for a call/put, in the Black as well as the
/// Black & Scholes model
fn delta_terms(d1: f64, d2: f64, exp_minus_qt: f64, exp_minus_rt: f64, opt: CallPut) -> (f64, f64) {
    match opt {
        CallPut::Call => (
            exp_minus_qt * StandardNormal.cdf(d1).as_f64(),
            exp_minus_rt * StandardNormal.cdf(d2).as_f64(),
        ),
        CallPut::Put => (
            -exp_minus_qt * StandardNormal.cdf(-d1).as_f64(),
            -exp_minus_rt * StandardNormal.cdf(-d2).as_f64(),
        ),
    }
}
/// The Back-Scholes model
pub mod blackschole {
    use super::*;
    use crate::optim::{halley, newton_raphson, RootFinding1d};
    /// Computes the greeks of an European option using the Black-Scholes model
    ///
    /// # Arguments
    ///  * `s` - the spot value of the stock
    ///  * `k` - the strike of the option
    ///  * `r` - the interest rate, per year
    ///  * `q` - the dividend rate, per year
    ///  * `sigma` - the annualised volatility of the option
    ///  * `ttm` - the time-to-maturity of the option, expressed in years
    ///  * `opt` - whether the option is a call or a put
    ///
    /// # Returns
    /// The greeks for the described option
    pub fn greeks(s: f64, k: f64, r: f64, q: f64, sigma: f64, ttm: f64, opt: CallPut) -> Greeks {
        let sqrt_ttm = ttm.sqrt();
        let d1 = d1(s, k, r, q, sigma, ttm);
        let d2 = d1 - sigma * sqrt_ttm;
        let exp_minus_qt = (-q * ttm).exp();
        let exp_minus_rt = (-r * ttm).exp();
        let phi_d1 = StandardNormal.pdf(d1);
        let vega = s * exp_minus_qt * phi_d1 * sqrt_ttm;
        let gamma = exp_minus_qt * phi_d1 / (s * sigma * sqrt_ttm);
        let vanna = vega / s * (1. - d1 / (sigma * sqrt_ttm));
        let vomma = vega * d1 * d2 / sigma;
        let (delta, delta_bar) = delta_terms(d1, d2, exp_minus_qt, exp_minus_rt, opt);
        let pv = s * delta - k * delta_bar;
        let rho = k * ttm * delta_bar;
        let theta = -exp_minus_qt * s * phi_d1 * sigma / (2. * sqrt_ttm) - r * k * delta_bar
            + q * s * delta;
        Greeks {
            pv,
            delta,
            vega,
            theta,
            rho,
            gamma,
            vanna,
            vomma,
        }
    }
    /// Computes the implied volatility for the observed option price, using a simple
    /// Halley method
    ///
    /// # Arguments
    ///  * `pv` - the observed option price
    ///  * `s` - the current spot price of the underlying
    ///  * `k` - the strike of the option
    ///  * `r` - the interest-rate, per year
    ///  * `q` - the dividend rate, per year
    ///  * `guess` - the initial guess for the volatility, annualised
    ///  * `ttm` - the time-to-maturity of the option, in years
    ///  * `opt` - whether the option is a call or a put
    ///  * `max_iter` - the maximum number of iterations allowed
    ///  * `tol` - the numerical tolerance: if the derived option price is within `tol`
    ///          of the observed price `obs`, then convergence is declared.
    ///
    /// # Returns
    ///  * `roots` - the root found, if any. Callers can call the `success` method of the
    ///            returned struct to know whether the root-finding succeeded
    pub fn implied(
        pv: f64,
        s: f64,
        k: f64,
        r: f64,
        q: f64,
        guess: f64,
        ttm: f64,
        opt: CallPut,
        max_iter: usize,
        tol: f64,
    ) -> RootFinding1d {
        let res = halley(
            |sigma| {
                let g = greeks(s, k, r, q, sigma, ttm, opt);
                (g.pv - pv, g.vega, g.vomma)
            },
            guess,
            max_iter,
            tol,
        );
        if res.success() {
            return res;
        }
        newton_raphson(
            |sigma| {
                let g = greeks(s, k, r, q, sigma, ttm, opt);
                (g.pv - pv, g.vega)
            },
            guess,
            max_iter,
            tol,
        )
    }
    #[cfg(test)]
    mod tests {
        use super::{greeks, implied, CallPut};
        use crate::testing::assert_allclose;

        /// Tests the call-put parity
        #[test]
        fn test_parity() {
            let strike = 100.0f64;
            let r = 0.01f64;
            let q = 0.005f64;
            let ttm = 0.3f64;
            let sigma = 0.25f64;
            let discount = (-r * ttm).exp();
            for moneyness in vec![-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0] {
                let spot = strike + moneyness;
                let forward = spot * ((r - q) * ttm).exp();
                let call = greeks(spot, strike, r, q, sigma, ttm, CallPut::Call);
                let put = greeks(spot, strike, r, q, sigma, ttm, CallPut::Put);
                assert_allclose(&(call.pv - put.pv), &(discount * (forward - strike)), 1e-9);
            }
        }
        #[test]
        fn test_implied() {
            let spot = 108.27f64;
            let strike = 100.0f64;
            let r = 0.01f64;
            let q = 0.005f64;
            let ttm = 0.3f64;
            let sigma = 0.25f64;
            let call = greeks(spot, strike, r, q, sigma, ttm, CallPut::Call);
            let vol = implied(
                call.pv,
                spot,
                strike,
                r,
                q,
                0.5,
                ttm,
                CallPut::Call,
                100,
                1e-12,
            );
            assert!(vol.success());
            // The difference of PV must be less than 1e-12: the vols should be even closer than that
            assert!((sigma - vol.value()).abs() < 1e-12);
        }
    }
}
/// The Black76 model
pub mod black {
    use super::*;
    use crate::optim::{halley, newton_raphson, RootFinding1d};

    /// Computes the greeks of an European option using the Black model
    ///
    /// # Arguments
    ///  * `f` - the current value of the forward
    ///  * `k` - the strike of the option
    ///  * `r` - the interest rate, per year
    ///  * `sigma` - the annualised volatility of the option
    ///  * `ttm` - the time-to-maturity of the option, expressed in years
    ///  * `opt` - whether the option is a call or a put
    ///
    /// # Returns
    /// The greeks for the described option
    pub fn greeks(f: f64, k: f64, r: f64, sigma: f64, ttm: f64, opt: CallPut) -> Greeks {
        let sqrt_ttm = ttm.sqrt();
        let d1 = d1(f, k, 0.0, 0.0, sigma, ttm);
        let d2 = d1 - sigma * sqrt_ttm;
        let exp_minus_rt = (-r * ttm).exp();
        let phi_d1 = StandardNormal.pdf(d1);
        let vega = f * exp_minus_rt * phi_d1 * sqrt_ttm;
        let gamma = exp_minus_rt * phi_d1 / (f * sigma * sqrt_ttm);
        let vanna = vega / f * (1. - d1 / (sigma * sqrt_ttm));
        let vomma = vega * d1 * d2 / sigma;
        let (delta, delta_bar) = delta_terms(d1, d2, exp_minus_rt, exp_minus_rt, opt);
        let pv = f * delta - k * delta_bar;
        let rho = k * ttm * delta_bar;
        let theta = -exp_minus_rt * f * phi_d1 * sigma / (2. * sqrt_ttm) - r * k * delta_bar
            + r * f * delta;
        Greeks {
            pv,
            delta,
            vega,
            theta,
            rho,
            gamma,
            vanna,
            vomma,
        }
    }
    /// Computes the implied volatility for the observed option price, using a simple
    /// Halley method
    ///
    /// # Arguments
    ///  * `pv` - the observed option price
    ///  * `f` - the value of the forward corresponding to the maturity of the option
    ///  * `k` - the strike of the option
    ///  * `r` - the interest-rate, per year
    ///  * `guess` - the initial guess for the volatility, annualised
    ///  * `ttm` - the time-to-maturity of the option, in years
    ///  * `opt` - whether the option is a call or a put
    ///  * `max_iter` - the maximum number of iterations allowed
    ///  * `tol` - the numerical tolerance: if the derived option price is within `tol`
    ///          of the observed price `obs`, then convergence is declared.
    ///
    /// # Returns
    ///  * `roots` - the root found, if any. Callers can call the `success` method of the
    ///            returned struct to know whether the root-finding succeeded
    pub fn implied(
        pv: f64,
        f: f64,
        k: f64,
        r: f64,
        guess: f64,
        ttm: f64,
        opt: CallPut,
        max_iter: usize,
        tol: f64,
    ) -> RootFinding1d {
        let res = halley(
            |sigma| {
                let g = greeks(f, k, r, sigma, ttm, opt);
                (g.pv - pv, g.vega, g.vomma)
            },
            guess,
            max_iter,
            tol,
        );
        if res.success() {
            return res;
        }
        newton_raphson(
            |sigma| {
                let g = greeks(f, k, r, sigma, ttm, opt);
                (g.pv - pv, g.vega)
            },
            guess,
            max_iter,
            tol,
        )
    }
    #[cfg(test)]
    mod tests {
        use super::{greeks, implied, CallPut};
        use crate::testing::{assert_allclose, round_at};

        /// Tests the call-put parity
        #[test]
        fn test_parity() {
            let strike = 100.0f64;
            let r = 0.01f64;
            let ttm = 0.3f64;
            let sigma = 0.25f64;
            let discount = (-r * ttm).exp();
            for moneyness in vec![-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0] {
                let call = greeks(strike + moneyness, strike, r, sigma, ttm, CallPut::Call);
                let put = greeks(strike + moneyness, strike, r, sigma, ttm, CallPut::Put);
                assert_allclose(&(call.pv - put.pv), &(discount * moneyness), 1e-9);
            }
        }
        #[test]
        fn test_implied() {
            let forward = 108.27f64;
            let strike = 100.0f64;
            let r = 0.01f64;
            let ttm = 0.3f64;
            let sigma = 0.25f64;
            let call = greeks(forward, strike, r, sigma, ttm, CallPut::Call);
            let vol = implied(
                call.pv,
                forward,
                strike,
                r,
                0.5,
                ttm,
                CallPut::Call,
                100,
                1e-12,
            );
            assert!(vol.success());
            // The difference of PV must be less than 1e-12: the vols should be even closer than that
            assert!((sigma - vol.value()).abs() < 1e-12);
        }
        #[test]
        fn test_ig() {
            // Values plucked from the EURGBP Sep20 option on IG on 2020-08-15
            let fwd = 9053.77;
            let strike = 9200.0;
            // BoE guiding rate is at 0.1%, BCE at 0%
            let rate = 0.0;
            let ttm = 0.0872;
            let pv_call = 27.7;
            let vol = implied(
                pv_call,
                fwd,
                strike,
                rate,
                0.2,
                ttm,
                CallPut::Call,
                100,
                1e-12,
            );
            assert!(vol.success());
            assert_eq!(round_at(vol.value(), 4), 0.0749);
        }
    }
}
