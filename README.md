# Overview

This project builds a complete, end-to-end workflow for constructing, smoothing, and calibrating an FX volatility surface, and then using this surface to price exotic options under a stochastic-volatility model.
The idea is simple: start from real market inputs, transform them into a volatility smile, clean the smile, then calibrate a dynamic model capable of pricing any payoff — including path-dependent structures like barrier options.

What this project demonstrates : 

- How FX markets quote volatility and why delta-based quoting requires numerical inversion.
- How to reconstruct a smile with Vanna-Volga, a practical method used extensively in trading floors.
- Why SVI is essential for building a clean, stable, arbitrage-free surface.
- How to calibrate Heston to the entire surface, and what each parameter means.
- How Monte Carlo simulation works when volatility is stochastic.

How to price a barrier option, and why its price is much lower than a vanilla option.

# Market Inputs & Delta→Strike Conversion

Before touching deltas or strikes, we must understand what the FX forward is and why it must take a specific form under no-arbitrage.
The entire delta convention used in FX options rests on these foundations.

2.1 The FX Forward: a direct consequence of no arbitrage

Consider two risk-free investments over the same maturity T:
- investing in the domestic currency at rate 𝑟𝑑
- investing in the foreign currency at rate 𝑟𝑓,
- then converting back at maturity.

In the absence of arbitrage, both strategies must have the same domestic value at maturity.

This leads immediately to the FX forward pricing relation:
F = S * exp((rd - rf) * T)


This formula is a consistency condition imposed by the existence of risk-free borrowing and lending in both currencies.
It is the backbone of FX derivatives pricing.

Because the forward is the natural reference price in FX, the sensitivity of an option to spot — its delta — is expressed relative to that forward.
Under Garman–Kohlhagen, the delta of a call is:

Delta_call = exp(-rf * T) * N(d1)

and for a put:
Delta_put  = -exp(-rf * T) * N(-d1)

with:
d1 = ( ln(F/K) + 0.5 * vol^2 * T ) / ( vol * sqrt(T) )

This delta is fully consistent with the hedging strategy: a trader who holds one call will hedge by selling delta units of the foreign currency (discounted), regardless of the strike. Delta is therefore the native risk metric in FX.

For this reason, the market quotes volatilities by delta buckets (10D, 25D, ATM, etc.) rather than by strike.
Deltas reflect liquidity, hedging cost, and market depth far better than strikes do.

Models, however, do not operate in delta space.
A volatility smile is a function of strike, not delta.
Every interpolation method (Vanna–Volga, SVI) and every stochastic model (Heston) must be calibrated on vol(K), not vol(Δ).

Thus, the first technical step consists of solving for the strike corresponding to a given delta quote.
This is done by inverting the delta formula:

fx_delta_spot(S, K, T, rd, rf, vol) = market_delta

FX markets use a specific definition of ATM: the strike for which d1 = 0.
This yields a closed-form expression:

K_ATM = F * exp(0.5 * vol_ATM**2 * T)

After performing the delta inversion, we obtain:

- X1 = strike of the 25-delta put
- X2 = ATM-delta strike
- X3 = strike of the 25-delta call

# Vanna-Volga (FX Smile Reconstruction)

Vanna-Volga is a legacy interpolation framework that reconstructs an implied-volatility smile from the three standard FX quotes: the 25-delta put, the at-the-money volatility, and the 25-delta call. Its objective is not to model volatility dynamics, but simply to recover a smile consistent with market conventions while remaining computationally straightforward.

The method positions each strike K in log-moneyness space, a representation that reflects the natural geometry of option pricing. The relative location of K with respect to the quoted wing strikes determines a set of weights:

- z1 = ( log(X2/K) * log(X3/K) ) / ( log(X2/X1) * log(X3/X1) )
- z2 = ( log(K/X1) * log(X3/K) ) / ( log(X2/X1) * log(X3/X2) )
- z3 = ( log(K/X1) * log(K/X2) ) / ( log(X3/X1) * log(X3/X2) )


Using these coefficients, a first-order interpolation produces the linear approximation:

sigma_linear = z1 * sigma_put + z2 * sigma_ATM + z3 * sigma_call


This linear component reproduces the market skew implied by the delta convention but remains insufficient to reflect the curvature observed in FX smiles, especially in the wings. To address this, Vanna-Volga adds a correction linked to the option’s volga, the second derivative of the option price with respect to volatility. In the Black framework, volga is proportional to the product d1 * d2, which is negligible near the money and increases markedly in deep-wing regions. This behaviour aligns naturally with the shape of FX smiles, where curvature intensifies as strikes move further from the forward.

The adjustment term is computed as:

volga_corr = z1 * d1(X1)*d2(X1)*(sigma_put  - sigma_ATM)^2
           + z3 * d1(X3)*d2(X3)*(sigma_call - sigma_ATM)^2


The squared volatility deviations measure the extremity of each wing relative to the centre, while the d1*d2 factors modulate the strength of curvature at those locations. The resulting implied volatility is:

sigma_VV(K) = sigma_ATM
              + z1*(sigma_put  - sigma_ATM)
              + z3*(sigma_call - sigma_ATM)
              + volga_corr


This construction ensures exact recovery of the three market quotes, produces a skew consistent with delta-based quoting conventions, and introduces a curvature profile aligned with empirical FX behaviour. Vanna-Volga is therefore well suited as a fast, intuitive first-pass reconstruction of the smile. Its limitations are equally well known: the method does not control arbitrage and may generate excessive convexity in short-dated slices.

For this reason, Vanna-Volga serves in this project as an initial layer, capturing market microstructure and the qualitative shape of the smile. It is subsequently refined by an arbitrage-free parameterisation (SVI), which imposes structural consistency across strikes while preserving the essential features extracted by Vanna-Volga.

# The SVI Parameterization

Once the smile has been reconstructed from market quotes using Vanna-Volga, the next step is to express it in a form that is smooth across strikes, arbitrage-free, and suitable for dynamic model calibration. The industry standard for this purpose is the Stochastic Volatility Inspired (SVI) parameterization.
SVI does not attempt to model time evolution; its role is to provide a deterministic, structurally sound description of the smile at each maturity.

SVI operates on total implied variance, defined as:
w(k) = sigma(k)^2 * T

where sigma(k) is the implied volatility at strike K, and T the maturity.

The natural input variable is log-moneyness:
k = ln(K / F)

with F the forward price for the same maturity.

Total variance behaves much more regularly than volatility when plotted against log-moneyness, which makes SVI extremely effective for smoothing and interpolation.

The SVI functional form is:

w(k) = a + b * ( rho*(k - m) + sqrt( (k - m)^2 + sigma^2 ) )

This expression combines a linear component and a square-root component, whose interplay determines the full shape of the smile:

- the linear term ρ*(k - m) controls the overall tilt of the wings,
- the square-root term provides the curvature near the centre and stabilises the growth of variance at extreme strikes.

Unlike ad-hoc interpolations, this geometry evolves gracefully from the minimum of the smile to the far wings, producing curves that are both smooth and structurally compatible with observed market behaviour.

Each parameter governs a specific, interpretable aspect of the smile:

- a sets the vertical level of total variance. Increasing a shifts the entire curve upward.

- b controls the overall amplitude of the smile—both skew and curvature scale with b.

- rho determines the direction and steepness of the skew. Negative values produce the typical FX left-skewed structure.

- m locates the minimum of the curve along the log-moneyness axis and governs horizontal positioning.

- sigma regulates the width of the central region and the smoothness of the transition between the smile’s minimum and its wings.

This parameterization is deliberately minimal—five parameters per slice—yet it reproduces the essential features of market smiles far more robustly than raw interpolation.

# Calibration Procedure

In this project, SVI is fitted independently at each maturity.
For a given tenor, the Vanna-Volga surface provides a dense set of implied volatilities. These are converted to total variances, and the following least-squares objective is minimised:

minimize  Σ_i [ w_SVI(k_i) - w_VV(k_i) ]^2

The goal is not to mimic every fluctuation of Vanna-Volga, some of which may reflect noise or inconsistencies, especially at short maturities. Instead, SVI extracts the coherent structural backbone of the smile: a curve that is smooth, convex in the appropriate regions, and free of butterfly arbitrage.

This calibration is repeated at each maturity to produce a full SVI volatility surface—one that is:

- stable across strikes,
- well-behaved along the term structure,
- consistent with market skew and wing behaviour,
- and suitable as input for stochastic-volatility models.

# Role of SVI in the Overall Pipeline

SVI acts as the bridge between market quotes and model-based valuation:

- Vanna-Volga captures the trader’s view of the smile but may produce irregularities.
- SVI filters this information into an arbitrage-free, smooth, and model-ready representation.
- Heston then calibrates to these SVI-implied prices to produce a dynamic model consistent with the surface.

By design, SVI provides the stability and regularity that stochastic-volatility models require.
It transforms a noisy and quote-driven structure into a mathematically coherent volatility surface, ensuring that subsequent simulation and exotic-product pricing rest on a reliable foundation.

# The Heston Model

Once an arbitrage-free volatility surface has been constructed using SVI, the next step consists in embedding this surface into a dynamic model able to generate realistic paths for the underlying. Among the various stochastic-volatility approaches, the Heston model remains the industry standard for combining analytical structure, economic intuition, and the ability to reproduce key features of implied-volatility smiles.

At its core, Heston specifies a joint evolution for the asset price and its instantaneous variance. The dynamics under the risk-neutral measure are:

- dS_t = S_t * (rd - rf) * dt + S_t * sqrt(v_t) * dW1_t
- dv_t = kappa * (theta - v_t) * dt + sigma * sqrt(v_t) * dW2_t
- corr(dW1_t, dW2_t) = rho


The innovation of this model lies in assigning a stochastic, mean-reverting structure to the variance process. The square-root diffusion keeps variance non-negative and allows volatility to rise sharply, decline gradually, and settle around a long-run level. This behaviour aligns with how volatility is observed in practice: it clusters, it reacts asymmetrically to market moves, and it exhibits persistence across time.

Each parameter governs a specific, interpretable aspect of volatility:

- kappa: the speed at which variance reverts to equilibrium. A high value implies that volatility shocks dissipate quickly; a low value allows prolonged periods of elevated or subdued volatility.

- theta: the long-run variance towards which the process gravitates. It determines the baseline level of volatility in the model.

- sigma: the volatility of variance itself (“vol-of-vol”). It controls how irregular or turbulent the volatility path can become. Larger values generate more pronounced volatility clustering.

- rho: the correlation between price shocks and variance shocks. It shapes the slope of the implied-volatility smile: negative values steepen downside skew, while positive values would favour upside asymmetry.

- v0: the initial variance, consistent with the short-maturity point of the surface.

Together, these parameters allow Heston to reproduce a broad family of smile shapes and term structures while remaining sufficiently tractable for calibration and simulation.

From SVI to Heston: building a dynamic model from a static surface

SVI provides a clean, arbitrage-free snapshot of implied volatilities across strikes and maturities. To use this information for pricing path-dependent products, a model must be capable of generating trajectories for both the asset and its volatility that are coherent with the given surface.

The calibration proceeds as follows:

- Convert each SVI volatility into a Black price (Garman–Kohlhagen).
These prices form the calibration targets.

- Simulate Heston terminal prices via Monte Carlo for each maturity using a fixed set of correlated Brownian increments.
Reusing the same random numbers across strikes eliminates simulation noise from the calibration and improves stability.

- Match Heston prices to SVI prices by solving:

minimize   Σ_j Σ_i  [ Price_Heston(j,i) - Price_SVI(j,i) ]^2 / Price_SVI(j,i)^2


This is implemented as a constrained nonlinear least-squares optimisation, using a trust-region algorithm. The constraints enforce parameter positivity and realistic ranges, ensuring numerical stability and economic consistency.

The combination of an arbitrage-free surface (SVI) and a smooth Monte Carlo estimator allows the optimisation to converge quickly and robustly. Once the parameters are identified, the model reproduces the entire implied-volatility surface in expectation, while providing a time-consistent stochastic mechanism for volatility.

# Role of Heston in the overall pipeline

With calibrated parameters, the Heston model becomes a dynamic extension of the static SVI surface. It generates price and variance paths consistent with the shape of the smile at every maturity, while embedding features absent in static models:

- volatility clustering
- path-dependent effects
- asymmetric volatility shocks
- realistic joint behaviour of price and volatility

This makes Heston particularly well suited for pricing exotic products—such as barriers, cliquets, double-no-touches, and other structures whose value depends on the entire trajectory of the underlying rather than on its terminal level alone.

In this project, the model is ultimately used to price a down-and-out put, a payoff whose valuation cannot rely solely on static implied volatilities. The SVI surface provides a coherent snapshot of market conditions, and Heston supplies the evolution needed to evaluate scenarios where the barrier may or may not be breached.

Together, SVI and Heston form a consistent workflow:
SVI creates an arbitrage-free cross-section; Heston gives it life across time.

# Pricing a Down-and-Out Put Using Monte Carlo Simulation

Barrier options belong to the class of path-dependent derivatives: their value is determined not only by the terminal level of the underlying but also by the sequence of prices observed throughout the life of the product. For a vanilla European put, only S_T matters. For a down-and-out structure, the entire trajectory plays a role, because the option ceases to exist as soon as the underlying touches a predefined barrier.

A down-and-out put is a European put with a lower barrier B < S_0.
Its payoff is:

if min_{0 ≤ t ≤ T} S_t > B:
        payoff = max(K - S_T, 0)
else:
        payoff = 0

The barrier effectively suppresses all scenarios in which the underlying trades through B, even if the final level S_T would otherwise place the option in-the-money. Pricing therefore requires a model capable of evaluating not only end-point distributions but entire paths: volatility clustering, correlation effects, and local excursions of the spot all influence the likelihood of knocking out.

Why Monte Carlo?

In the presence of stochastic volatility (here, Heston dynamics), the distribution of the running minimum has no closed-form representation. This eliminates the possibility of analytical formulas and makes Monte Carlo the natural tool: it handles barrier features without approximation and preserves all interactions between spot and variance.

A Monte Carlo engine generates a large number of paths {S_t^i, v_t^i}.
For each path i, the barrier condition is monitored continuously on the simulation grid:

knocked_out_i = ( min_t S_t^i ≤ B )

If the barrier is not breached, the discounted payoff is:
exp(-rd * T) * max(K - S_T^i, 0)

Otherwise the payoff is zero.

The model price is obtained as the average across all scenarios:
Price_MC = exp(-rd * T) * mean( payoff_i )


The accuracy of the estimation depends on:

- the number of paths, which controls statistical variance,
- the time discretization, which must be fine enough to avoid missing barrier events,
- the consistency of the volatility model, since the probability of touching the barrier is highly sensitive to volatility clustering.

Using Heston dynamics rather than a lognormal diffusion changes the distribution of first-passage times significantly: volatility spikes increase the likelihood of early knock-outs, while mean reversion affects the timing of such events. Monte Carlo correctly incorporates these effects.

# Interpretation in the context of this project

Once SVI provides an arbitrage-free implied-volatility surface and Heston delivers a calibrated stochastic-volatility model, the down-and-out put becomes a natural test case for the pipeline. The payoff is sufficiently structured to reveal whether the model captures:

- the behaviour of the smile in the wings (important for barrier proximity),
- the joint evolution of spot and volatility,
- the influence of correlation on knock-out probability,
- and the correct scaling of risk across maturities.

By construction, the barrier mechanism filters the sample space: only paths remaining above B contribute to the value. In a stochastic-volatility setting this filtering is extremely sensitive to how volatility fluctuates along the path. Monte Carlo is therefore not simply a pricing tool; it is a diagnostic of whether the calibrated model produces realistic dynamics around the barrier region.

In practice, the down-and-out put highlights the difference between terminal-value consistency (matched by SVI) and path-wise realism (delivered by Heston). A model that matches the smile but fails to generate plausible volatility trajectories will misprice barrier products dramatically. The simulation performed here validates the full pipeline: SVI fixes the snapshot, Heston animates it, and the Monte Carlo engine reveals how the model behaves when timing and excursions matter as much as the final distribution.
