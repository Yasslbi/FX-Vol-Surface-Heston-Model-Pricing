# Overview

This project builds a complete, end-to-end workflow for constructing, smoothing, and calibrating an FX volatility surface, and then using this surface to price exotic options under a stochastic-volatility model.
The idea is simple: start from real market inputs, transform them into a volatility smile, clean the smile, then calibrate a dynamic model capable of pricing any payoff — including path-dependent structures like barrier options.

The full pipeline is:

- Market inputs (FX quotes + interest rates)
We use standard FX volatility quotes: 25-delta put, ATM, and 25-delta call, across multiple maturities.
In FX markets, volatility is quoted in delta space, so the first step is to convert these quotes into strikes.

- Delta → Strike conversion (Garman–Kohlhagen)
FX options follow the Garman–Kohlhagen framework (the FX equivalent of Black–Scholes), where the forward price F = Se(rd − rf)T
ensures no-arbitrage between spot, interest rates, and forwards.
We invert the FX delta formula to compute the strike of each quoted option.

- Vanna-Volga smile reconstruction
A fast, trading-desk method used in FX.
Vanna-Volga reconstructs a full smile using only three market points: Put-25D, ATM, and Call-25D.
It captures skew + curvature and is extremely popular among FX traders, even though it can produce irregular shapes in the very short maturities.

- SVI calibration (slice-by-slice)
SVI (“Stochastic Volatility Inspired”) is a widely-used, arbitrage-free parametrization of volatility smiles.
We fit an SVI curve for each maturity to smooth the Vanna-Volga output and enforce a clean structure across strikes and maturities.
This step removes noise, eliminates curvature spikes, and gives a professional-grade surface.

- Surface comparison: Vanna-Volga vs SVI
We analyze the ATM term structure, risk reversals, butterflies, and full 3D surface differences.
We show that Vanna-Volga can be unstable at short maturities, whereas SVI restores consistency and no-arbitrage smoothness.

- Heston model calibration
Once we have a clean volatility surface, we calibrate a dynamic model — the Heston stochastic volatility model — to the entire surface.
This allows pricing products whose value depends on the path of volatility, not only on its instantaneous level.

- Monte Carlo simulation under Heston
We simulate full Heston paths (spot + variance) using a stable discretization scheme.
This is required for path-dependent products.

- Pricing an exotic option: Down-and-Out Put
We price a barrier option using the calibrated Heston model.
We compare its price to a vanilla European put to quantify the “discount” created by the barrier.

What this project demonstrates : 

- How FX markets quote volatility and why delta-based quoting requires numerical inversion.
- How to reconstruct a smile with Vanna-Volga, a practical method used extensively in trading floors.
- Why SVI is essential for building a clean, stable, arbitrage-free surface.
- How to calibrate Heston to the entire surface, and what each parameter means.
- How Monte Carlo simulation works when volatility is stochastic.

How to price a barrier option, and why its price is much lower than a vanilla option.


# Market Inputs & Delta→Strike Conversion

FX markets do not quote implied volatilities directly at a given strike.
Instead, traders quote volatilities at certain deltas:

- 25-delta Put
- ATM (delta-neutral strike)
- 25-delta Call

This quoting convention comes from trading practice: deltas are more stable and directly linked to hedging.
But to build a volatility smile, we need volatility as a function of strike, not delta.
So the first technical challenge in FX is:

Convert delta-quoted vols into strike-quoted vols.

This section explains how.

# The FX Forward (Garman–Kohlhagen)

In equities you discount using one interest rate.
In FX, two rates always matter:

- rd = domestic interest rate  
- rf = foreign interest rate

For EUR/USD:

- EUR/USD = number of USD per 1 EUR
- Domestic currency = USD
- Foreign currency = EUR
- rd = USD rate  
- rf = EUR rate


The FX forward respecting no-arbitrage is:

F = S * exp( (rd - rf) * T )

If EUR pays less interest than USD, then holding USD is financially more attractive.
Therefore, the EUR/USD forward must adjust so that no trader can borrow USD, lend EUR, and earn a free profit.

This condition eliminates carry arbitrage.

