# Probabilistic Quadrature

A modular Python project for numerical integration with a strong focus on Bayesian Quadrature (BQ), active sampling, and reproducible comparisons against deterministic and Monte Carlo baselines.

## 1. Project Idea

The project treats integration as expectation under a measure:

- target quantity: `I_mu(f) = integral f(x) dmu(x)`
- finite interval integration is a special case of expectation under a uniform measure
- probabilistic integration is implemented with Gaussian Process based Bayesian Quadrature

The codebase is intentionally modular:

- math primitives (`Interval`, `Domain`, `Function`)
- random variables and measures
- kernels
- interpolation and derivative estimation
- integration backends (grid, Gaussian rules, Monte Carlo, Bayesian)
- factories and enums for method dispatch

## 2. Current Entry Points

There is currently no CLI script in `source/` (for example, no `source/main.py` in the current repository state).

Primary usage entry points are notebooks:

- `source/numeric_intergration.ipynb`
- `source/polynomial_interpolations.ipynb`
- `source/final_project_report.ipynb`

## 3. Full Abstraction Catalog (all classes/enums/dataclasses)

This section documents every class and enum currently present in `source/**/*.py`.

### 3.1 Core Math Layer (`source/functions`)

- `Interval` (file: `interval.py`; base: `-`): Closed interval `[left, right]`, ordering/comparison helpers, membership check, width.
- `Range` (file: `range.py`; base: `-`): Output range object for codomain validation and clamping.
- `Domain` (file: `domain.py`; base: `-`): Domain as one or multiple intervals; supports `contains`, `get_interval` (single), `get_intervals` (list).
- `Function` (file: `function.py`; base: `-`): Callable wrapper with domain/codomain checks and metadata (`name`).
- `MeasuredFunction` (file: `measured_function.py`; base: `Function`): Vectorized function tied to a measure, optional known true integral.

### 3.2 Derivative Layer (`source/functions/derivatives`)

- `DerivativeEstimator` (file: `derivative_estimator.py`; base: `ABC`): Abstract derivative interface and tangent plot helper.
- `ForwardDerivativeEstimator` (file: `forward_derivative_estimator.py`; base: `DerivativeEstimator`): Forward finite difference derivative.
- `CentralDifferenceDerivativeEstimator` (file: `central_difference_derivative_estimator.py`; base: `DerivativeEstimator`): Central finite difference derivative.

### 3.3 Polynomial Interpolation Layer (`source/functions/polynomial_interpolations`)

- `InterpolationPattern` (file: `interpolation_pattern.py`; base: `Enum`): Dispatch enum: `LAGRANGE`, `NEWTON`, `HERMIT`.
- `InterpolationPoly` (file: `interpolation_polynomial.py`; base: `ABC`): Abstract interpolation polynomial API with node/value validation.
- `LagrangeInterpolationPoly` (file: `lagrange_interpolation_polynomial.py`; base: `InterpolationPoly`): Lagrange polynomial evaluator.
- `NewtonInterpolationPoly` (file: `newton_interpolation_polynomial.py`; base: `InterpolationPoly`): Newton divided-difference polynomial evaluator.
- `HermitInterpolationPoly` (file: `hermit_interpolation_polynomial.py`; base: `InterpolationPoly`): Hermite interpolation using derivative estimates.
- `PolyInterpolationAbstractFactory` (file: `polynomial_interpolation_factory/polynomial_interpolation_abstract_factory.py`; base: `ABC`): Abstract factory for interpolation objects.
- `PolynomialInterpolationFactory` (file: `polynomial_interpolation_factory/polynomial_interpolation_factory.py`; base: `PolyInterpolationAbstractFactory`): Concrete interpolation factory.

### 3.4 Random Variable Layer (`source/random_variables`)

- `RandomVariable` (file: `random_variable.py`; base: `ABC`): Abstract RV API (`sample`, `log_prob`, `mean`, `var`).
- `Uniform` (file: `continuous_random_variables/uniform.py`; base: `RandomVariable`): 1D uniform RV.
- `ContinuousUniformBox` (file: `continuous_random_variables/uniform_box.py`; base: `RandomVariable`): Axis-aligned box uniform RV in R^d.
- `Normal` (file: `continuous_random_variables/normal.py`; base: `RandomVariable`): Diagonal Gaussian RV in R^d.

### 3.5 Measure Layer (`source/measures`)

- `Measure` (file: `measure.py`; base: `ABC`): Abstract measure API (`dim`, `sample`).
- `UniformBoxMeasure` (file: `uniform_box_measure.py`; base: `Measure`): Uniform measure over hyper-rectangle.
- `GaussianMeasure` (file: `gaussian_measure.py`; base: `Measure`): Gaussian measure with full/diag covariance sampling path.

### 3.6 Kernel Layer (`source/kernels`)

- `Kernel` (file: `kernel.py`; base: `ABC`): Abstract kernel API; includes squared-Euclidean helper.
- `RBFKernel` (file: `rbf_kernel.py`; base: `Kernel`): RBF kernel with `lengthscale`, `variance`.
- `Matern32Kernel` (file: `matern32_kernel.py`; base: `Kernel`): Matern 3/2 kernel with `lengthscale`, `variance`.

### 3.7 Integration Core Layer (`source/numeric_integration`)

- `NumericIntegral` (file: `numeric_integral.py`; base: `ABC`): Base integration abstraction with optional node validation.
- `NumericIntegrationPattern` (file: `numeric_integration_pattern.py`; base: `Enum`): Integration dispatch enum (`RECTANGLE`, `TRAPEZOID`, `SIMPSON`, `MONTE_CARLO`, `LEGENDRE`, `HERMITE`, `LAGUERRE`, `CHEBYSHEV`, `BAYESIAN`).
- `UniformGridNumericIntegral` (file: `uniform_grid_numeric_integral.py`; base: `NumericIntegral`): Base class for equally spaced grid methods.
- `WeightedNodesNumericIntegral` (file: `weighted_nodes_numeric_integral.py`; base: `NumericIntegral`): Base class for precomputed nodes/weights methods.
- `NumericIntegralAbstractFactory` (file: `numeric_integral_factory/numeric_integral_abstract_factory.py`; base: `ABC`): Abstract factory for integration methods.
- `NumericIntegralFactory` (file: `numeric_integral_factory/numeric_integral_factory.py`; base: `NumericIntegralAbstractFactory`): Concrete global integration factory.

### 3.8 Uniform Grid Deterministic Rules (`source/numeric_integration/uniform_grid_numeric_integrals`)

- `RectangleNumericIntegral` (file: `rectangle_numeric_integral.py`; base: `UniformGridNumericIntegral`): Midpoint rectangle rule.
- `TrapezoidNumericIntegral` (file: `trapezoid_numeric_integral.py`; base: `UniformGridNumericIntegral`): Trapezoidal rule.
- `SimpsonNumericIntegral` (file: `simpson_numeric_intergral.py`; base: `UniformGridNumericIntegral`): Simpson rule over each sub-interval.

### 3.9 Weighted-Node Gaussian Rules (`source/numeric_integration/weighted_nodes_numeric_integrals`)

- `AffineTransformation` (file: `affine_transformation.py`; base: `-`): Affine map from `[-1,1]` nodes/weights to target finite interval.
- `GaussLegendreIntegral` (file: `gauss_legendre_integral.py`; base: `WeightedNodesNumericIntegral`): Legendre Gaussian quadrature.
- `GaussHermiteIntegral` (file: `gauss_hermite_integral.py`; base: `WeightedNodesNumericIntegral`): Hermite Gaussian quadrature.
- `GaussLaguerreIntegral` (file: `gauss_laguerre_integral.py`; base: `WeightedNodesNumericIntegral`): Laguerre Gaussian quadrature.
- `GaussChebyshevIntegral` (file: `gauss_chebyshev_integral.py`; base: `WeightedNodesNumericIntegral`): Chebyshev Gaussian quadrature.

### 3.10 Monte Carlo Layer (`source/numeric_integration/monte_carlo`)

- `MonteCarloIntegrationStrategy` (file: `monte_carlo_stretegies.py`; base: `Enum`): MC dispatch enum: `STANDARD`, `WEIGHTED`, `RECURSIVE`.
- `MonteCarloNumericIntegral` (file: `monte_carlo_numeric_integral.py`; base: `NumericIntegral`): MC base class with sample count and stderr slot.
- `StandardMonteCarloIntegral` (file: `standard_monte_carlo_integral.py`; base: `MonteCarloNumericIntegral`): Standard sample-average estimator.
- `WeightedMonteCarloIntegral` (file: `weighted_monte_carlo_integral.py`; base: `MonteCarloNumericIntegral`): Weighted MC estimator via proposal and weight callbacks.
- `RecursiveMonteCarloIntegral` (file: `recursive_monte_carlo_integral.py`; base: `MonteCarloNumericIntegral`): Recursive 1D stratified-style estimator over interval splits.
- `MonteCarloAbstractFactory` (file: `monte_carlo_factory/monte_carlo_abstract_factory.py`; base: `ABC`): Abstract MC factory.
- `MonteCarloFactory` (file: `monte_carlo_factory/monte_carlo_factory.py`; base: `MonteCarloAbstractFactory`): Concrete MC factory.

### 3.11 Bayesian Quadrature Layer (`source/numeric_integration/bayesian_integral`)

- `BayesianQuadratureIntegral` (file: `bayesian_quadrature_integral.py`; base: `NumericIntegral`): Integration wrapper that fits `BayesianQuadratureModel` and returns posterior mean.
- `GaussianProcess` (file: `gaussian_process.py`; base: `-`): GP training/prediction helper used by BQ model.
- `ActiveBQSelector` (file: `active_bayesian_quadrature.py`; base: `-`): Active point selector using variance-reduction criterion.
- `BQConfig` (file: `bayesian_quadrature_model/bayesian_quadrature_config.py`; base: `dataclass`): BQ hyperparameters and validation (`noise`, `jitter`, MC sample counts).
- `BQDataset` (file: `bayesian_quadrature_model/bayesian_quadrature_data_set.py`; base: `dataclass`): Training data container with append/update utilities.
- `BQIntegralTermsComputer` (file: `bayesian_quadrature_model/bayesian_quadrature_integral_terms.py`; base: `-`): Computes kernel mean vector and integrated kernel variance terms.
- `BQPosteriorState` (file: `bayesian_quadrature_model/bayesian_quadrature_posterior_cache.py`; base: `dataclass`): Cached posterior state (`K_inv`, `mu_f`, `sigma_f2`).
- `BayesianQuadratureModel` (file: `bayesian_quadrature_model/bayesian_quadrature_model.py`; base: `-`): Main BQ model: fit/update, integral posterior, predictive posterior.

### 3.12 Top-Level Utility Functions

File: `source/numeric_integration/bayesian_integral/active_bayesian_quadrature.py`
- `variance_reduction_with_model`: Convenience wrapper around `ActiveBQSelector.variance_reduction`.
- `greedy_select_with_model`: Convenience wrapper around `ActiveBQSelector.greedy_select`.

File: `source/numeric_integration/bayesian_integral/bayesian_quadrature_model/utils.py`
- `ensure_2d`: Shape normalization utility for arrays.
- `_gaussian_kernel_mean_rbf`: Closed-form RBF kernel mean for Gaussian measure.
- `_gaussian_kernel_variance_rbf`: Closed-form integrated RBF kernel variance for Gaussian measure.
- `kernel_mean_vector`: Generic or closed-form kernel mean vector calculator.
- `kernel_integral_variance`: Generic or closed-form integrated kernel variance calculator.
- `gp_posterior_predictive`: GP predictive mean/variance helper.

## 4. Inheritance Hierarchies

### 4.1 Integration hierarchy

```text
NumericIntegral (ABC)
  |- UniformGridNumericIntegral
  |    |- RectangleNumericIntegral
  |    |- TrapezoidNumericIntegral
  |    |- SimpsonNumericIntegral
  |
  |- WeightedNodesNumericIntegral
  |    |- GaussLegendreIntegral
  |    |- GaussHermiteIntegral
  |    |- GaussLaguerreIntegral
  |    |- GaussChebyshevIntegral
  |
  |- MonteCarloNumericIntegral
  |    |- StandardMonteCarloIntegral
  |    |- WeightedMonteCarloIntegral
  |    |- RecursiveMonteCarloIntegral
  |
  |- BayesianQuadratureIntegral
```

### 4.2 Interpolation hierarchy

```text
InterpolationPoly (ABC)
  |- LagrangeInterpolationPoly
  |- NewtonInterpolationPoly
  |- HermitInterpolationPoly

DerivativeEstimator (ABC)
  |- ForwardDerivativeEstimator
  |- CentralDifferenceDerivativeEstimator
```

### 4.3 Measure, RV, kernel hierarchies

```text
RandomVariable (ABC)
  |- Uniform
  |- ContinuousUniformBox
  |- Normal

Measure (ABC)
  |- UniformBoxMeasure
  |- GaussianMeasure

Kernel (ABC)
  |- RBFKernel
  |- Matern32Kernel
```

### 4.4 Factory hierarchy

```text
NumericIntegralAbstractFactory (ABC)
  |- NumericIntegralFactory

MonteCarloAbstractFactory (ABC)
  |- MonteCarloFactory

PolyInterpolationAbstractFactory (ABC)
  |- PolynomialInterpolationFactory
```

## 5. Runtime Logic and Data Flow

### 5.1 Deterministic grid flow

1. Build `Function` with `Domain`.
2. Build node list for interval.
3. Use `NumericIntegralFactory` with `NumericIntegrationPattern.RECTANGLE/TRAPEZOID/SIMPSON`.
4. Call `integrate()`.

### 5.2 Gaussian quadrature flow

1. Build `Function`.
2. Select rule via `NumericIntegrationPattern.LEGENDRE/HERMITE/LAGUERRE/CHEBYSHEV`.
3. Rule class obtains nodes/weights from `numpy.polynomial.*`.
4. Weighted sum is returned by `integrate()`.

### 5.3 Monte Carlo flow

1. Build `Measure` and `MeasuredFunction`.
2. Create MC integrator via `MonteCarloFactory` or `NumericIntegralFactory` with `MONTE_CARLO`.
3. Sample from measure (optionally with explicit RV/proposal).
4. Estimate expectation (and stderr for standard/weighted variants).

### 5.4 Bayesian Quadrature flow

1. Prepare `X, y` observations of integrand.
2. Build kernel (`RBFKernel` or `Matern32Kernel`) and measure.
3. Fit `BayesianQuadratureModel` (`GaussianProcess` + integral terms + cache).
4. Query `integral_posterior()` for posterior mean and variance.

### 5.5 Active Bayesian Quadrature flow

1. Start with fitted `BayesianQuadratureModel`.
2. Build candidate pool.
3. Use `ActiveBQSelector.greedy_select()` to maximize variance reduction.
4. Evaluate selected point, update model, repeat.

### 5.6 Interpolation + BQ hybrid logic

Implemented in report experiments:

- build interpolation surrogate (`Lagrange` or `Hermite`)
- tune derivative quality for Hermite using `CentralDifferenceDerivativeEstimator`
- use surrogate as control variate and run BQ on residual

## 6. Notebooks and Their Purpose

- `source/numeric_intergration.ipynb`
  - full comparison across deterministic rules, Gaussian quadrature, MC, BQ
  - includes interpolation + integration comparison section

- `source/polynomial_interpolations.ipynb`
  - interpolation diagnostics
  - error vs node count
  - integration error impact of interpolation quality

- `source/final_project_report.ipynb`
  - theory + reproducible BQ-heavy experiments
  - kernel sensitivity, budget scaling, active sampling, interpolation/derivative-assisted BQ

## 7. References

- https://www.youtube.com/watch?v=T63ATAXn63Y
- https://store.fmi.uni-sofia.bg/fmi/statist/personal/vandev/lectures/stochproc.pdf
- https://medium.com/@tinonucera/building-quadratic-approximation-in-bayesian-inference-from-scratch-step-by-step-example-f17167cbe9eb
