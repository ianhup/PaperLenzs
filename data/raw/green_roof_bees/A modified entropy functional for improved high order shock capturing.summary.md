## Title
A modified entropy functional for improved high order shock capturing methods

## Topics
Computational fluid dynamics; entropy indicators; discontinuous Galerkin methods; shocks; near-vacuum flows

## Introduction
Classical logarithmic entropy becomes numerically sensitive near vacuum, potentially undermining detection of elements requiring stabilization in high-order compressible-flow simulations.

## Methods
The author develops a piecewise continuously differentiable entropy functional, analyzes compatibility and convexity using generalized entropy theory, and substitutes it into a modal-filtering troubled-element detector. Numerical tests include difficult shock, high-Mach-number, and viscous-flow configurations.

## Results
The modified detector improves reliability in demanding cases, including near-vacuum shock-tube behavior and a Mach-10,000 astrophysical jet. The functional forms a convex entropy-flux pair above a computable threshold; below it, strict convexity is lost.

## Discussion
The low-pressure branch is a regularized detection indicator, not a proof of global entropy stability. The work preserves the baseline stabilization framework and identifies steady bow-shock behavior as requiring further refinement.

## Conclusion
A carefully regularized entropy indicator can improve robustness of high-order shock capturing while retaining explicit limits on its mathematical interpretation.

## Relevance to research
Not relevant to the mockup's bee-and-green-roof question. It studies numerical treatment of extreme compressible flows, not urban ecology or count-data models. Entropy here is a mathematical/thermodynamic quantity, not a measured biodiversity response.
