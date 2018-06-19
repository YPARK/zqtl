# Variance calculation

## Notations

- p : the number of SNPs

- n : the number of individuals

- n0 : the number of individuals in the mediation QTL models

- K : the number of mediators

- X : a standardized genotype matrix (n x p); mean = 0 and variance 1

- y : phenotype vector (n x 1)

- z : z-score vector of XWAS (p x 1)

- a : the estimated univariate effect size of mediation QTLs (p x K)

- S0 : the diagonal matrix of standard errors (p x p)

- za : the z-score matrix for mediation QTLs

- α : multivariate effect size of mediation QTLs

- β : mediation effect size (K x 1)

- γ : unmediated effect size (p x 1)


## Estimation of residual effects on reference panel

From the linear model `y = X θ + ε`, we can derive the estimated
univariate statistics.

```
θ.uni = t(X)y / n
      = t(X)(X θ + ε) / n
      = t(X)X θ / n + t(X)ε / n
```

With `S = σ / sqrt(n)` and exploiting SVD `X/sqrt(n) = UD t(V)`,

```
z     = θ.uni * sqrt(n) / σ
      = R (θ * sqrt(n)/σ) + t(X/sqrt(n)) ε /σ 

t(V)z = t(V) V D^2 t(V) (θ * sqrt(n)/σ) + t(V)VD t(U) (ε/σ)
      = D^2 t(V) (θ * sqrt(n)/σ) + D^2 inv(D)t(U) (ε/σ)
```
Treating `y.tilde = t(V)z` and `Δ = inv(D)t(U)`,
we can estimate the individual-level residual values `ε` fitting the regression :

```
y.tilde ~ ... + D^2 Δ (ε/σ) + N(0, D^2)
```

Then we can estimate the scaled residual variance by `V[ε/σ]`, and we
can approximate that value with `sum(ε^2)/nσ^2` if `E[ε_i] = 0`.  For
unbiased reference panel, it is not so hard to see that `V[ε/σ] -> 1`.

## Variance components

We define variance explained by the linear model on the reference cohort genotype matrix, 
`var_model = V[Xθ]/σ^2`. Defining `η = Xθ` we can estimate

```
V[Xθ] = Σ η_j^2/n - (Σ η_j/n)^2.
```

Alternatively we could use `t(θ) R θ`, but we may not have `Ε[Xθ] = 0` in practice.
Variance unexplained:

```
var_residual = V[ε/σ] ~ Σ ε_i^2 / nσ^2
```

## Reconciling different samples sizes between the mediation and GWAS models

The model for mediated QTLs:

```
E[a]  = S0 R inv(S0)α
α     = S0 inv(R) inv(S0) E[a]
```

For simplicity we assume `S = (σ/sqrt(n))I` and `S0 = (σ0/sqrt(n0))I` and `X` is standardized.

```
E[θ.uni] = S R inv(S) θ
         = S R inv(S) (αβ + γ)
         = S R inv(S) S0 inv(R) inv(S0) E[a] β + S R inv(S) γ
         = E[a] β + S R inv(S) γ
E[z]     = inv(S) E[a] β + R inv(S) γ
t(V)E[z] = (t(V) inv(S) E[a]) β + D^2 t(V) inv(S) γ
         = (t(V) Λ inv(S0) E[a]) β + D^2 t(V) inv(S) γ
where
Λ        = sqrt(n)/sqrt(n0) I
```

- Variance explained by mediated component:

```
αβ            = inv(R) E[a] β = V inv(D^2) t(V) E[a] β
var_mediation = Σ (αβ)_j^2 / σ^2
```
