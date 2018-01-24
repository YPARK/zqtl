# Technical details

## Removing discrepancy between two reference panels in mediation analysis

Assume `X1 / sqrt(n1) = U1 D1 t(V1)` and `X / sqrt(n) = U D t(V)`.

```
E[α.hat]           = S1 R1 inv(S1) α
```
therefore
```
E[α]               = S1 inv(R1) inv(S1) α.hat
                   = S1 V1 inv(D1^2) t(V1) inv(S1) α.hat
```
Construct a "design matrix" for the mediation effect:
```
t(V) R inv(S) E[α] = t(V) V D^2 t(V) inv(S) α
                   = D^2 t(V) inv(S) S1 V1 inv(D1^2) t(V1) inv(S1) α.hat
M                  = D^2 t(V) inv(S) S1 (V1/D1) t(inv(D1)V1) * Zα
```
Mediation regression model assuming mean-field distribution of `β`:
```
E[Δ] = M E[β]
V[Δ] = M^2 V[β]
```
For a perfect mediation:
```
E[t(V)Zθ] = E[Δ]
```

