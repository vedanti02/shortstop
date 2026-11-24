# Balanced stopping: why its error bars are valid

Theory note for the `balanced` design of `predictor/runtime.py` (Deepika). It builds on
`coverage/THEORY.md` (Assumption A, Lemmas 1 and 2, Theorem 2a). The proofs use only standard
results and are checked numerically (Section 6), but still need a second reader who checks
each step.

**In plain words.** Balanced stopping deals the lottery numbers of each group of 50 tasks like
cards, so luck shared by the group cancels. Each run stays fair, so the score was already
known to be unbiased. The error bar needed its own proof, and working it out showed two things:

- **The earlier error-bar formula is slightly too narrow.** Its finite-population factor
  1 − m/N is right for *which* runs start. It also shrinks the noise from *how far* the started
  runs go, which dealing does not remove. The formula misses a share m/N of that later-step
  noise (m/N is the share of runs that start, about 15%). It cannot be proven valid, because in
  some cases it is not. The runtime keeps it as `design="balanced-unproven"` for comparison.
- **Two small changes make it provably valid**, in `design="balanced"`:
  1. Every run in a group gets the same chance to start, a whole number of 1/N slices.
  2. The error bar adds one term: the later-step noise, estimated from each started run's own
     observed steps.

  With both, the expected variance estimate is at least the true variance for *every* set of
  runs (Proposition B3). The interval reaches at least 95% coverage as the number of tasks
  grows (Theorem B4).

The cost, against the earlier formula: compute for the reported width is 45.4% of full runs on
SWE-bench Verified (instead of 43.2%) and 49.8% on SWE-rebench (instead of 48.0%). The
`standard` design needs 47.7% and 52.4% (Section 6).

## 1. The design: equal-chance strata

Section 1 of `coverage/THEORY.md` holds for every run: Assumption A, and a floor $p_{\min}$.
Run lengths are bounded by some $T_{\max}$.

1. **Strata.** Before any run starts, sort the $n$ tasks by their step-1 reach probability
   $\pi_1$ (ties by $f_0$). Cut them into strata of $N$ consecutive tasks ($N = 50$). The last
   stratum also takes the remainder, so it has $N$ to $2N - 1$ tasks.
2. **One chance to start per stratum.** In stratum $g$, let
   $m_g = \operatorname{round}\big(\sum_{j \in g} \pi_1(h_{j,0})\big)$, kept between
   $\max(2, \lceil N p_{\min} \rceil)$ and $N$. **Every run in the stratum gets the same step-1
   reach probability $p_g = m_g / N$.** Later reach probabilities are the policy's, capped at
   $p_g$, so they stay non-increasing.
3. **Dealing.** Draw the lottery numbers $U_j = (\sigma(j) + V_j)/N$ for a uniformly random
   permutation $\sigma$ of $\{0, \dots, N-1\}$ and independent $V_j \sim U(0,1)$, independently
   across strata. A run reaches step $k$ iff $U_j < P_{j,k}$.

The strata and the $p_g$ use only information from before any run starts, so they are fixed
before any draw. Each $U_j$ is uniform and independent of run $j$, so Assumption A holds for
every run. The policy may use other tasks' pre-run information, which is fixed before any draw.

**Exactly $m_g$ runs start in stratum $g$**: those in slots $0, \dots, m_g - 1$, because
$U_j < p_g$ iff $\sigma(j) < m_g$.

Code: `balanced.equal_chance_uniforms`, and `BalancedPlan` in `runtime.py`.

## 2. The estimate and the error bar

For run $j$, write:
- $c_j = \hat Y_j - f_{j,0}$, its correction ($c_j = 0$ if it does not start);
- $e_j = Y_j - f_{j,0}$;
- $\Delta_{j,l} = f_{j,l} - f_{j,l-1}$ and $w_{j,k} = 1/P_{j,k} - 1/P_{j,k-1}$, with $P_{j,0} = 1$, as in
  `coverage/THEORY.md`.

**Later-step noise of one run, estimated from its own steps.** For a run that started and stopped
after step $s \ge 1$:

$$\hat\omega = \sum_{2 \le k \le l \le s} w_k \, \frac{\Delta_l \,(\Delta_l + 2 (f_{l-1} - f_{k-1}))}{P_l}.$$

It uses only the steps the run took (if the run finished, $f_s = Y$). Code:
`balanced.later_step_table` and `balanced.omega_hat`.

**Score:** $\hat\theta = \frac1n \sum_j \hat Y_j$. **Error bar:** $\hat\theta \pm t \sqrt{\hat V}$, with

$$\hat V = \frac{\hat\theta (1 - \hat\theta)}{n} + \frac{1}{n^2} \sum_g \hat V_g, \qquad
\hat V_g = (1 - f_g)\, m_g\, s_g^2 + \gamma_g \sum_{j \in S_g} \hat\omega_j.$$

Here:
- $S_g$ is the set of started runs ($|S_g| = m_g$), and $s_g^2$ the sample variance of their
  corrections $c_j$;
- $f_g = m_g / N_g$, and $\gamma_g = f_g + \frac{m_g - f_g}{(m_g - 1)(N_g - 1)}$;
- $t$ is the Student quantile with $\sum_g (m_g - 1)$ degrees of freedom.

Code: `balanced.proven_variance`.

The first piece of $\hat V_g$ is the earlier formula (`balanced.design_variance` with the
finite-population factor; with equal chances its $M$ is $m_g$). The second piece is new. For
$N = 50$, $\gamma_g$ is $f_g$ plus 0.02 to 0.03.

## 3. Results

**Lemma B1 (each run stays fair).** $E[\hat Y_j \mid \text{run } j] = Y_j$, so $E[\hat\theta \mid \text{runs}] = \bar Y_n$.

*Proof.* Lemma 1 of `coverage/THEORY.md` for each run. Only the law of $U_j$ enters, and it is
uniform and independent of run $j$. $\square$

**Lemma B2 (later-step noise can be estimated).** For a run that starts,
$\omega := \operatorname{Var}(c \mid \text{run}, \text{started}) = \frac{1}{P_1} \sum_{k=2}^{T} w_k (Y - f_{k-1})^2$,
and $E[\hat\omega \mid \text{run}, \text{started}] = \omega$.

*Proof.*
- *The variance.* Given that the run started, $U$ is uniform on $[0, P_1)$, and
  $c = \frac{1}{P_1}\big(\Delta_1 + \sum_{k \ge 2} \frac{\Delta_k}{P_k / P_1} \mathbf 1\{s \ge k\}\big)$.
  This is $1/P_1$ times the estimate for the run continued from step 1, with reach probabilities
  $P_k/P_1$. Lemma 2 applied to that continuation gives
  $\operatorname{Var}(c) = \frac{1}{P_1^2} \sum_{k \ge 2} (\frac{P_1}{P_k} - \frac{P_1}{P_{k-1}}) e_{k-1}^2 = \omega$.
- *The mean.* Fix $k \ge 2$ and write $D_l = (f_l - f_{k-1})^2 - (f_{l-1} - f_{k-1})^2$. Given a start,
  $P(s \ge l) = P_l / P_1$, so the sum telescopes ($f_T = Y$):
  $E[\sum_{l=k}^{s} D_l / P_l] = \frac{1}{P_1} \sum_{l=k}^{T} D_l = \frac{(Y - f_{k-1})^2}{P_1}$.
  The terms with $k > s$ are empty sums, so summing $w_k$ times this over $k = 2, \dots, T$ gives
  $E[\hat\omega] = \omega$. $\square$

For the next result, fix a stratum and drop the index $g$. Notation:
- $x_j = e_j / p$, the mean correction of run $j$ if it starts.
- $\nu_{j,b} = E[c_j \mid \sigma(j) = b]$ for the started slots $b = 0, \dots, m-1$.
- $W_j = (\nu_{j,b} - x_j)_{b < m} \in \mathbb R^m$, run $j$'s **depth profile**: how its expected
  correction depends on how deep its lottery number lets it go.
- $\hat C = \sum_j c_j$, and $\Pi = \sum_{j \ne k} \langle W_j, W_k \rangle = \lVert \sum_j W_j \rVert^2 - \sum_j \lVert W_j \rVert^2$.

**Proposition B3 (exact bias, for any runs).** Conditional on the runs of the stratum,

$$E[\hat V_g] - \operatorname{Var}(\hat C) = \underbrace{\frac{m - f}{(m-1) N (N-1)}\,\Pi}_{A}
\;+\; f (\gamma - f) \sum_{j=1}^{N} \omega_j \;\ge\; 0.$$

*Proof.*
1. *Two stages.* $S = \{j : \sigma(j) < m\}$ is a uniformly random $m$-subset. Given $S$:
   - the slots of the started runs are a uniformly random bijection onto $\{0, \dots, m-1\}$;
   - the $V_j$ stay independent;
   - so their lottery numbers form a Latin hypercube with $m$ slots on $[0, p)$.

   Each started $U_j$ is uniform on $[0, p)$, so $E[c_j \mid S] = x_j$ and
   $\operatorname{Var}(c_j \mid S) = \omega_j$.
2. *Variance.* $\operatorname{Var}(\hat C) = \operatorname{Var}(\sum_{j \in S} x_j) + E[L(S)]$, with
   $L(S) = \operatorname{Var}(\hat C \mid S)$. Because $S$ samples $m$ of $N$ without replacement,
   $\operatorname{Var}(\sum_{j \in S} x_j) = m (1 - f) S_x^2$, where $S_x^2 = \frac{1}{N-1} \sum_j (x_j - \bar x)^2$.
3. *Covariances in the Latin hypercube.* For $j \ne k$ in $S$, the pair $(\sigma(j), \sigma(k))$ is
   uniform over ordered pairs of distinct slots, and the $V$'s are independent. With
   $\sum_b \nu_{j,b} = m x_j$, this gives
   $\operatorname{Cov}(c_j, c_k \mid S) = \frac{1}{m(m-1)} \sum_{b \ne b'} \nu_{j,b}\nu_{k,b'} - x_j x_k = -\frac{\langle W_j, W_k\rangle}{m(m-1)}$.
   Hence $L(S) = \sum_{j \in S} \omega_j - \frac{1}{m(m-1)} \sum_{j \ne k \in S} \langle W_j, W_k \rangle$.
4. *Sample variance.* With $s_x^2(S)$ the sample variance of the $x_j$ over $S$,
   $E[(m-1) s^2 \mid S] = \sum_{j \in S} (x_j^2 + \omega_j) - \frac1m \big[(\sum_{j \in S} x_j)^2 + L(S)\big]
   = (m-1)\, s_x^2(S) + \sum_{j \in S} \omega_j - L(S)/m$.
5. *Average over $S$.*
   - $E[s_x^2(S)] = S_x^2$, and $E[\sum_{j \in S} \omega_j] = f \sum_j \omega_j$.
   - Each pair is in $S$ with probability $\frac{m(m-1)}{N(N-1)}$, so
     $E[L(S)] = f \sum_j \omega_j - \frac{\Pi}{N(N-1)}$.
   - By Lemma B2, $E[\sum_{j \in S} \hat\omega_j] = f \sum_j \omega_j$.
6. *Combine.* $\operatorname{Var}(\hat C) = m(1-f) S_x^2 + f \sum_j \omega_j - \frac{\Pi}{N(N-1)}$. With
   $\gamma = f$, steps 4 and 5 give
   $E[\hat V] = m (1-f) S_x^2 + f \sum_j \omega_j + \frac{(1-f)\,\Pi}{(m-1)N(N-1)}$, and the
   difference is $A$. A general $\gamma$ adds $(\gamma - f) f \sum_j \omega_j$.
7. *Sign.* $\Pi \ge -\sum_j \lVert W_j \rVert^2$, and $\lVert W_j \rVert^2 \le m\, \omega_j$ (the spread of
   the slot means is part of $\omega_j$). So $A \ge -\frac{m (m-f)}{(m-1)N(N-1)} \sum_j \omega_j$, and
   $f(\gamma - f) \sum_j \omega_j$ cancels this exactly, because $m / (f N) = 1$. $\square$

**Theorem B4 (coverage).** Let the number of strata grow, with $N$ fixed.

- (a) *Fixed benchmark.* Suppose $\frac1n \sum_g \operatorname{Var}(\hat C_g \mid \text{runs})$ stays
  bounded away from 0. Then $P(\bar Y_n \in \hat\theta \pm t\sqrt{\hat V}) \ge 0.95 - o(1)$: the
  interval contains the benchmark's full-run score.
- (b) *Population* (tasks drawn independently, as in Theorem 2a). Suppose moreover that
  $\frac1n \sum_g \operatorname{Var}(\hat C_g \mid \text{runs})$ converges in probability. Then the
  interval contains $\theta = E[Y]$ with probability at least $0.95 - o(1)$.

*Proof of (a).*
- Given the runs, the strata are independent, and each $\hat C_g - C_g$ is bounded by
  $2N (1/p_{\min} + 1)$ (Lemma 1). The Lindeberg–Feller CLT gives
  $(\hat\theta - \bar Y_n) / \sqrt{\operatorname{Var}(\hat\theta \mid \text{runs})} \to N(0, 1)$, given the runs.
- Each $\hat V_g$ is bounded (in terms of $N$, $p_{\min}$ and $T_{\max}$) and independent across
  strata. By Chebyshev, $n \cdot \frac{1}{n^2}\sum_g (\hat V_g - E[\hat V_g \mid \text{runs}]) \to 0$.
- By Proposition B3, $\frac{1}{n^2} \sum_g E[\hat V_g \mid \text{runs}] \ge \operatorname{Var}(\hat\theta \mid \text{runs})$.
- The first piece of $\hat V$ is nonnegative, and $t \to 1.96$.

*Proof of (b).*
- Split $\hat\theta - \theta = (\bar Y_n - \theta) + (\hat\theta - \bar Y_n)$. By the CLT, the first
  term is asymptotically $N(0, \operatorname{Var} Y / n)$.
- Given the runs, the second term is asymptotically $N(0, \sigma_L^2 / n)$, with $\sigma_L^2$ the
  assumed limit. Conditioning on the runs (characteristic functions and dominated convergence),
  the sum is asymptotically $N(0, (\operatorname{Var} Y + \sigma_L^2)/n)$.
- Because $Y$ is 0 or 1, $\hat\theta(1 - \hat\theta) \to \theta(1-\theta) = \operatorname{Var} Y$.
  By (a), the second piece of $n\hat V$ is at least $\sigma_L^2 - o_P(1)$. $\square$

## 4. Remarks

- **Where the gain comes from.** The same covariance computation over all $N$ slots gives
  $\operatorname{Var}(\hat C) = \sum_j v_j - \frac{1}{N(N-1)} \big(\lVert \sum_j \tilde\mu_j \rVert^2 - \sum_j \lVert \tilde\mu_j \rVert^2\big)$.
  Here $v_j$ is run $j$'s variance under independent draws (Lemma 2), and $\tilde\mu_j$ its profile
  over all $N$ slots.
  - Balancing removes the part of the stopping noise that the runs of a stratum share. That is
    mostly a shared misjudgment, such as a predictor that overrates the agent.
  - It never costs more than a factor $N/(N-1)$ (about 2%) against independent draws.
  - This is why weak agents gain most (Section 6).
- **The earlier formula** (`balanced-unproven`'s) has no $\hat\omega$ term. On equal-chance
  strata its bias is $A - f^2 \sum_j \omega_j$. It misses the share $f$ of the later-step noise
  $f \sum_j \omega_j$, and gets some of it back only when the runs' depth profiles line up ($A > 0$).
  If every run starts ($f = 1$), it estimates zero later-step noise.
- **The cheaper $\gamma_g = f_g$** has bias exactly $A$.
  - $A \ge 0$ on average when runs sorted together are alike. But it is not guaranteed for every
    set of runs: in small strata of unrelated runs, $A$ reaches −30% of the variance (Section 6).
  - On the 37 Verified agents it was at least −0.3% of the variance.
  - It is in `balanced.py`'s simulation (interval "design + fpc + later steps, gamma = f"), not
    in the runtime.
- **Unequal chances are not covered.** `balanced-unproven`'s draws let each run keep its own
  $P_1$, and its started runs are then not a simple random sample of the stratum. Equal chances
  cost almost nothing: the actual compute (true variance) is 43.4% against 43.3% on Verified,
  and 47.7% against 46.7% on SWE-rebench.

## 5. What this does not cover

- **Finite samples.** Theorem B4 is asymptotic, like Theorem 2a. Coverage at 500 tasks is checked
  by simulation (Section 6).
- **In-batch correction.** The predictor and the policy must be fixed before the batch.
  Calibration from earlier runs is fine; `coverage/adaptive.py`'s in-batch correction is not covered.
- **Tool A/B differences under balanced stopping.** The analysis should carry over with each run's
  correction replaced by the difference of the two arms' corrections, but this is not done.
- **Assumption A in live runs**, as in `coverage/THEORY.md`: the agent must not see the lottery.
- **Batches.** The design needs at least 2 tasks, and the task list before the first run starts.

## 6. Checks

**Exact, no Monte Carlo** (`tests/test_balanced.py::test_variance_identities_are_exact`):
- **Setup.** 200 random strata of 3 to 5 runs, with every $m$ and up to 5 steps. Every
  permutation is enumerated, and the within-slot uniforms are integrated exactly.
- **What holds, to machine precision:**
  - Lemma B2;
  - Proposition B3, for $\gamma = f$ and for the earlier formula;
  - $E[\hat V_g] \ge \operatorname{Var}(\hat C)$ with the safe $\gamma$;
  - the $N/(N-1)$ bound.
- **Production code.** `test_proven_variance_matches_its_expectation` checks
  `proven_variance` against Proposition B3's exact value.

**Real runs** (`predictor/balanced_check.py` → `results/balanced_check.csv`):
- **Setup.** 37 Verified agents, the 500 tasks fixed, equal-chance strata of 50, 20,000 draws of
  the production code.
- **Expected value of each formula, divided by the true variance:**

| Formula | Median | Range | Agents below 1 |
| --- | --- | --- | --- |
| Earlier formula | 0.959 | 0.912 to 1.161 | 35 of 37 |
| $\gamma = f$ | 1.011 | 0.997 to 1.248 | 4 |
| **Proven** | **1.019** | **1.005 to 1.263** | **0** |

- **The term $A$** ranges from −0.3% to +24.8% of the variance (median +1.1%). It is negative in
  122 of the 370 strata.
- **Simulation against the exact values.** The simulated expectations match within 0.7% of the
  variance, and the simulated variance is 0.98 to 1.02 times the exact one.

**Coverage and compute** (`balanced.py`: tasks resampled, learned-remaining-cost policy, 1,000
replicates, geometric means over agents; `results/balanced.csv` and `results/balanced_rebench.csv`):

| | Verified: compute (reported) | Coverage, mean / worst | SWE-rebench: compute (reported) | Coverage, mean / worst |
| --- | --- | --- | --- | --- |
| `standard` (Theorem 2a) | 47.7% | 94.8% / 92.5% | 52.4% | 94.9% / 93.1% |
| `balanced-unproven` (earlier formula) | 43.2% | 94.6% / 92.9% | 48.0% | 94.8% / 93.5% |
| **`balanced` (Theorem B4)** | **45.4%** | **95.2% / 93.6%** | **49.8%** | **95.3% / 93.8%** |
| Equal chances, $\gamma = f$ | 45.1% | 95.1% / 93.6% | 49.6% | 95.3% / 93.7% |

- **Against the tracking baseline** (51.3% and 57.0%), `balanced` needs 12% and 13% less compute
  (per-agent geometric-mean ratios 0.884 and 0.874). The earlier formula claimed 16% (0.842 and
  0.843).
- **Against `standard`**, `balanced` saves 5% overall (ratios 0.952 and 0.950). It is cheaper for
  23 of 37 Verified agents and 24 of 36 SWE-rebench configurations.
- **The gain sits with agents the predictor misjudges.** Against the baseline, by resolve rate on
  Verified: 0.44 up to 30%, 0.71 for 30 to 50%, 0.97 for 50 to 65%, 0.92 above 65%.
