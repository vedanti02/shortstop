# Theory notes (Vedanti)

Two results: the cost-optimal stopping policy, and why the confidence intervals are
valid. These are proof sketches built from standard theorems. They have been checked
numerically (see the end of each section) but not yet reviewed by anyone else.

## 1. Setup

One run of an agent on one task has steps $k = 1, \dots, T$, a final outcome
$Y \in \{0, 1\}$, and a predictor that outputs $f_k \in [0, 1]$ after step $k$. Set
$f_T = Y$. Step $k$ costs $c_k \ge 0$.

A stopping policy picks reach probabilities $1 \ge P_1 \ge P_2 \ge \dots \ge P_T \ge p_{\min} > 0$,
where $P_k$ may depend on anything seen before step $k$. A single uniform draw $U$,
independent of the run, sets the stop step $s = \max\{k : U < P_k\}$. The run is paid for
up to step $s$. The estimate is

$$\hat Y = f_0 + \sum_{k=1}^{s} \frac{f_k - f_{k-1}}{P_k}.$$

Write $u_k = 1 / P_k$ with $u_0 = 1$, $w_k = u_k - u_{k-1} \ge 0$, and $e_k = Y - f_k$
(so $e_T = 0$).

**Lemma 1 (unbiased and bounded).** $E[\hat Y \mid \text{run}] = Y$ and
$-(1/p_{\min} - 1) \le \hat Y \le 1/p_{\min}$.

*Proof.* Given the run, every $P_k$ is a fixed number and $P(s \ge k) = P_k$, so each
term has expectation $f_k - f_{k-1}$ and the sum telescopes to $Y$. For the bound, sum by
parts: $\hat Y = f_s u_s - \sum_{k<s} f_k w_{k+1}$. Both pieces lie in $[0, u_s]$ and
$\sum_{k<s} w_{k+1} = u_s - 1$. $\square$

**Lemma 2 (variance).**
$\operatorname{Var}(\hat Y \mid \text{run}) = \sum_{k=1}^{T} w_k \, e_{k-1}^2.$

*Proof.* Let $Z_k = \mathbf 1\{s \ge k\} / P_k$, $Z_0 = 1$. Then
$\hat Y - Y = \sum_k (e_{k-1} - e_k)(Z_k - 1) = \sum_k e_{k-1} (Z_k - Z_{k-1})$ by summation
by parts. $Z_k$ is a martingale in $k$, so its increments are uncorrelated and
$E[(Z_k - Z_{k-1})^2] = 1/P_k - 1/P_{k-1} = w_k$. $\square$

This is the same quantity as `stopping.exact_moments`, written so that every term is
non-negative: stopping adds variance only through increases in $1/P$, each weighted by
the squared error of the prediction in hand at that moment.

## 2. The cost-optimal policy

**Problem.** Over a population of runs, minimize the compute needed for a fixed interval
width, $(\operatorname{Var}(Y) + E[\operatorname{Var}(\hat Y \mid \text{run})]) \cdot E[\text{cost}]$.
A minimizer of this product also minimizes
$E[\operatorname{Var}(\hat Y \mid \text{run})] + \lambda \, E[\text{cost}]$ for one value
of $\lambda > 0$, so it is enough to solve the second problem for every $\lambda$ and scan.

Let $s_{k-1}$ be the state before step $k$: whatever the policy may look at. Assume it is
Markov (otherwise take the whole history as the state). Define the expected drop in
squared error and the expected cost at step $k$:

$$a_k(s) = E[e_{k-1}^2 - e_k^2 \mid s_{k-1} = s], \qquad \bar c_k(s) = E[c_k \mid s_{k-1} = s].$$

**Theorem 1.** Define the index of a state as the best ratio of error removed to cost
paid, over every rule $\tau$ for how far to look ahead (a stopping time of the state
process, $\tau \ge k$):

$$\nu_k(s) = \max_{\tau \ge k} \;
\frac{E\big[e_{k-1}^2 - e_\tau^2 \mid s_{k-1} = s\big]}{E\big[\sum_{j=k}^{\tau} c_j \mid s_{k-1} = s\big]}.$$

Then the optimal policy is

$$P_k = \min\Big(P_{k-1},\; \operatorname{clip}\big(\sqrt{\nu_k(s_{k-1})^{+} / \lambda},\; p_{\min},\; 1\big)\Big).$$

*Proof sketch.*
1. By Lemma 2 the objective is $E\big[\sum_k (u_k - u_{k-1}) e_{k-1}^2 + \lambda c_k / u_k\big]$
   with $1 \le u_1 \le \dots \le u_T \le 1/p_{\min}$. It is jointly convex in the $u_k$ and
   the constraints are linear, so dynamic programming gives the global optimum.
2. Let $r_k(s) = E[e_{k-1}^2 \mid s]$ and
   $G_k(s, u) = u \, r_k(s) + \lambda \bar c_k(s) / u + E[V_{k+1}(s', u) \mid s]$, where
   $V_{k+1}(s', u) = -u \, r_{k+1}(s') + \min_{u' \ge u} G_{k+1}(s', u')$. Each $G_k$ is
   convex in $u$, so the best choice at step $k$ is $u_k = \max(u_{k-1}, u^*_k(s))$ with
   $u^*_k(s)$ the minimizer of $G_k(s, \cdot)$. This is the running minimum in the theorem.
3. Differentiate. Since $\partial_u V_{k+1}(s', u) = -r_{k+1}(s') + (\partial_u G_{k+1}(s', u))^+$
   and $r_k(s) - E[r_{k+1}(s') \mid s] = a_k(s)$, the derivative $M_k = \partial_u G_k$
   satisfies
   $M_k(s, u) = a_k(s) - \lambda \bar c_k(s) / u^2 + E[(M_{k+1}(s', u))^+ \mid s]$,
   with $M_{T+1} = 0$.
4. That recursion is an optimal stopping problem:
   $M_k(s, u) = \max_{\tau \ge k} E\big[\sum_{j=k}^{\tau} (a_j - \lambda c_j / u^2) \mid s\big]$.
   So $M_k(s, u) \ge 0$ exactly when $1/u^2 \le \nu_k(s) / \lambda$, which puts the
   minimizer of $G_k$ at $u^*_k(s) = \sqrt{\lambda / \nu_k(s)}$. The sum of $a_j$ from $k$
   to $\tau$ telescopes to $e_{k-1}^2 - e_\tau^2$ in expectation. $\square$

**What it says.**
- The optimal policy has the shape already used here: a target that depends on the
  state, followed down but never back up.
- **Learned tracking is the case $\tau = T$** (always look ahead to the end of the run):
  $\sqrt{E[e_{k-1}^2] / (\lambda E[\text{remaining cost}])}$. The index is a maximum over
  more rules, so the optimal policy continues at least as often as learned tracking, and
  the two are the same whenever looking to the end is the best horizon.
- **The original tracking rule is the calibrated case.** If the predictor is calibrated,
  $E[e_{k-1}^2 \mid s] = f_{k-1}(1 - f_{k-1})$, which gives
  $\kappa \sqrt{f(1 - f)}$ when the remaining cost is treated as constant.
- The index is high where a cheap step is about to teach the predictor a lot, even if the
  rest of the run is expensive and uninformative. That is the situation in which stopping
  inside a run beats deciding once at the start.
- A schedule that depends only on the step number is the special case with one state per
  step.

**Numerical check.** `tests/test_index_policy.py` builds small trees of runs where the
policy sees the full history, computes the index by backward induction, and compares
against a direct convex optimization over every reach probability. The two agree to six
digits in 36 cases, and looking only to the end of the run is strictly worse in 10 of
them.

**On the SWE-bench data** (`optimal_policy.py`). The index is estimated on a state of
(step, prediction decile). It does not beat learned tracking: 54% against 52% median
compute on held-out agents, and 55.7% against 55.4% when fitted and scored on all agents
together. With the stand-in predictor the largest drop in squared error comes at the very
end of a run, when the tests reveal the outcome, so looking to the end is already the
best horizon and the two rules coincide. The index can only separate from learned
tracking with a predictor that resolves uncertainty mid-run.

## 3. Validity of the confidence interval

Tasks are drawn independently from a population (or resampled from the benchmark), the
target is $\theta = E[Y]$, and the interval is
$\bar Y_n \pm 1.96 \, \hat\sigma_n / \sqrt n$ with $\bar Y_n$ and $\hat\sigma_n$ the mean
and standard deviation of the per-task estimates $\hat Y_1, \dots, \hat Y_n$.

**Theorem 2a (fixed predictor and policy).** The interval has asymptotic coverage 95%.

*Proof.* By Lemma 1 the $\hat Y_i$ are independent, identically distributed, bounded, and
have mean $\theta$. Their variance $\sigma^2 = \operatorname{Var}(Y) + E[\sum_k w_k e_{k-1}^2]$
is finite and positive. The central limit theorem gives
$\sqrt n (\bar Y_n - \theta) \to N(0, \sigma^2)$, the law of large numbers gives
$\hat\sigma_n \to \sigma$, and Slutsky's lemma combines them. $\square$

No property of the predictor is used. A bad predictor makes $\sigma^2$ larger and the
interval wider, never wrong. Deterministic stopping fails the first line of this proof:
its per-task scores do not have mean $\theta$.

*If the target is the mean over one fixed set of tasks* rather than a population, the
estimate is still exactly unbiased, and the same interval is conservative, because
$\hat\sigma_n^2$ also counts the spread of $Y$ across tasks, which no longer contributes
to the error.

**Theorem 2b (predictor corrected during the evaluation).** Suppose the predictor and
policy used for task $i$ depend on earlier tasks through a parameter $\delta_i$ computed
from tasks $1, \dots, i-1$ only, the per-task variance $v(\delta)$ is continuous in
$\delta$, and $\delta_i \to \delta^*$ in probability. Then the same interval has
asymptotic coverage 95%.

*Proof.* Let $\mathcal F_{i-1}$ be everything observed through task $i - 1$. Task $i$ is
drawn independently of $\mathcal F_{i-1}$ and $\delta_i$ is fixed given $\mathcal F_{i-1}$,
so Lemma 1 gives $E[\hat Y_i \mid \mathcal F_{i-1}] = \theta$: the differences
$\xi_i = \hat Y_i - \theta$ form a bounded martingale difference sequence. Hence
$\bar Y_n$ is unbiased for every $n$.

For the limit, $E[\xi_i^2 \mid \mathcal F_{i-1}] = v(\delta_i) \to v(\delta^*)$, so the
average conditional variance converges to the constant $\sigma_*^2 = v(\delta^*)$. The
Lindeberg condition holds because the $\xi_i$ are bounded. The martingale central limit
theorem then gives $\sqrt n (\bar Y_n - \theta) \to N(0, \sigma_*^2)$. For the variance
estimate, $\xi_i^2 - E[\xi_i^2 \mid \mathcal F_{i-1}]$ is again a bounded martingale
difference, so $\frac1n \sum \xi_i^2 \to \sigma_*^2$ and therefore
$\hat\sigma_n \to \sigma_*$. $\square$

*The conditions hold for `adaptive.py`.* The shift $\delta$ is recomputed between batches
from finished tasks only. It solves "mean shifted step-0 prediction on seen tasks =
running estimate" and is kept in $[-8, 8]$. The running estimate converges to $\theta$ by
the martingale law of large numbers, the left side converges uniformly in $\delta$, and
it is strictly increasing in $\delta$, so $\delta_i$ converges to the unique solution of
$E[\operatorname{sigmoid}(\operatorname{logit} f_0 + \delta)] = \theta$. The reach
probabilities are continuous in $\delta$, so $v$ is too.

**How large $n$ has to be.** The theorems are asymptotic. The Berry-Esseen bound puts the
coverage error at order $E|\xi|^3 / (\sigma^3 \sqrt n)$. In the worst case an estimate is
of size $1/p$ with probability about $p$, where $p$ is a typical reach probability at the
floor, which gives $E|\xi|^3 \sim p^{-2}$ and $\sigma^3 \sim p^{-3/2}$. The error is then
of order

$$\frac{1}{\sqrt{n \, p_{\min}}}.$$

What matters is $n \cdot p_{\min}$, roughly the number of runs that are finished from the
group the policy is most willing to drop, not $n$ itself. With 500 tasks, a floor of 0.05
leaves about 25 such runs and a floor of 0.2 leaves about 100.

**Numerical check** (`adaptive.py`, `results/adaptive_*.csv`).
- The largest bias over 37 agents is 0.6 points at floor 0.05 and 0.3 points or less at
  floors 0.1 and 0.2.
- The estimated variance of the mean is within 9% of its true variance for every agent
  and within 1% on average (`variance_ratio`), so the variance estimate is not the problem.
- The misses are one-sided (`miss_below`, `miss_above`). On the five agents checked at
  floor 0.05 the interval missed on one side about two to three times as often as on the
  other. That is the signature of skew from a few heavily weighted runs.
- Coverage of the corrected method rises with the floor, in the direction the
  $1/\sqrt{n \, p_{\min}}$ rate predicts: 92.4% at 0.05, 94.0% at 0.1 and 94.6% at 0.2
  (the last two with the damped correction), against 94.8% for full runs.
- Hall's skewness-corrected interval was tried and did not help consistently, plausibly
  because the sample skewness is driven by the same few runs.
