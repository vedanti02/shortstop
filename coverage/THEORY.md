# Theory notes (Vedanti)

Two results: the cost-optimal stopping policy, and why the confidence intervals are
valid. Theorem 1 and Theorem 2 are proved from standard results under the assumptions
listed; the finite-sample statement in Section 3 is a heuristic and is labelled as one.
The file was revised on Nov 20 after a first outside review. It still needs a reader who
checks each step.

## 1. Setup

One run of an agent on one task has steps $k = 1, \dots, T$, a final outcome
$Y \in \{0, 1\}$, and a predictor that outputs $f_k \in [0, 1]$ after step $k$. Set
$f_T = Y$. Step $k$ costs $c_k \ge 0$. The history before step $k$ is
$h_{k-1} = (f_0, \dots, f_{k-1}, c_1, \dots, c_{k-1})$ plus anything else logged by then.

**Assumption A (the run does not depend on the stopping decision).**
- The whole run $(f_0, \dots, f_T, c_1, \dots, c_T)$, including the part that is never
  observed when the run is stopped, is defined whether or not we stop, and stopping does
  not change it.
- A stopping policy is a set of functions $P_k = \pi_k(h_{k-1})$ with
  $1 \ge P_1 \ge \dots \ge P_T \ge p_{\min} > 0$. Each $P_k$ is computed from the history
  alone. It does not use the random draw below.
- One uniform draw $U$ is independent of the whole run. The stop step is
  $s = \max\{k : U < P_k\}$ ($s = 0$ if there is none), and the run is paid for up to $s$.

Offline replay of logged runs satisfies this exactly. A live evaluation satisfies it as
long as the agent cannot observe the stopping machinery before it is stopped.

The estimate is

$$\hat Y = f_0 + \sum_{k=1}^{s} \frac{f_k - f_{k-1}}{P_k}.$$

Write $u_k = 1 / P_k$ with $u_0 = 1$, $w_k = u_k - u_{k-1} \ge 0$, and $e_k = Y - f_k$
(so $e_T = 0$).

**Lemma 1 (unbiased and bounded).** Under Assumption A, $E[\hat Y \mid \text{run}] = Y$
and $-(1/p_{\min} - 1) \le \hat Y \le 1/p_{\min}$.

*Proof.* Condition on the whole run. Every $P_k = \pi_k(h_{k-1})$ is then a fixed number,
and $U$ is still uniform because it is independent of the run, so $P(s \ge k \mid \text{run}) = P(U < P_k) = P_k$
(the $P_k$ are non-increasing, so $s \ge k$ exactly when $U < P_k$). Each term
$(f_k - f_{k-1}) \mathbf 1\{s \ge k\} / P_k$ therefore has conditional expectation
$f_k - f_{k-1}$ and the sum telescopes to $f_T = Y$. For the bound, sum by parts:
$\hat Y = f_s u_s - \sum_{k<s} f_k w_{k+1}$. Both pieces lie in $[0, u_s]$ and
$\sum_{k<s} w_{k+1} = u_s - 1$. $\square$

**Lemma 2 (variance).** Under Assumption A,
$\operatorname{Var}(\hat Y \mid \text{run}) = \sum_{k=1}^{T} w_k \, e_{k-1}^2 = \sum_{k=1}^{T} u_k (e_{k-1}^2 - e_k^2) - e_0^2.$

*Proof.* Let $Z_k = \mathbf 1\{s \ge k\} / P_k$, $Z_0 = 1$. Then
$\hat Y - Y = \sum_k (e_{k-1} - e_k)(Z_k - 1) = \sum_k e_{k-1} (Z_k - Z_{k-1})$ by summation
by parts. Given the run, $Z_k$ is a martingale in $k$, so its increments are uncorrelated
and $E[(Z_k - Z_{k-1})^2] = 1/P_k - 1/P_{k-1} = w_k$. The second form is another summation
by parts, using $u_0 = 1$ and $e_T = 0$. $\square$

The first form shows every term is non-negative: stopping adds variance only through
increases in $1/P$, each weighted by the squared error of the prediction in hand. The
second form is `stopping.exact_moments`.

## 2. The cost-optimal policy

**Problem.** Over a population of runs, minimize the compute needed for a fixed interval
width,

$$\big(\operatorname{Var}(Y) + v(\pi)\big) \cdot c(\pi), \qquad
v(\pi) = E[\operatorname{Var}(\hat Y \mid \text{run})], \quad c(\pi) = E\Big[\sum_k c_k P_k\Big].$$

**Lemma 3 (reduction).** If $\pi^*$ minimizes the product, it minimizes
$v(\pi) + \lambda \, c(\pi)$ for $\lambda = (\operatorname{Var}(Y) + v(\pi^*)) / c(\pi^*)$.

*Proof.* Write $A = \operatorname{Var}(Y) + v(\pi)$, $C = c(\pi)$, and $A^*, C^*$ for
$\pi^*$. For any $\pi$,
$A + \lambda C \ge 2\sqrt{\lambda A C} \ge 2\sqrt{\lambda A^* C^*} = 2A^* = A^* + \lambda C^*$. $\square$

So it is enough to solve the second problem for each $\lambda > 0$ and scan $\lambda$. In
the code the scan is over $\kappa = 1/\sqrt\lambda$.

**Assumption B (state).**
- The policy acts on a state $s_{k-1}$, a function of the history $h_{k-1}$, taking
  finitely many values, and $(s_k)$ is a Markov chain under the run's own law. By
  Assumption A that law does not depend on the policy.
- Given $s_{k-1}$, the pair $(e_{k-1}^2 - e_k^2, \; c_k)$ and the next state $s_k$ are
  independent of the earlier history.
- Every step has positive expected cost: $\bar c_k(s) > 0$ for every reachable state.

Taking the whole history as the state satisfies the first two points whenever the run
takes finitely many values. Define

$$a_k(s) = E[e_{k-1}^2 - e_k^2 \mid s_{k-1} = s], \qquad \bar c_k(s) = E[c_k \mid s_{k-1} = s].$$

For a stopping time $\tau \ge k$ of the state process (a rule that decides, after seeing
$s_k, \dots, s_j$, whether to look one step further), let

$$A_\tau(s) = E\big[e_{k-1}^2 - e_\tau^2 \mid s_{k-1} = s\big], \qquad
C_\tau(s) = E\Big[\sum_{j=k}^{\tau} c_j \mid s_{k-1} = s\Big] \ge \bar c_k(s) > 0.$$

**Theorem 1.** Under Assumptions A and B, define the index

$$\nu_k(s) = \max_{\tau \ge k} \frac{A_\tau(s)}{C_\tau(s)}.$$

The maximum is attained, and among policies that depend on the history through the state
and the current reach probability, a minimizer of $v(\pi) + \lambda c(\pi)$ is

$$P_k = \min\Big(P_{k-1},\; \operatorname{clip}\big(\sqrt{\nu_k(s_{k-1})^{+} / \lambda},\; p_{\min},\; 1\big)\Big), \qquad P_0 = 1.$$

*Proof.*

*Step 1: the objective.* By Lemma 2 and the tower property (each $u_k$ is a function of
$h_{k-1}$),

$$v(\pi) + \lambda c(\pi) = E\Big[\sum_{k=1}^{T} g_k(s_{k-1}, u_k)\Big] - E[e_0^2], \qquad
g_k(s, u) = u \, a_k(s) + \frac{\lambda \bar c_k(s)}{u}.$$

The constraint is $1 \le u_1 \le \dots \le u_T \le \bar u$ with $\bar u = 1/p_{\min}$.
Each $g_k(s, \cdot)$ is convex on $u > 0$ whatever the sign of $a_k(s)$: it is a linear
term plus a convex one.

*Step 2: dynamic programming.* The controlled state is $(s_{k-1}, u_{k-1})$, and the
horizon is finite, so backward induction gives an optimal policy. Set $W_{T+1} \equiv 0$
and, for $u \in [1, \bar u]$,

$$H_k(s, u) = g_k(s, u) + E\big[W_{k+1}(s_k, u) \mid s_{k-1} = s\big], \qquad
W_k(s, u) = \min_{u' \in [u, \bar u]} H_k(s, u').$$

*Claim.* For every $k$ and $s$, $H_k(s, \cdot)$ is convex and continuously differentiable
on $[1, \bar u]$, and $\partial_u W_k(s, u) = (\partial_u H_k(s, u))^+$.

It holds for $k = T$ since $H_T = g_T$. Suppose $H_k(s, \cdot)$ is convex and $C^1$ with
minimizer $u^*_k(s)$ on $[1, \bar u]$. Then $W_k(s, u) = H_k(s, \max(u, u^*_k(s)))$: it is
constant for $u \le u^*_k(s)$, where $\partial_u H_k \le 0$, and equals $H_k$ beyond,
where $\partial_u H_k \ge 0$. So $W_k(s, \cdot)$ is convex and $C^1$ with derivative
$(\partial_u H_k)^+$. $H_{k-1}$ is $g_{k-1}$ plus a finite average of such functions, so
it is convex and $C^1$ too. This also shows the optimal action is
$u_k = \max(u_{k-1}, u^*_k(s_{k-1}))$, which is the running minimum in the theorem.

Writing $M_k = \partial_u H_k$, the claim gives the recursion

$$M_k(s, u) = a_k(s) - \frac{\lambda \bar c_k(s)}{u^2} + E\big[(M_{k+1}(s_k, u))^+ \mid s_{k-1} = s\big], \qquad M_{T+1} \equiv 0.$$

*Step 3: the recursion is an optimal stopping problem.* Fix $u$ and let
$x_j = a_j(s_{j-1}) - \lambda \bar c_j(s_{j-1}) / u^2$. The recursion
$M_k = x_k + E[(M_{k+1})^+]$ is the backward induction for the problem "collect $x_k$,
then after each step choose whether to collect the next one". For a finite horizon its
value is attained by the rule $\tau^* = \min\{j \ge k : M_{j+1}(s_j, u) \le 0\}$, so

$$M_k(s, u) = \max_{\tau \ge k} E\Big[\sum_{j=k}^{\tau} x_j \mid s_{k-1} = s\Big]
= \max_{\tau \ge k} \Big(A_\tau(s) - \frac{\lambda}{u^2} C_\tau(s)\Big).$$

The last equality uses the Markov property to replace $a_j, \bar c_j$ by
$e_{j-1}^2 - e_j^2$ and $c_j$ inside the expectation; the sum of the first telescopes.
There are finitely many states and steps, hence finitely many stopping rules, so every
maximum here and in the definition of $\nu_k(s)$ is attained.

*Step 4: where the derivative changes sign.* Each $A_\tau(s) - \lambda C_\tau(s)/u^2$ is
continuous and strictly increasing in $u$ because $C_\tau(s) > 0$. A maximum of finitely
many such functions is continuous and strictly increasing, so $M_k(s, \cdot)$ has at most
one zero. Moreover

$$M_k(s, u) \ge 0 \iff \exists \tau : \frac{A_\tau(s)}{C_\tau(s)} \ge \frac{\lambda}{u^2}
\iff \nu_k(s) \ge \frac{\lambda}{u^2}.$$

If $\nu_k(s) \le 0$, then $M_k(s, u) < 0$ for every $u$, $H_k(s, \cdot)$ is decreasing, and
its minimizer on $[1, \bar u]$ is $\bar u$, that is $P = p_{\min}$. If $\nu_k(s) > 0$,
then $M_k$ is negative below $\sqrt{\lambda / \nu_k(s)}$ and non-negative above, so the
minimizer on $[1, \bar u]$ is $\sqrt{\lambda / \nu_k(s)}$ clipped to that interval. In
terms of $P = 1/u$ both cases read
$\operatorname{clip}(\sqrt{\nu_k(s)^+ / \lambda}, p_{\min}, 1)$. $\square$

**Scope.** The theorem is about the true conditional expectations for a Markov state. The
policy in `optimal_policy.py` estimates them on a coarse state (step, prediction decile)
that is not exactly Markov, so it is an approximation to the optimal policy, and it can
lose to a simpler rule on held-out data.

**What it says.**
- The optimal policy has the shape already used here: a target that depends on the
  state, followed down but never back up.
- **Learned tracking is the choice $\tau = T$** (always look ahead to the end of the run):
  $\sqrt{E[e_{k-1}^2 \mid s] / (\lambda E[\text{remaining cost} \mid s])}$. The index is a
  maximum over more rules, so the optimal target is at least the learned-tracking target,
  and they are equal whenever looking to the end is the best horizon.
- **The original tracking rule needs a strong calibration condition.** If
  $E[Y \mid s_{k-1}] = f_{k-1}$, that is, the predictor is calibrated given everything the
  policy's state contains, then $E[e_{k-1}^2 \mid s_{k-1}] = f_{k-1}(1 - f_{k-1})$, and
  learned tracking becomes $\kappa \sqrt{f(1 - f)}$ when the remaining cost is treated as
  constant. Ordinary calibration, $E[Y \mid f_{k-1}] = f_{k-1}$, is enough only when the
  state is the prediction alone.
- The index is high where a cheap step is about to teach the predictor a lot, even if the
  rest of the run is expensive and uninformative.
- A schedule that depends only on the step number is the special case with one state per
  step.

**Numerical checks.**
- *Exact small cases* (`tests/test_index_policy.py`). On trees of runs where the policy
  sees the full history, the index policy is compared against a direct convex
  optimization over every reach probability. They agree to six digits in 36 cases, and
  looking only to the end of the run is strictly worse in 10 of them. This checks the
  algebra; it is not a substitute for the proof above.
- *SWE-bench runs with the stand-in predictor* (`optimal_policy.py`). The index policy
  does not beat learned tracking: 54% against 52% median compute on held-out agents, and
  55.7% against 55.4% when fitted and scored on all agents together.
- *Controlled mid-run signal* (`midrun_signal.py`). The same logged runs and costs,
  but from a set point in each run the predictor is shown the true outcome with a given
  accuracy. Median compute on held-out agents, relative to full runs:

  | Signal arrives | Accuracy | Learned tracking | Index | Agents where index is better |
  | --- | --- | --- | --- | --- |
  | never | | 52% | 54% | 13 of 37 |
  | halfway | 0.9 | 53% | 53% | 21 of 37 |
  | a quarter in | 0.9 | 54% | 47% | 28 of 37 |
  | a tenth in | 0.9 | 54% | 42% | 33 of 37 |
  | a quarter in | 0.75 | 48% | 50% | 18 of 37 |

  The index policy gains when an accurate signal arrives early, and the gain grows the
  earlier it arrives. This supports the explanation for the tie on the stand-in predictor
  but does not prove it is the only cause. With a weaker signal the index policy is no
  better and slightly worse on the median, so its estimate on the coarse state is not
  reliably ahead of the simpler rule. The signal here is artificial; whether a real
  learned predictor produces one this strong and this early is an open question.

## 3. Validity of the confidence interval

Tasks are drawn independently from a population (or resampled from the benchmark), the
target is $\theta = E[Y]$, and the interval is
$\bar Y_n \pm 1.96 \, \hat\sigma_n / \sqrt n$ with $\bar Y_n$ and $\hat\sigma_n$ the mean
and standard deviation of the per-task estimates $\hat Y_1, \dots, \hat Y_n$.

**Theorem 2a (fixed predictor and policy).** The interval has asymptotic coverage 95%.

*Proof.* Assumption A holds for each task. By Lemma 1 the $\hat Y_i$ are independent,
identically distributed, bounded, and
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

**How large $n$ has to be: a heuristic, not a bound.** Theorems 2a and 2b are asymptotic
and say nothing about coverage at a given $n$. The following is a scaling argument for
one regime, offered as an explanation of the simulations and not proved as a coverage
guarantee.

For independent tasks (the setting of Theorem 2a), the Berry-Esseen theorem bounds the
error of the normal approximation to the standardized mean with known variance by a
constant times $E|\xi|^3 / (\sigma^3 \sqrt n)$. Now consider the regime where the
estimate is dominated by rare, heavily weighted runs: a run is finished with probability
about $p$, and when it is, its estimate is of size $1/p$. Then $E|\xi|^3 \sim p^{-2}$ and
$\sigma^3 \sim p^{-3/2}$, and the bound is of order

$$\frac{1}{\sqrt{n \, p}}.$$

In that regime what matters is $n \cdot p$, roughly the number of finished runs, not $n$.
With 500 tasks, $p = 0.05$ gives about 25 and $p = 0.2$ about 100.

What this does not establish:
- It is about the mean with known variance. The interval uses an estimated variance, and
  the skew that hurts the mean also makes the variance estimate and the mean dependent.
- It takes the worst case in which every run sits at the floor. A policy that keeps most
  runs well above the floor does better than the formula suggests.
- It is for independent tasks. No rate is shown for the corrected method of Theorem 2b.

**Numerical check** (`adaptive.py`, `results/adaptive_*.csv`).
- The largest bias over 37 agents is 0.6 points at floor 0.05 and 0.3 points or less at
  floors 0.1 and 0.2.
- The estimated variance of the mean is within 9% of its true variance for every agent
  and within 1% on average (`variance_ratio`), so the variance estimate is not the problem.
- The misses are one-sided (`miss_below`, `miss_above`). On the five agents checked at
  floor 0.05 the interval missed on one side about two to three times as often as on the
  other. That is the signature of skew from a few heavily weighted runs.
- Coverage of the corrected method rises with the floor, in the direction the heuristic
  suggests: 92.4% at 0.05, 94.0% at 0.1 and 94.6% at 0.2
  (the last two with the damped correction), against 94.8% for full runs.
- Hall's skewness-corrected interval was tried and did not help consistently, plausibly
  because the sample skewness is driven by the same few runs.
