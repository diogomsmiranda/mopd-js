# MOPD Equation 1 vs. PR #7215 Loss Normalization

## Scope

This note compares the trajectory-level normalization in Equation 1 of the MOPD paper with the token-level
normalization implemented in [TRL PR #7215](https://github.com/huggingface/trl/pull/7215).

The code references use the immutable PR head commit
[`69e9324a7dc39d8c3b80448dde79f748f572a616`](https://github.com/huggingface/trl/commit/69e9324a7dc39d8c3b80448dde79f748f572a616),
so the cited line numbers do not change when the PR receives new commits.

## MOPD Equation 1

The MOPD paper defines its reverse-KL objective as:

\[
\mathcal{L}_{\mathrm{rev\text{-}KL}}
=
\mathbb{E}_{x,\,y\sim\pi_\theta}
\left[
\frac{1}{|y|}
\sum_t
\sum_v
\pi_\theta(v)
\log
\frac{\pi_\theta(v)}{\pi_{\phi_d}(v)}
\right].
\]

Here:

- \(x\) is a prompt.
- \(y\sim\pi_\theta\) is a trajectory sampled from the student.
- \(|y|\) is the number of valid tokens in that trajectory.
- \(\pi_{\phi_d}\) is the domain teacher routed to the prompt.
- Both distributions at token \(t\) are conditioned on \((x,y_{<t})\).

### From the expectation to a finite batch

Equation 1 contains an expectation because, conceptually, training averages over prompts \(x\) and student-generated
trajectories \(y\sim\pi_\theta\). In practice, that expectation cannot be calculated over every possible prompt and
trajectory. A training batch provides a Monte Carlo estimate using \(B\) sampled prompt-trajectory pairs:

\[
(x_1,y_1), (x_2,y_2), \ldots, (x_B,y_B).
\]

The expectation does not become \(1/B\) exactly. Instead, it is estimated using the sample mean of these \(B\)
observations. More generally, if \(Z\) is a random variable and \(f(Z)\) is the quantity of interest, then samples
\(Z_1,\ldots,Z_B\) drawn from the distribution of \(Z\) give the Monte Carlo estimator:

\[
\mathbb{E}[f(Z)]
\;\approx\;
\frac{1}{B}\sum_{i=1}^{B}f(Z_i).
\]

The factor \(1/B\) appears because the empirical batch assigns equal weight to each of its \(B\) sampled observations.
This estimator is unbiased when the samples follow the distribution in the expectation:

\[
\mathbb{E}
\left[
\frac{1}{B}\sum_{i=1}^{B}f(Z_i)
\right]
=
\frac{1}{B}\sum_{i=1}^{B}\mathbb{E}[f(Z_i)]
=
\mathbb{E}[f(Z)].
\]

In MOPD, one observation \(Z_i\) is the complete prompt-trajectory pair \((x_i,y_i)\), not one token. Consequently,
the outer sample mean assigns weight \(1/B\) to each trajectory. Token averaging happens separately inside each
trajectory.

For one sampled trajectory \(y_i\), define its per-token divergence at position \(t\) as:

\[
d_{i,t}
=
D_{\mathrm{KL}}
\left(
\pi_\theta(\cdot\mid x_i,y_{i,<t})
\;\|\;
\pi_{\phi_{d_i}}(\cdot\mid x_i,y_{i,<t})
\right).
\]

The loss for that one trajectory is its mean divergence over its \(L_i=|y_i|\) valid tokens:

\[
\ell_i
=
\frac{1}{L_i}
\sum_{t=1}^{L_i} d_{i,t}.
\]

The outer expectation in Equation 1 is then estimated by averaging the \(B\) independently sampled trajectory
losses:

\[
\mathbb{E}_{x,\,y\sim\pi_\theta}[\ell(x,y)]
\;\approx\;
\frac{1}{B}\sum_{i=1}^{B}\ell_i.
\]

Substituting the definition of \(\ell_i\) gives the finite-batch estimator:

\[
\boxed{
\mathcal{L}_{\mathrm{paper}}
=
\frac{1}{B}
\sum_{i=1}^{B}
\left(
\frac{1}{L_i}
\sum_{t=1}^{L_i} d_{i,t}
\right)
}
\]

The approximation symbol is important: the batch average estimates the expectation using the trajectories sampled in
that batch. It is not an algebraic replacement of the expectation over every possible trajectory.

The finite-batch loss is therefore calculated in two stages:

1. Average the token divergences within each trajectory.
2. Average the resulting trajectory means across the batch.

Each trajectory has total weight \(1/B\), independently of its length.

The two denominators therefore have different roles:

- \(1/L_i\) averages the token divergences inside trajectory \(i\).
- \(1/B\) averages the \(B\) sampled trajectories used to estimate the outer expectation.

This is why the paper's finite-batch estimator uses \(B\), rather than the total token count \(\sum_i L_i\), as its
outer denominator.

### Plain-language interpretation

Equation 1 describes the expected loss over all possible prompt-trajectory pairs, rather than the loss of only one
specific trajectory. Inside that expectation, however, it defines how to calculate the loss of one sampled trajectory:

\[
\ell_i
=
\frac{1}{L_i}
\sum_{t=1}^{L_i}d_{i,t}.
\]

The token divergences of trajectory \(i\) are divided by its number of valid tokens \(L_i\). This gives a single mean
loss \(\ell_i\) for that trajectory, regardless of whether it contains 100 tokens or 2,000 tokens.

When training uses a batch containing \(B\) sampled trajectories, the expectation is estimated by weighting those
trajectory losses uniformly:

\[
\widehat{\mathcal L}_{\mathrm{paper}}
=
\frac{1}{B}
\sum_{i=1}^{B}\ell_i
=
\frac{1}{B}
\sum_{i=1}^{B}
\left(
\frac{1}{L_i}
\sum_{t=1}^{L_i}d_{i,t}
\right).
\]

Thus, each trajectory contributes total weight \(1/B\). Its length only determines how the weight assigned to that
trajectory is distributed among its tokens.

The PR code instead combines every valid token first and divides by the total valid-token count:

\[
\widehat{\mathcal L}_{\mathrm{code}}
=
\frac{
\sum_i\sum_{t=1}^{L_i}d_{i,t}
}{
\sum_iL_i
}.
\]

This uniformly weights tokens rather than trajectories. As a result, trajectory \(i\) receives total weight:

\[
w_i^{\mathrm{code}}
=
\frac{L_i}{\sum_jL_j},
\]

instead of the paper's uniform trajectory weight:

\[
w_i^{\mathrm{paper}}
=
\frac{1}{B}.
\]

Therefore, the interpretation is:

1. Equation 1 defines a length-normalized mean loss for each sampled trajectory.
2. A finite batch estimates the expectation by averaging the \(B\) trajectory means uniformly.
3. The PR does not perform that uniform trajectory average when trajectory lengths differ; it gives longer
   trajectories proportionally more influence because it averages all valid tokens together.
4. If every trajectory has the same length, both formulations are equivalent.

## PR #7215 Implementation

### Counting valid tokens

After generation, the PR constructs a completion loss mask and sums it across the distributed processes:

[`trl/trainer/distillation_trainer.py:1901-1902`](https://github.com/cmpatino/trl/blob/69e9324a7dc39d8c3b80448dde79f748f572a616/trl/trainer/distillation_trainer.py#L1901-L1902)

```python
loss_mask = completion_mask if tool_mask is None else completion_mask * tool_mask
num_items_in_batch = self.accelerator.gather(loss_mask.sum()).sum()
```

`loss_mask.sum()` counts valid completion tokens, not trajectories. After the distributed gather,
`num_items_in_batch` is the number of valid completion tokens in the global effective batch.

### Compensating for DDP gradient averaging

During training, the global valid-token count is divided by the number of processes:

[`trl/trainer/distillation_trainer.py:2111-2117`](https://github.com/cmpatino/trl/blob/69e9324a7dc39d8c3b80448dde79f748f572a616/trl/trainer/distillation_trainer.py#L2111-L2117)

```python
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    # transformers computes `num_items_in_batch` from the raw dataloader labels, before on-policy generation
    # replaces the completions; use the count over the generated completions instead (computed in
    # `_generate_and_score_completions`). Divide by the process count so the per-process loss compensates for DDP
    # gradient averaging.
    if self.model.training and inputs.get("num_items_in_batch") is not None:
        num_items_in_batch = inputs["num_items_in_batch"].clamp(min=1.0) / self.accelerator.num_processes
```

This division correctly compensates for DDP averaging. It does not change the type of reduction: the denominator is
still based on the number of valid tokens rather than the number of trajectories.

### Sharing one denominator across teacher groups

The multi-teacher loss splits a microbatch into groups according to the routed teacher. Every group is given the same
valid-token denominator:

[`trl/trainer/distillation_trainer.py:2225-2230`](https://github.com/cmpatino/trl/blob/69e9324a7dc39d8c3b80448dde79f748f572a616/trl/trainer/distillation_trainer.py#L2225-L2230)

```python
# The group losses are summed, so they must share one denominator. `num_items_in_batch` is already one;
# where it is `None` (evaluation) the loss would otherwise normalize each group by its own valid-token
# count, and the sum of the group means would scale with the number of teachers in the microbatch. The
# count matches the one the loss computes from the same mask, so a single group reduces to the legacy
# `mean` exactly.
denominator = loss_mask.sum().clamp(min=1) if num_items_in_batch is None else num_items_in_batch
```

This is necessary for the chosen global token-mean objective. Without a shared denominator, summing one independently
normalized loss per teacher would incorrectly scale the result with the number of teacher groups.

### Passing the denominator into each teacher-group loss

Each teacher group receives that shared denominator:

[`trl/trainer/distillation_trainer.py:2249-2257`](https://github.com/cmpatino/trl/blob/69e9324a7dc39d8c3b80448dde79f748f572a616/trl/trainer/distillation_trainer.py#L2249-L2257)

```python
group_loss, group_entropy_sum, group_n_valid = _chunked_divergence_loss(
    student_hidden_states[rows],
    targets.to(student_hidden_states.device),
    student_lm_head_weight,
    teacher_lm_head_weight,
    loss_mask[rows],
    self.beta,
    _CHUNKED_LM_HEAD_CHUNK_SIZE,
    num_items_in_batch=denominator,
```

The normalized group losses are then summed:

[`trl/trainer/distillation_trainer.py:2266`](https://github.com/cmpatino/trl/blob/69e9324a7dc39d8c3b80448dde79f748f572a616/trl/trainer/distillation_trainer.py#L2266)

```python
loss = loss + group_loss
```

Because all groups use the same denominator, this sum is equivalent to summing the divergence over every valid token
and dividing once by the total number of valid tokens.

### Final division by the valid-token count

The chunked divergence function performs the actual normalization:

[`trl/trainer/distillation_trainer.py:287-293`](https://github.com/cmpatino/trl/blob/69e9324a7dc39d8c3b80448dde79f748f572a616/trl/trainer/distillation_trainer.py#L287-L293)

```python
if num_items_in_batch is None:
    # Clamped for the same reason: a fully-masked rank reduces to a finite zero rather than `0 / 0`.
    loss = loss / n_valid_tensor.clamp(min=1)
else:
    if isinstance(num_items_in_batch, torch.Tensor):
        num_items_in_batch = num_items_in_batch.to(loss.device)
    loss = loss / num_items_in_batch
```

The function's docstring also states that `num_items_in_batch` is the total number of valid tokens:

[`trl/trainer/distillation_trainer.py:205-208`](https://github.com/cmpatino/trl/blob/69e9324a7dc39d8c3b80448dde79f748f572a616/trl/trainer/distillation_trainer.py#L205-L208)

```python
num_items_in_batch (`torch.Tensor` or `int`, *optional*):
    Total number of valid tokens across the global batch. When provided, the loss is reduced as `sum /
    num_items_in_batch` (gradient-accumulation-correct); when `None`, reduction is `mean` over local valid
    positions.
```

## Objective Implemented By The Code

Let \(m_{i,t}\in\{0,1\}\) indicate whether token \(t\) of trajectory \(i\) is included in the loss. This mask excludes
padding and can also exclude tool-output tokens.

The code implements:

\[
\boxed{
\mathcal{L}_{\mathrm{code}}
=
\frac{
\sum_{i=1}^{B}\sum_t m_{i,t}d_{i,t}
}{
\sum_{i=1}^{B}\sum_t m_{i,t}
}
}
\]

If \(L_i=\sum_t m_{i,t}\) is the number of valid tokens in trajectory \(i\), this can be rewritten as:

\[
\mathcal{L}_{\mathrm{code}}
=
\sum_{i=1}^{B}
\frac{L_i}{\sum_j L_j}
\left(
\frac{1}{L_i}\sum_t m_{i,t}d_{i,t}
\right).
\]

Here, \(j\) indexes every trajectory in the batch, so \(L_j\) is the number of valid tokens in trajectory \(j\) and
\(\sum_j L_j\) is the total number of valid tokens across the batch.

The trajectory-level weight is therefore:

\[
w_i^{\mathrm{code}}=\frac{L_i}{\sum_j L_j},
\]

whereas Equation 1 gives every trajectory:

\[
w_i^{\mathrm{paper}}=\frac{1}{B}.
\]

## Worked Example

Consider two trajectories:

| Trajectory | Valid tokens | Mean token divergence |
|------------|-------------:|----------------------:|
| A | 2 | 1.0 |
| B | 8 | 0.1 |

For the paper's per-trajectory reduction:

\[
\mathcal{L}_{\mathrm{paper}}
=
\frac{1}{2}(1.0+0.1)
=
0.55.
\]

For the code's global token reduction:

\[
\mathcal{L}_{\mathrm{code}}
=
\frac{2\times1.0+8\times0.1}{2+8}
=
0.28.
\]

The two reductions produce different losses and different gradients. In the code reduction, trajectory B receives
four times the total weight of trajectory A because it contains four times as many valid tokens.

## When The Reductions Are Equivalent

If every trajectory has the same valid length \(L\), then:

\[
\frac{L_i}{\sum_jL_j}
=
\frac{L}{BL}
=
\frac{1}{B},
\]

and therefore:

\[
\mathcal{L}_{\mathrm{code}}=\mathcal{L}_{\mathrm{paper}}.
\]

The difference only becomes observable when valid trajectory lengths differ.

## Why This Matters For Multi-Domain MOPD

MOPD routes different domains to different teachers. Those domains can have systematically different completion
lengths, for example:

| Domain | Illustrative completion length |
|--------|-------------------------------:|
| Instruction following | 100 tokens |
| Mathematical reasoning | 500 tokens |
| Software engineering | 2,000 tokens |

Under Equation 1, one sampled trajectory from each domain has the same total weight before considering the magnitude
of its divergence. The prompt-sampling ratio therefore directly determines the expected domain mixture.

Under the code's token mean, a 2,000-token software-engineering trajectory can contribute approximately twenty times
the total gradient weight of a 100-token instruction-following trajectory. Consequently, a nominal sampling ratio
such as Math : IF : SWE = 0.35 : 0.35 : 0.30 does not necessarily remain the effective optimization ratio.

This does not automatically make token averaging an invalid design. Token-level means are common in language-model
training, and the implementation carefully makes that reduction consistent across teacher groups, gradient
accumulation, and distributed processes. It is nevertheless different from the per-trajectory normalization written
in MOPD Equation 1.

## Suggested Review Question

> MOPD Eq. 1 first averages the divergence over the valid tokens of each trajectory and then averages the trajectory
> means, giving every trajectory equal total weight. The current implementation appears to sum divergence over all
> valid completion tokens and divide by the global valid-token count, giving each trajectory weight proportional to
> its length. Is this token-weighted reduction intentional to preserve the existing `DistillationTrainer` semantics,
> or should the multi-teacher path use per-trajectory normalization to match Eq. 1?

## References

- MOPD paper: [MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training](https://huggingface.co/papers/2606.30406)
- Pull request: [huggingface/trl#7215](https://github.com/huggingface/trl/pull/7215)
- Reviewed PR commit: [`69e9324a7dc39d8c3b80448dde79f748f572a616`](https://github.com/huggingface/trl/commit/69e9324a7dc39d8c3b80448dde79f748f572a616)
