# Responsible AI Notes

## What this model is

A decision-support signal that estimates the probability a customer will
churn, based on account attributes, to help a retention team prioritize
outreach.

## What this model is not

- Not an automated decision system. A `high_risk` label should route to
  human review, never directly trigger an account action, credit
  decision, or differential pricing.
- Not a causal explanation. A high churn probability does not mean a
  specific feature *caused* the customer to be at risk — it means the
  pattern of their attributes resembles customers who historically left.
- Not validated for any bank other than the one (synthetic/illustrative)
  dataset it was trained on.

## Fairness

`Geography` and `Gender` are included as model inputs because they carry
real signal in this dataset (see the EDA notebook). This is a deliberate
choice, not an oversight — but it means:

- Churn *probabilities* will differ systematically by geography and
  gender, because the underlying churn *rates* differ in the training
  data.
- If this model's output were ever used to decide who receives retention
  offers, better rates, or other differential treatment, that use should
  be reviewed for disparate-impact risk before deployment — a group
  churning more often does not automatically justify treating individuals
  in that group differently.
- A fairness-conscious follow-up would compare recall and false-negative
  rates across Geography/Gender subgroups (not done in this project's
  scope — see `docs/PROJECT_BRIEF.md` non-goals) before any operational
  rollout.

## Human review requirement

Every prediction from `predict_churn_risk()` includes a plain-language
message, not just a label, so a reviewer sees context rather than a bare
number. The most important sentence for any presentation of this project:
**this model is an early support signal, not an automatic decision.**

## Data limitations

The dataset is small (10,000 rows), public, and commonly used for
tutorials — it may not represent any specific real bank's customer
population. Retraining and re-validating on an institution's own data
would be required before any real deployment.
