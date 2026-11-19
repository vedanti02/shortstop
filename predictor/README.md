# predictor (Deepika)

The learned step predictor, trained to minimize the estimator variance per unit of compute.

- Data: CoderForge and Nebius OpenHands trajectories
- Stand-in: fixed stopping rule (continue with probability 0.5 per step)
- Done when: variance reduction over LightGBM is shown on held-out agents
