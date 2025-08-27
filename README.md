# Block-MIMO-Solver
Project started at Redwood Center for Theoretical Neuroscience

This repo contains two Jupyter notebooks that implement a gradient-flow-based detector
for QPSK over complex MIMO channels. The detector minimizes a loss combining
data-fit, constant-modulus, and phase-bias terms, and integrates the resulting drift
using [JAX](https://github.com/google/jax) and [Diffrax](https://github.com/patrick-kidger/diffrax).

We compare against **Zero-Forcing (ZF)** and **Linear MMSE (LMMSE)** baselines and
report SER/BER and latency.

---

## Contents

- `notebooks/01_visual_flow.ipynb`  
  A slower, *visual* notebook that uses adaptive step size. It shows constellation
  trajectories so readers can see how the flow solver behaves.

- `notebooks/02_benchmark_ser_ber_latency.ipynb`  
  A faster notebook designed for benchmarking. It measures symbol error rate (SER),
  bit error rate (BER), and latency, and compares results to ZF/LMMSE.

---

## Quick Start

1. **Clone the repo**
   ```bash
   git clone https://github.com/matthewymoon/Block-MIMO-Solver.git
   cd mimo-qpsk-flow
