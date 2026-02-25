# PSSig
For an $n \times n$ self-adjoint matrix $M \in \mathbb{C}^{n\times n}$, and signals $f_1,\ldots, f_m \in \mathbb{C}^n$, compute the power spectrum signature of each signal. For distinct, real eigenvalues $\lambda_1 < \cdots < \lambda_L$, the power spectrum signature is a measure

$$ \mu_f = \sum_{i=1}^L \|P_i f\|^2 \delta_{\lambda_i} $$

where $P_i \in \mathbb{C}^{n \times n}$ is the projection matrix onto the eigenspace corresponding to eigenvalue $\lambda_i$. 

See our paper [Power Spectrum Signatures of Graphs](https://arxiv.org/html/2503.09660v2) for further details.


## Basic Usage

Recommend using `uv` package and project manager.

Install package first by running `uv pip install .` in top directory before running `examples/basic_example.ipynb`. 
