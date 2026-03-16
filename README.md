# Comparative Analysis of Turbo and LDPC Codes

This repository contains a **theory + simulation + tutorial** study of **Turbo codes** and **Low-Density Parity-Check (LDPC) codes** over an **AWGN channel** with **BPSK modulation**.

The project combines:

- mathematical background and article-style explanations
- Python implementations of Turbo and LDPC encoding/decoding
- BER and convergence experiments
- runtime and throughput comparison
- tutorial notebooks for understanding and reproducing the code

## Project goals

The repository is organized around three main comparison questions:

1. **Error-rate performance**  
   How do Turbo and LDPC codes compare in BER across different code rates and iteration counts?

2. **Iterative decoder behavior**  
   How do the soft outputs evolve over iterations for each decoder?

3. **Computational cost**  
   What runtime and throughput tradeoffs appear as the maximum number of decoder iterations increases?

## Repository structure

```text
turbo-vs-ldpc-analysis/
├── figures/                         # Plots used in the article
├── ldpc/                            # LDPC encoder, decoder, config, and simulation code
├── turbo/                           # Turbo encoder, decoder, config, and simulation code
├── tutorials/                       # Jupyter tutorial notebooks
├── turbo-ldpc-article.ipynb         # Main article notebook
├── turbo-vs-ldpc.py                 # Combined Turbo-vs-LDPC comparison script
├── README.md
└── requirements.txt
```

## Main components

### `turbo/`
Contains the Turbo-code implementation, including:

- configuration values
- recursive systematic convolutional (RSC) encoding
- iterative Turbo decoding
- convolutional baseline comparison
- simulation utilities

### `ldpc/`
Contains the LDPC implementation, including:

- sparse parity-check construction
- RA-style LDPC encoding
- normalized min-sum decoding
- simulation utilities

### `turbo-vs-ldpc.py`
A compact comparison script that:

- runs one worked Turbo example
- runs one worked LDPC example
- generates smooth comparison BER curves
- benchmarks runtime and throughput
- produces BER, convergence, and decoder-cost plots

### `turbo-ldpc-article.ipynb`
The main notebook for the article/report, including:

- theory sections
- formulas
- figure explanations
- comparison discussion
- references

### `tutorials/`
Contains step-by-step notebooks for:

- Turbo-code explanation
- LDPC-code explanation
- Turbo-vs-LDPC comparison explanation

## Methods and assumptions

The experiments and tutorial examples are based on:

- **Channel model:** Additive White Gaussian Noise (AWGN)
- **Modulation:** Binary Phase Shift Keying (BPSK)
- **Decoding:** iterative soft decoding
- **Comparison metrics:**
  - Bit Error Rate (BER)
  - soft-output convergence
  - decoder runtime
  - decoder throughput

The code supports multiple code rates, including:

- `1/3`
- `1/2`
- `3/4`
- `7/8`

## Installation

Create and activate a virtual environment if you want an isolated setup.

### Option 1: virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

## How to run

### Run the article notebook

```bash
jupyter lab
```

Then open:

```text
turbo-ldpc-article.ipynb
```

### Run the comparison script

```bash
python turbo-vs-ldpc.py
```

### Open the tutorials

Launch JupyterLab and open the notebooks inside:

```text
tutorials/
```

## Outputs

Depending on the script or notebook you run, the repository can generate:

- BER-vs-SNR plots
- iteration/convergence plots
- runtime plots
- throughput plots
- comparison figures for Turbo vs LDPC

Many generated figures are already stored in `figures/`.

## Notes

- The repository is optimized for **clarity and tutorial use**, not only for maximum simulation scale.
- Some comparison curves are intentionally kept smooth and presentation-friendly in tutorial material.
- The code is structured so that iteration counts and comparison settings are easy to modify.

## Dependencies

The project uses standard scientific Python packages:

- NumPy
- SciPy
- Matplotlib
- Pandas
- JupyterLab / Notebook

Install them with:

```bash
pip install -r requirements.txt
```
