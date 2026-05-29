# CNN-Based ADC Linearizer

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deijany/CNN_linearizer/blob/main/main_nnlinearizer.ipynb)  [![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/deijany/CNN_linearizer/blob/main/main_nnlinearizer.ipynb)

This repository contains a CNN-based linearizer for ADC frequency dependent nonlinearity mitigation
, developed in 2022 during my PhD.

The target application is real-time signal processing. Hardware constraints on FPGAs and ASICs make every additional layer, neuron, or connection costly in latency and power, so minimizing complexity is a first-class design requirement, not an afterthought.

Inspired by Hammerstein and Wiener structures and by how neural networks parametrize nonlinear functions, I designed a parallel architecture of low-complexity nonlinear branches: each branch applies a cheap activation function (ReLU or absolute value), followed by an FIR filter, and one linear FIR branch handles any residual linear distortion. The branches are summed to produce the linearized estimate. The entire model has **36 trainable parameters** in the default configuration (N=8 branches, K=2 FIR order) -- scaling linearly with both.

Training uses a two-stage strategy: Adam from multiple initialization points (basin exploration) followed by BFGS fine-tuning on every solution (precise convergence). Both stages run in parallel via `ThreadPoolExecutor` -- Adam dispatches across all available GPUs on CUDA platforms, or across CPU cores on Apple Silicon (where the Metal backend serializes GPU ops). BFGS always runs on CPU: `scipy.minimize` makes ~1500 objective calls per initialization, each requiring a Python--GPU roundtrip (~1 ms). At 36 parameters, the actual forward+backward pass takes microseconds -- the sync overhead alone exceeds the compute cost, so CPU is faster on every platform.

Working on the CNN-based training of this structure led to developing a better approach: decomposing the problem into a set of subconvex sub-problems, each solvable analytically by matrix inversion. Nonconvex optimization is avoided. That approach converges faster, is more interpretable, and achieves higher SNDR. It was published in 2025.


## Published Paper

Low-Complexity Frequency-Dependent Linearizers Based on Parallel Bias-Modulus and Bias-ReLU Operations

- IEEE Access (open access): https://ieeexplore.ieee.org/document/11293818
- arXiv: https://arxiv.org/abs/2412.16210


## Repository Structure

```
CNN_linearizer/
|
|-- main_nnlinearizer.ipynb              CNN-based linearizer (this notebook)
|-- proposed_linearizer_in_IEEA_access2025.py  Matrix-inversion linearizer (paper)
|-- hammerstein_linearizer.py            Hammerstein baseline (polynomial branches)
|-- requirements.txt
|-- datasets/                            Input data (small dataset): frequency dependent signals (distortion order 2) and 9 nonlinear terms (polynomial degrees 2 to 10)
|-- datasets/nonlinear_coeff.h5 (.txt)   Coefficients used to generate the dataset      
|-- trained_model/                       Saved model outputs
|
|-- myclasses/
|   |-- linearizers_v19.py               MatrixInversionLinearizer, ActivationFunctions
|   |-- file_manipulation.py             PathManager, DataSetLoader
|   |-- env_utils.py                     is_colab(), is_macos(), get_cpu_info()
|   |-- system_check.py                  SystemCheck (TF + GPU + RAM summary)
|   `-- callbacks.py                     UniversalPrintCallback (Jupyter + terminal)
|
`-- myfunctions/
    `-- functions_helper.py              compute_statistics(), SpectrumAnalyzer,
                                         storate_dictionary(), load_dictionary()
```


## Background

### ADC nonlinearity

An ideal ADC maps an analog voltage to a digital code linearly. Real ADCs deviate from this: the transfer curve has a nonlinear component that distorts the output, degrading the signal-to-noise-and-distortion ratio (SNDR). A linearizer is a post-processing block that suppress this distortion.

### Where the nonlinearity occurs

The paper treats two cases:

- **Digital-domain model** -- the nonlinearity acts on the already-sampled signal. Distortion products remain within the Nyquist band, so the linearizer can operate directly at the sampling rate. This is the case implemented in this project.

- **Analog-domain model** -- the nonlinearity acts on the analog waveform before sampling. Harmonics and intermodulation products are not bandlimited and extend beyond the signal band; a proper linearizer then requires interpolation, processing at a higher rate, and downsampling at the output. This case is covered in the paper but not included in this sample project.

### Why cheap activation functions

In a real-time FPGA or ASIC implementation, the activation function is evaluated at every sample and for every branch. Polynomial activations require multipliers that scale badly. ReLU (a comparator and a gate) and absolute value (a sign flip) are implementable with minimal logic and negligible latency.


## Setup

```bash
conda create -n linearizer python=3.11
conda activate linearizer
pip install -r requirements.txt

# Apple Silicon only (Metal GPU)
pip install tensorflow-macos tensorflow-metal
```

The notebook runs on any platform -- CUDA GPUs, Apple Silicon (Metal), or CPU-only.
All three paths are handled automatically; no code changes needed.

Tested on Python 3.11, TensorFlow 2.16, NumPy 1.26, macOS M4 Pro.
Also compatible with Google Colab and Kaggle (TensorFlow 2.19+).


## Results

Default configuration: N=8 nonlinear branches, K=2 (3-tap FIR), 20 initialization points, 20 Adam epochs each, BFGS fine-tuning per run.

| Signal | SNDR (test mean) |
|--------|-----------------|
| Distorted ADC output V | 30.54 dB |
| Best Adam only | ~40 dB |
| **CNN + BFGS** | **54 dB** |
| Clean reference X | 67.04 dB |

The linearizer achieves **+23 dB of improvement** over the distorted signal using 36 trainable parameters, with no feedback and no look-ahead.

> The distortion model has 9 nonlinear branches. Using `BRANCH_NUMBER = 8` (default) leaves a structural mismatch; increasing `BRANCH_NUMBER` and `NUM_RESTARTS` closes the remaining gap.


## Citation

```bibtex
@article{rodriguez2025lowcomplexity,
  title   = {Low-Complexity Frequency-Dependent Linearizers Based on Parallel Bias-Modulus and Bias-ReLU Operations},
  author  = {Rodr{\'i}guez Linares, Deijany and Johansson, H{\aa}kan},
  journal = {IEEE Access},
  year    = {2025},
  doi     = {10.1109/ACCESS.2025.3642613},
  url     = {https://ieeexplore.ieee.org/document/11293818}
}
```


## License

MIT License. See [LICENSE](LICENSE) for details.
