# CNN-Based ADC Linearizer

This repository contains a CNN-based linearizer for ADC nonlinearity correction, developed in 2022 during my PhD.

The target application is real-time signal processing. Hardware constraints on FPGAs and ASICs make every additional layer, neuron, or connection costly in latency and power, so minimizing complexity is a first-class design requirement, not an afterthought.

Inspired by Hammerstein and Wiener structures and by how neural networks parametrize nonlinear functions, I designed a parallel architecture of low-complexity nonlinear branches: each branch applies a cheap activation function, such as ReLU, absolute value, or 1-bit quantization (sign), followed by an FIR filter, and one linear FIR branch handles any residual linear distortion. The branches are summed to produce the linearized estimate.

Working on the CNN-based training of this structure led to developing a better approach: decomposing the problem into a set of subconvex sub-problems, each solvable analytically by matrix inversion. No gradient descent is required. That approach converges faster, is more interpretable, and achieves higher SNDR. It was published in 2025.


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
|-- datasets/                            Input data (not tracked in git)
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

An ideal ADC maps an analog voltage to a digital code linearly. Real ADCs deviate from this: the transfer curve has a nonlinear component that distorts the output, degrading the signal-to-noise-and-distortion ratio (SNDR). A linearizer is a post-processing filter that inverts this distortion.

### Where the nonlinearity occurs

The paper treats two cases:

- Digital-domain model: the nonlinearity acts on the already-sampled signal. Distortion products stay within the Nyquist band and no interpolation is needed. This is the case implemented in this repository.
- Analog-domain model: the nonlinearity acts on the analog waveform before sampling. Harmonics and intermodulation products are not bandlimited; a complete linearizer must upsample, process at a higher rate, and then downsample. This case is covered in the paper but not included in this sample project.

### Why cheap activation functions

In a real-time FPGA or ASIC implementation, the activation function is evaluated at every sample and for every branch. Polynomial activations require multipliers that scale badly. ReLU (a comparator and a gate), absolute value (a sign flip), and 1-bit quantization (a single comparator) are all implementable with minimal logic and negligible latency.


## Setup

```bash
# conda
conda create -n linearizer python=3.11
conda activate linearizer
pip install -r requirements.txt

# Apple Silicon (Metal GPU)
pip install tensorflow-macos tensorflow-metal
```

Tested on Python 3.11, TensorFlow 2.16, NumPy 1.26, macOS M4 Pro.
The notebook is also compatible with Google Colab.


## Citation

```bibtex
@article{rodriguez2025lowcomplexity,
  title   = {Low-Complexity Frequency-Dependent Linearizers Based on
             Parallel Bias-Modulus and Bias-ReLU Operations},
  author  = {Rodr{\'i}guez Linares, Deijany and others},
  journal = {IEEE Access},
  year    = {2025},
  url     = {https://ieeexplore.ieee.org/document/11293818}
}
```


## License

MIT License. See [LICENSE](LICENSE) for details.
