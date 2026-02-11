[![DOI](https://zenodo.org/badge/241174650.svg)](https://zenodo.org/badge/latestdoi/241174650)
[![License](https://img.shields.io/badge/License-GPL--3.0-brightgreen)](https://github.com/PTRRupprecht/CascadeTorch/tree/master/LICENSE)
[![Size](https://img.shields.io/github/repo-size/PTRRupprecht/CascadeTorch?style=plastic)](https://img.shields.io/github/repo-size/PTRRupprecht/CascadeTorch?style=plastic)
[![Language](https://img.shields.io/github/languages/top/PTRRupprecht/CascadeTorch?style=plastic)](https://github.com/PTRRupprecht/CascadeTorch)

## CascadeTorch: Calibrated spike inference from calcium imaging data using PycTorch

<!---![Concept of supervised inference of spiking activity from calcium imaging data using deep networks](https://github.com/PTRRupprecht/CascadeTorch/tree/master/etc/Figure%20concept.png)--->
<p align="center"><img src="https://github.com/PTRRupprecht/CascadeTorch/tree/master/etc/CA1_deconvolution_CASCADE.gif "  width="85%"></p>

*Cascade* translates calcium imaging ΔF/F traces into spiking probabilities or discrete spikes.

*Cascade* is described in detail in **[the main paper](https://www.nature.com/articles/s41593-021-00895-5)**. There are follow-up papers which describe the application of Cascade to **[spinal cord data](https://www.biorxiv.org/content/10.1101/2024.07.17.603957)** and the application of Cascade to **[GCaMP8](https://www.biorxiv.org/content/10.1101/2025.03.03.641129)**.

*Cascade's* toolbox consists of

- A large and continuously updated ground truth database spanning brain regions, calcium indicators, species
- A deep network that is trained to predict spike rates from calcium data
- Procedures to resample the training ground truth such that noise levels and frame rates of calcium recordings are matched
- A large set of pre-trained deep networks for various conditions (additional models upon request)
- Tools to quantify the out-of-dataset generalization for a given model and noise level
- A tool to transform inferred spike rates into discrete spikes

Get started quickly with the following two *Colaboratory Notebooks*:

## [Spike inference from calcium data](https://colab.research.google.com/github/PTRRupprecht/CascadeTorch/blob/master/Demo%20scripts/Calibrated_spike_inference_with_Cascade.ipynb)

Upload your calcium data, use Cascade to process the data, download the inferred spike rates.

Spike inference with Cascade improves the temporal resolution, denoises the recording and provides an absolute spike rate estimate.

No parameter tuning, no installation required.

You will get started within few minutes.

[Spike inference from calcium data](https://colab.research.google.com/github/PTRRupprecht/CascadeTorch/blob/master/Demo%20scripts/Calibrated_spike_inference_with_Cascade.ipynb)

<p align="center">
<a href="https://colab.research.google.com/github/PTRRupprecht/CascadeTorch/blob/master/Demo%20scripts/Calibrated_spike_inference_with_Cascade.ipynb" rel="Spike inference from calcium data, showing activations of intermediate network layers"><img src="https://github.com/HelmchenLabSoftware/Calibrated-inference-of-spiking/blob/master/etc/Network_activations_output.gif "  width="85%"></a>
</p>

## Getting started

#### Without installation

If you want to try out the algorithm, just open **[this online Colaboratory Notebook](https://colab.research.google.com/github/PTRRupprecht/CascadeTorch/blob/master/Demo%20scripts/Calibrated_spike_inference_with_Cascade.ipynb)**. With the Notebook, you can apply the algorithm to existing test datasets, or you can apply **pre-trained models** to **your own data**. No installation will be required since the entire algorithm runs in the cloud (Colaboratory Notebook hosted by Google servers; a Google account is required). The entire Notebook is designed to be used by researchers with little background in Python, but it is also the best starting point for experienced programmers. Try it out - within a couple of minutes, you can start using the algorithm!

#### With a local installation (Ubuntu/Windows)

If you want to modify the code, if you want to integrate the algorithm into your existing pipeline (e.g., with CaImAn or Suite2P), or if you want to train your own networks, an installation on your local machine is necessary. Important: Although *Cascade* is based on deep networks, **GPU-support is not necessary**. Training of models for Cascade runs smoothly without (albeit GPUs speed up the process). Therefore, the installation is much easier than for typical deep learning-based toolboxes that require GPU-based processing.

Details on the package versions will be noted later. For the moment, Python with a version of 3.10 or newer and Torch version 2.5.1 works well in the Colab Cloud and on Windows. Compatibility with newer versions of Torch, Python, and other operating systems (Ubuntu, macOS) will be evaluated during the next months. Feedback (also positive feedback about working environments) is very welcome and can be submitted in form of issues or pull requests.

### Updates, FAQs, further info:

Check the parent [CASCADE repository](https://github.com/HelmchenLabSoftware/Cascade). FAQs and updates are updated only there for simplicity.

### References


> Please cite the [paper](https://www.nature.com/articles/s41593-021-00895-5) as primary reference for Cascade:
>
> Rupprecht P, Carta S, Hoffmann A, Echizen M, Blot A, Kwan AC, Dan Y, Hofer SB, Kitamura K, Helmchen F\*, Friedrich RW\*, *A database and deep learning toolbox for noise-optimized, generalized spike inference from calcium imaging*, Nature Neuroscience (2021).
>
> (\* = co-senior authors)
>
> And the following papers specific for models trained with GCaMP8 and spinal cord data, respectively:
>
> Rupprecht P, Rózsa M, Fang X, Svoboda K, Helmchen F. *[Spike inference from calcium imaging data acquired with GCaMP8 indicators](https://www.biorxiv.org/content/10.1101/2025.03.03.641129)*, bioRxiv (2025).
>
> Rupprecht P, Fan W, Sullivan S, Helmchen F, Sdrulla A. *[Spike rate inference from mouse spinal cord calcium imaging data](https://www.jneurosci.org/content/45/18/e1187242025)*, J Neuroscience (2025).
