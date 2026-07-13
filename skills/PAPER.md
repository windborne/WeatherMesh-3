# WeatherMesh-3: Paper Reference

## Document purpose

This document is a concise, evidence-bound reference for **“WeatherMesh-3: Fast and Accurate Operational Global Weather Forecasting”** by Du et al., published as an ICLR 2025 workshop paper.

It describes the forecasting model and reported results rather than an application implementation. Statements marked as **reported** are claims made by the authors and should not be interpreted as independently reproduced results. Details not supplied by the paper are explicitly identified rather than inferred.

Primary source:

---

## 1. Research objective

WeatherMesh-3, or **WM-3**, is a transformer-based system for deterministic, global weather forecasting.

The paper targets three operational limitations of existing machine-learning weather-prediction systems:

1. Forecast accuracy relative to real operational numerical weather prediction systems.
2. Excessive spatial blurring in machine-learned forecasts.
3. Hardware, memory, and operational complexity that limit deployment.

Its two principal architectural contributions are:

* **Latent rollout:** the atmospheric state remains in latent space while it is advanced through time, avoiding repeated decoding to and re-encoding from the physical grid.
* **Modular mixed-horizon architecture:** separate one-hour and six-hour processors operate in a common latent space, while multiple encoders can incorporate different operational analyses into a blended initial condition.

The resulting model accepts one global weather state and predicts the global state at an arbitrary integer-hour lead time.

---

## 2. System summary

| Property                             | Paper specification                                                      |
| ------------------------------------ | ------------------------------------------------------------------------ |
| Forecast type                        | Deterministic global weather forecast                                    |
| Physical resolution                  | 0.25° latitude-longitude                                                 |
| Internal physical grid               | `720 × 1440`; the original `721 × 1440` grid's south-pole row is omitted |
| Latent resolution                    | 2°                                                                       |
| Temporal resolution                  | Any integer-hour lead time through one-hour and six-hour processors      |
| Demonstrated forecast horizon        | 14 days                                                                  |
| Dynamic input                        | One global weather state                                                 |
| Operational analyses                 | ECMWF IFS and NOAA GFS                                                   |
| Output variables                     | 17 surface variables and 5 atmospheric variables at 28 pressure levels   |
| Main network pattern                 | Encoder → repeated latent processor steps → decoder                      |
| Processor backbone                   | Neighborhood Attention Transformer, or NATTEN                            |
| Hidden dimension                     | 1024                                                                     |
| Training objective                   | Mean squared error, or MSE                                               |
| Reported RTX 4090 inference          | 14-day global forecast in 12 seconds                                     |
| Reported minimum deployment hardware | 16 GB VRAM and 32 GB system RAM                                          |
| Training hardware                    | Six RTX 4090 GPUs, half precision                                        |
| Current uncertainty output           | None described; WM-3 is evaluated as a deterministic model               |

---

## 3. Model architecture

### 3.1 State representation

WM-3 represents weather on a regular latitude-longitude grid. Pressure level is treated as a third spatial dimension rather than merely as a flat collection of unrelated channels.

At 0.25° resolution, the nominal field size is `721 × 1440`. The model removes the last latitude row, corresponding to the south pole, and operates on `720 × 1440` fields.

The input contains:

* Dynamic atmospheric and surface variables.
* Sine and cosine encodings of latitude and longitude.
* Sea-land mask.
* Soil type.
* Topography.
* Elevation.

The paper does not specify:

* Tensor axis ordering.
* File format.
* Unit conversion or normalization.
* Missing-value handling.
* Per-variable preprocessing.
* The exact representation used to combine surface-only fields with the pressure-level dimension.

### 3.2 Encoder

The encoder maps the 0.25° physical weather state into a learned 2° latent representation.

It contains:

* Convolutional layers.
* ResNet blocks.
* Two NATTEN blocks.
* A NATTEN window of `(5, 7, 7)` over the pressure/depth, latitude, and longitude dimensions.
* Hidden dimension 1024.

The exact convolution kernels, strides, number of ResNet blocks, activation functions, normalization layers, and resulting latent tensor dimensions are not reported.

### 3.3 Temporal processors

WM-3 has two separate temporal processors:

* A **six-hour processor**, where each invocation advances the latent state by six hours.
* A **one-hour processor**, where each invocation advances the latent state by one hour.

Each processor contains **10 NATTEN blocks**.

The processors share a compatible latent representation but do not appear to share weights. The paper does not report their parameter counts or whether their NATTEN window settings are identical to the encoder's window.

### 3.4 Latent rollout

For a requested lead time, WM-3:

1. Encodes the physical weather state once.
2. Repeatedly applies a greedy combination of one-hour and six-hour processors until the requested lead time is reached.
3. Decodes the resulting latent state into physical weather fields.

This differs from a conventional autoregressive rollout that repeatedly:

1. Encodes a physical state.
2. advances it by one forecast step,
3. decodes it,
4. and re-encodes the prediction for the next step.

The reported benefits of the latent rollout are:

* Less encoding and decoding computation.
* Removal of error specifically introduced by repeated encoding and decoding.
* Support for training against multiple forecast horizons during one latent rollout.
* Modularity because encoders, processors, and decoders communicate through a shared latent space.

The latent rollout does **not** eliminate forecast error accumulated by repeatedly applying the temporal processor itself.

Figure 1 of the paper also depicts decoders attached to intermediate latent states, indicating that intermediate forecast times can be decoded when needed. The exact scheduling and caching behavior for producing a complete hourly forecast sequence is not described.

### 3.5 Neighborhood attention

The processor backbone uses NATTEN rather than conventional global self-attention or Swin attention.

The authors report that, compared with an earlier WeatherMesh implementation based on Swin Transformer, NATTEN:

* Produced better forecasting performance.
* Provided a more suitable locality bias for atmospheric information transfer.
* Was faster.
* Used less memory when paired with fused neighborhood-attention kernels.

No numerical Swin-versus-NATTEN ablation is included, so the size of each improvement cannot be determined from the paper.

### 3.6 Spherical handling and position encoding

To operate on a global spherical grid, WM-3 uses:

* Custom circular padding.
* NATTEN's “bump attention” behavior at the poles.
* Rotary position embeddings, or RoPE, for token positions.

The paper does not provide the exact circular-padding algorithm, the axis or axes on which it is applied, or a mathematical definition of the polar handling.

### 3.7 Decoder

The decoder reverses the encoder's transformation:

1. Two NATTEN blocks first process the latent representation.
2. ResNet and deconvolutional layers map it back to the 0.25° physical grid.

The exact number and configuration of the ResNet and deconvolutional layers are not reported.

### 3.8 Operational dual-encoder design

The ERA5-pretrained encoder is replaced in the operational model by:

* One encoder for ECMWF IFS analyses.
* One encoder for NOAA GFS analyses.

Figure 4 shows the two encoded representations being combined at an addition node before entering the common processor and decoder. The abstract calls this a blended initial condition.

The paper does not describe:

* Learned or fixed weighting between IFS and GFS.
* Whether the addition is normalized.
* How discrepancies between the two analysis products are reconciled.
* Whether the processor and decoder are frozen during operational fine-tuning.
* Behavior when one analysis source is unavailable.

The authors state that WeatherMesh had been operating in production since March 2024 as of the paper's writing.

---

## 4. Input and output data

### 4.1 Pressure levels

All five atmospheric variables are forecast at these 28 pressure levels:

`10, 30, 50, 70, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 875, 900, 925, 950, 975, 1000 hPa`

### 4.2 Surface fields

| Surface field                            | ECMWF parameter ID | Used as operational input? |
| ---------------------------------------- | -----------------: | -------------------------- |
| 10 metre u wind component                |                165 | Yes                        |
| 10 metre v wind component                |                166 | Yes                        |
| 2 metre temperature                      |                167 | Yes                        |
| Mean sea-level pressure                  |                151 | Yes                        |
| Total cloud cover                        |                 45 | Yes                        |
| 2 metre dewpoint temperature             |                168 | Yes                        |
| 100 metre u wind component               |                246 | Yes                        |
| 100 metre v wind component               |                247 | Yes                        |
| Mean shortwave radiation flux            |                 15 | No                         |
| Large-scale precipitation                |                142 | No                         |
| Convective precipitation                 |                143 | No                         |
| Maximum 2 metre temperature              |                201 | No                         |
| Minimum 2 metre temperature              |                202 | No                         |
| Large-scale precipitation, 6-hour form   |                142 | No                         |
| Convective precipitation, 6-hour form    |                143 | No                         |
| Maximum 2 metre temperature, 6-hour form |                201 | No                         |
| Minimum 2 metre temperature, 6-hour form |                202 | No                         |

Only the first eight surface fields are used as operational inputs because most of the other forecast surface variables are not available in the IFS or GFS analyses used by WM-3.

### 4.3 Atmospheric fields

Each is represented at all 28 pressure levels.

| Atmospheric field    | ECMWF parameter ID |
| -------------------- | -----------------: |
| Geopotential, Z      |                129 |
| Temperature, T       |                130 |
| U wind component, U  |                131 |
| V wind component, V  |                132 |
| Specific humidity, Q |                133 |

This corresponds to:

* **140 atmospheric input fields:** 5 variables × 28 levels.
* **148 dynamic operational input fields:** 140 atmospheric fields + 8 surface fields.
* **157 predicted output fields:** 140 atmospheric fields + 17 surface fields.

These counts describe logical gridded fields; the actual tensor packing is not specified.

### 4.4 Example data shown in the paper

The paper does not include sample numerical records or downloadable example tensors.

Its principal qualitative example is a forecast initialized at **00Z on July 1, 2024**, showing:

* 10-metre wind speed.
* Mean sea-level pressure.
* A 72-hour lead time.
* Hurricane Beryl.

The figure compares WeatherMesh-3, AIFS, IFS HRES, and the IFS ensemble mean.

The authors observe that the machine-learning forecasts contain less fine-grained spatial detail but depict the hurricane's intensity more clearly than the IFS ensemble mean. This is a qualitative example rather than a comprehensive cyclone evaluation.

---

## 5. Training procedure

### 5.1 Training stages

The complete procedure has three practical phases.

#### Phase A: ERA5 training of the six-hour system

The encoder, six-hour processor, and decoder are trained using ERA5 reanalysis.

Reported settings:

* 42,000 training steps.
* Cosine learning-rate schedule.
* Maximum learning rate `3 × 10⁻⁴`.
* MSE objective at selected forecast lead times.
* Batch size 1.
* Half-precision training.
* Distributed Shampoo optimizer.
* Six RTX 4090 GPUs.

The target-horizon curriculum is:

| Starting step | Available target lead times            |
| ------------: | -------------------------------------- |
|             0 | 0, 6, 12 hours                         |
|         1,000 | 0, 6, 12, 18, 24 hours                 |
|        15,000 | 0, 6, 12, 18, 24, 30 hours             |
|        21,000 | 0, 6, 12, 18, 24, 30, 36 hours         |
|        26,000 | 0, 6, 12, 18, 24, 30, 36, 42 hours     |
|        30,000 | 0, 6, 12, 18, 24, 30, 36, 42, 48 hours |

At each step, the paper says that five target lead times are randomly chosen while always including the largest currently permitted lead time. The inclusion of a zero-hour target is reported but not explained.

The paper does not specify how multiple target losses are combined.

#### Phase B: Extended six-hour training

After the initial 42,000 steps, the system undergoes another **16,000-step annealing cycle** using target lead times extending to **120 hours**, or five days.

Although the longest reported training target is five days, inference and evaluation extend to 14 days by continuing to apply the temporal processors.

#### Phase C: One-hour processor training

After training the six-hour processor:

* The encoder and decoder are frozen.
* The one-hour processor is trained for 25,000 steps.
* Five target lead times between 0 and 24 hours are sampled at each step.

#### Phase D: Operational adaptation

Two new encoders are attached for IFS and GFS analyses.

The operational encoder training data covers:

* March 2021 through February 2024.

The paper does not report:

* Number of operational fine-tuning steps.
* Learning rate for this phase.
* Which existing components remain frozen.
* Loss weighting.
* Validation split.
* Data normalization or harmonization between IFS and GFS.

### 5.2 ERA5 date inconsistency

The paper gives two different descriptions of the ERA5 training period:

* Section 2.2 says **1979 through 2022**, with 2020 withheld.
* Appendix A.3 says **January 23, 1979 through December 28, 2019**, plus **February 1, 2021 through December 21, 2023**, again withholding 2020.

The appendix is more precise but conflicts with the main text. This document does not select one as correct.

### 5.3 Loss function and blurring

The training target is MSE.

The paper simultaneously notes that MSE-trained weather models can reduce loss by producing excessively smooth forecasts. No perceptual, spectral, adversarial, probabilistic, or explicit anti-blurring loss is reported for WM-3.

WM-3 therefore addresses blurring primarily through:

* Architectural design.
* Empirical evaluation of the RMSE-versus-blur trade-off.

It does not claim to have eliminated the underlying MSE incentive to smooth unpredictable fine-scale structure.

---

## 6. Training-memory problem and Matepoint

### 6.1 Problem

A single sample contains the complete global state, so the authors report that the model cannot be trained on only part of the Earth. The resulting implementation uses a batch size of one.

Naively storing all activations for backpropagation would require hundreds of GiB of GPU memory.

Ordinary activation checkpointing reduces memory use by recomputing activations during the backward pass, but it must retain the input of every checkpointed transformer block.

The paper's example states:

* A six-day forecast requires more than 200 transformer-layer executions.
* One latent state is approximately 200 MiB.
* Retaining these states would consume approximately 40 GiB solely for checkpoint inputs.

That is impractical on a 24 GB RTX 4090.

### 6.2 Resolution

The authors created **Matepoint**, a fork of PyTorch's checkpointing system.

Matepoint:

1. Moves checkpoint input tensors from GPU memory to CPU RAM.
2. Transfers them back to the GPU on an independent CUDA stream.
3. Pipelines transfers so required tensors arrive before each backward computation.

The paper claims that this removes the additional GPU-memory cost associated with extending the training rollout, although CPU memory, data-transfer bandwidth, and recomputation costs remain.

The library is said to have been released as an open-source Python package, but the paper itself does not provide its repository or package URL.

---

## 7. Development problems and reported responses

| Problem or observed limitation                                                                | WM-3 response                                                              | Resolution status                                         |
| --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | --------------------------------------------------------- |
| Repeated physical-space decoding and re-encoding adds computation and another source of error | Keep the state in latent space across forecast steps                       | Addressed architecturally; temporal rollout error remains |
| Six-hour-only models cannot directly produce arbitrary integer-hour leads                     | Separate one-hour and six-hour processors with greedy scheduling           | Addressed                                                 |
| A single temporal model may not serve all horizons equally well                               | Mixed-horizon training and separate temporal processors                    | Addressed in design; no ablation supplied                 |
| Earlier Swin implementation was slower, larger in memory, and reportedly less accurate        | Replace Swin with NATTEN and fused kernels                                 | Reported as improved; no quantitative comparison          |
| Standard planar attention does not directly handle a spherical grid                           | Circular padding, polar bump attention, rotary embeddings                  | Addressed at a high level; implementation details absent  |
| Long rollout backpropagation exceeds consumer-GPU memory                                      | Matepoint CPU offload and pipelined GPU return                             | Addressed for training according to authors               |
| The full globe prevents ordinary spatial minibatching                                         | Batch size 1 and distributed optimizer                                     | Constraint remains                                        |
| ERA5 pretraining does not directly match real-time operational analyses                       | Replace the encoder with IFS- and GFS-specific encoders                    | Addressed at the input layer                              |
| Many forecast surface variables are absent from operational analyses                          | Use only the first eight surface variables as inputs while forecasting 17  | Output-input asymmetry remains                            |
| MSE encourages blurry forecasts                                                               | Measure spectral blur and compare the accuracy-blur trade-off              | Evaluated, not fundamentally resolved                     |
| Deterministic forecasts do not quantify uncertainty                                           | Future work proposes large ensembles                                       | Unresolved                                                |
| Additional live observations are not assimilated directly                                     | Future encoder pathways and a live data-assimilation pipeline are proposed | Unresolved                                                |

---

## 8. Evaluation methodology

### 8.1 Data periods

For the HRES and AIFS scorecards in Figure 2:

* March 10, 2024 through July 31, 2024.
* Forecasts initialized only at 00Z.

For the blur and Hurricane Beryl analysis in Figure 3:

* March 21, 2024 through July 31, 2024.
* Forecasts initialized only at 00Z.

The authors say these dates reflect data availability rather than deliberate event selection.

### 8.2 Reference and comparison systems

WM-3 is compared against:

* ECMWF IFS HRES.
* ECMWF AIFS.
* ECMWF IFS ENS and subsets of its ensemble members.

ERA5 is used as the verification target and is described by the authors as the best available representation of ground truth.

This is analysis-to-analysis verification rather than direct verification against weather-station, satellite, aircraft, radar, radiosonde, or cyclone best-track observations.

### 8.3 Accuracy metric

The paper uses latitude-weighted RMSE:

[
\mathrm{RMSE}
=============

\frac{1}{T}
\sum_{t=1}^{T}
\sqrt{
\frac{1}{H W}
\sum_{i=1}^{H}
\sum_{j=1}^{W}
w(i)
\left(\hat{X}*{i,j}^{t}-X*{i,j}^{t}\right)^2
}
]

with:

* `H = 720`.
* `W = 1440`.
* `X̂` as the forecast.
* `X` as the ERA5 target.
* `t` as an evaluation date.

The paper prints the latitude weight as:

[
w(i)=\cos\left(\frac{i\pi}{720}\right)
]

There is a reproducibility ambiguity: when combined literally with an index from 1 to 720, this expression becomes negative over half of the grid, which is not normal for latitude area weighting. The paper does not state whether `i` is intended to be a centered latitude coordinate or whether the printed expression omits an offset.

### 8.4 Blur metric

The paper defines:

[
\text{Blur score}=\frac{1}{\sqrt{S_{500}}}
]

where `S₅₀₀` is spectral power at a 500 km wavelength.

Under this definition:

* Greater 500 km spectral power produces a lower blur score.
* A lower blur score therefore represents less smoothing at that scale.

The authors acknowledge that 500 km is somewhat arbitrary. It was selected to approximate the scale of a hurricane while avoiding domination by artifacts near the model's approximately 25 km grid spacing.

### 8.5 Evaluation coverage

The HRES scorecard reports 690 targets. Figure 2 indicates that this consists of:

* Five atmospheric variables.
* Thirteen plotted pressure levels.
* Four surface variables: 10U, 10V, MSL, and 2T.
* Ten daily lead times.

That gives `(5 × 13 + 4) × 10 = 690`.

Consequently, the headline 690-target result does **not** appear to cover:

* All 28 pressure levels generated by WM-3.
* All 17 forecast surface variables.
* Precipitation, cloud cover, dewpoint, 100-metre winds, shortwave flux, or surface extrema.

The additional blur plots cover:

* The four named surface fields.
* Z, T, U, V, and Q at 1000, 850, 500, and 250 hPa.
* Lead times of 1, 3, 5, and 7 days.

IFS ensemble subsets contain `1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, and 51` members.

---

## 9. Reported findings

### 9.1 Accuracy relative to HRES

The authors report that WM-3 has lower RMSE than IFS HRES for **689 of 690 evaluated targets**.

The single exception is:

* Geopotential at 50 hPa at a 10-day lead time.

The largest highlighted improvement is:

* **37.7% lower RMSE** for 2-metre temperature at a one-day lead time.

This is the maximum highlighted improvement, not a reported average improvement over all targets.

The authors caution that:

* HRES is not optimized specifically for RMSE.
* Some earlier comparisons evaluate HRES against its own analysis.
* This paper instead evaluates both systems against ERA5.

### 9.2 Accuracy relative to AIFS

The paper reports that WM-3:

* Consistently outperforms AIFS at 50 hPa.
* Is comparatively weaker at a 14-day lead time.
* Is particularly weaker at that horizon for geopotential.

The paper does not reduce the AIFS comparison to one aggregate score.

### 9.3 Accuracy-blur trade-off

For some fields and lead times, WM-3 reportedly achieves both:

* Lower RMSE.
* Lower blur score.

than AIFS and the IFS ensemble mean.

For most of the displayed variables and lead times, the paper reports a better accuracy-versus-blur trade-off: either more RMSE improvement for a given degree of smoothing or less smoothing for a given RMSE improvement.

The plots do not establish that WM-3 is universally sharper or more accurate for every output field.

### 9.4 Computational efficiency

Reported inference performance includes:

* A 14-day, 0.25° global forecast in 12 seconds on one RTX 4090.
* Hourly global forecasts through 14 days in under 10 seconds on one H100 server.
* Operation on a consumer-class machine with 16 GB VRAM and 32 GB RAM.
* Approximately 143,000 times faster than physics-based numerical weather prediction when measured using node-seconds per forecast lead hour.

The 143,000× figure is a cross-system computational comparison, not a controlled same-hardware benchmark.

### 9.5 Operational status

The paper states that WeatherMesh had been operational since March 2024. It does not provide operational uptime, failure rate, data-latency statistics, service-level objectives, or an independent operational audit.

---

## 10. Limitations and unresolved details

### Scientific and evaluation limitations

* Evaluation spans only approximately four to five months.
* Only 00Z initializations are evaluated.
* Seasonal and multi-year operational performance is not shown.
* ERA5 analysis is treated as ground truth rather than direct observations.
* The main accuracy result covers only a subset of the model's output fields and levels.
* Precipitation and several other operationally important outputs are absent from the headline scorecards.
* The 500 km blur scale is acknowledged as arbitrary.
* A single Hurricane Beryl visualization is qualitative evidence, not a systematic extreme-weather study.
* No probabilistic calibration or uncertainty estimate is produced.
* No regional, user-facing, economic, or decision-quality evaluation is included.

### Architectural and training gaps

The paper does not provide:

* Parameter count.
* Exact latent tensor shape.
* Detailed convolution and deconvolution configuration.
* Processor attention-window dimensions.
* Activation and normalization choices.
* Variable normalization and loss weighting.
* Checkpoint size or serialization format.
* Random seeds or run-to-run variance.
* Exact greedy rollout algorithm.
* Operational fine-tuning schedule.
* Failure handling for absent or delayed IFS/GFS data.
* Numerical ablations for latent rollout, dual encoders, NATTEN, RoPE, or Matepoint.
* Comparison between a single encoder and the blended two-encoder initialization.
* Training wall-clock time.
* Inference precision and batching details.

### Internal inconsistencies or ambiguities

* The ERA5 training end date differs between the main text and appendix.
* The printed latitude-weighting equation is ambiguous.
* The Figure 2 caption says “all pressure levels and surface variables,” but the displayed evaluation uses 13 pressure levels and four surface variables rather than the model's complete 28-level, 17-surface-variable output.
* The paper says the model code and Matepoint package are open source but does not include their URLs.

---

## 11. Application-facing interpretation

The paper defines a forecast engine, not an application.

### Paper-supported forecast-engine contract

Conceptually, an application can treat WM-3 as accepting:

1. A synchronized global weather analysis.
2. The required dynamic atmospheric and surface fields.
3. Static geographic fields.
4. An integer-hour forecast lead.

It returns:

* Global gridded numerical forecasts at 0.25° resolution.
* Seventeen surface fields.
* Five atmospheric fields at 28 pressure levels.

The shared latent design also permits:

* Decoding at selected intermediate lead times.
* Adding new analysis-specific encoders.
* Reusing the same processor and decoder for multiple data sources.

### Not defined by the paper

The paper does not specify:

* An API.
* A model checkpoint or loading interface.
* An input file schema.
* Units and conversions.
* Local interpolation from the global grid.
* Map tiles or visualization.
* Location search.
* Alert thresholds.
* Forecast wording.
* Data refresh scheduling.
* Authentication or authorization.
* Caching and storage.
* Monitoring and fallback logic.
* Licensing terms.
* User-facing uncertainty.
* Downscaling or bias correction.

Therefore, the model outputs should be understood as scientific gridded fields rather than ready-made consumer forecast statements.

---

## 12. Every cited source and its relevance

The bibliography contains **18 cited sources: 17 research papers and one ECMWF technical document**.

### Direct architectural and optimization foundations

1. **Bi et al. (2023), “Accurate medium-range global weather forecasting with 3D neural networks” / Pangu-Weather.**
   Establishes a high-resolution, three-dimensional transformer representation in which pressure is treated as a spatial dimension. WM-3 explicitly identifies this as an architectural precedent, but introduces a different latent-rollout and neighborhood-attention design. ([arXiv][1])

2. **Dosovitskiy et al. (2020), “An Image Is Worth 16×16 Words.”**
   Introduces the Vision Transformer approach of processing spatial patches as transformer tokens. It is the general transformer foundation cited for WM-3's vision-style treatment of global weather grids. ([arXiv][2])

3. **He et al. (2016), “Deep Residual Learning for Image Recognition.”**
   Introduces residual blocks that learn corrections relative to their inputs, enabling deeper networks to train more effectively. WM-3 uses ResNet blocks in its encoder and decoder. ([arXiv][3])

4. **Hassani et al. (2023), “Neighborhood Attention Transformer.”**
   Introduces local sliding-window neighborhood attention and the NATTEN implementation. NATTEN blocks form the central attention mechanism in WM-3's encoder, processors, and decoder. ([arXiv][4])

5. **Hassani, Hwu, and Shi (2024), “Faster Neighborhood Attention.”**
   Develops fused and more efficient neighborhood-attention kernels. WM-3 cites these kernels as the reason NATTEN can be faster and use less memory than its earlier Swin implementation. ([arXiv][5])

6. **Liu et al. (2021), “Swin Transformer.”**
   Introduces hierarchical attention using shifted, non-overlapping windows. Earlier WeatherMesh versions used Swin; WM-3 reports replacing it with NATTEN because NATTEN performed better in their implementation. ([arXiv][6])

7. **Su et al. (2024), “RoFormer: Enhanced Transformer with Rotary Position Embedding.”**
   Introduces RoPE, which encodes absolute positions through rotations while making relative position information available to attention. WM-3 uses rotary position embeddings for its tokens. ([arXiv][7])

8. **Gupta, Koren, and Singer (2018), “Shampoo: Preconditioned Stochastic Tensor Optimization.”**
   Introduces the Shampoo tensor-aware second-order optimizer. It is the underlying optimizer used for WM-3 training. ([arXiv][8])

9. **Shi et al. (2023), “A Distributed Data-Parallel PyTorch Implementation of the Distributed Shampoo Optimizer.”**
   Provides the scalable PyTorch implementation used to run Shampoo across WM-3's six-GPU training setup. ([arXiv][9])

### Weather-model precedents and comparators

1. **Kurth et al. (2023), “FourCastNet: Accelerating Global High-Resolution Weather Forecasting Using Adaptive Fourier Neural Operators.”**
    Demonstrates high-resolution global machine-learning forecasts using adaptive Fourier neural operators and large inference speedups over NWP. WM-3 cites it as part of the preceding ML-weather landscape, not as a component of its architecture. ([arXiv][10])

2. **Chen et al. (2023), “FuXi: A Cascade Machine Learning Forecasting System for 15-Day Global Weather Forecast.”**
    Uses a cascade of horizon-specialized models to reduce long-range error accumulation and produce 15-day forecasts. It provides related evidence for horizon-specific model components, although WM-3 uses shared-latent one-hour and six-hour processors rather than FuXi's cascade. ([arXiv][11])

3. **Lam et al. (2023), “Learning Skillful Medium-Range Global Weather Forecasting” / GraphCast.**
    Uses a graph-neural-network encoder-processor-decoder and autoregressive physical-grid forecasts. WM-3 cites GraphCast as a leading accuracy benchmark, as precedent for MLWP-versus-HRES evaluation, and as evidence of forecast blurring. GraphCast's physical-space autoregression also contrasts with WM-3's latent rollout. ([arXiv][12])

4. **Bodnar et al. (2024), “Aurora: A Foundation Model of the Atmosphere.”**
    Presents a large pretrained Earth-system foundation model that can be adapted to several forecasting domains. WM-3 cites Aurora as a state-of-the-art comparison and uses it in claims concerning relative inference compute and VRAM requirements. Aurora is not part of WM-3. ([arXiv][13])

5. **Lang et al. (2024a), “AIFS—ECMWF's Data-Driven Forecasting System.”**
    Describes ECMWF's operational machine-learning forecast model, which combines graph-based encoding and decoding with a sliding-window transformer processor. AIFS is a direct accuracy and blur comparator for WM-3. ([arXiv][14])

6. **Lang et al. (2024b), “AIFS-CRPS.”**
    Develops a stochastic AIFS variant trained with a CRPS-derived proper scoring objective to generate exchangeable ensemble members. WM-3 cites it in connection with ML forecast blurring; it also represents a probabilistic alternative to WM-3's deterministic MSE training. ([arXiv][15])

7. **Price et al. (2023), “GenCast: Diffusion-Based Ensemble Forecasting for Medium-Range Weather.”**
    Uses a diffusion model to generate probabilistic 15-day global forecast ensembles. WM-3 cites it as related progress in machine-learning weather prediction; its probabilistic design is not incorporated into the current WM-3 model. ([arXiv][16])

### Evaluation and operational references

1. **Rasp et al. (2024), “WeatherBench 2.”**
    Defines an open benchmark and evaluation framework for global data-driven weather models. WM-3 follows its convention of withholding 2020 and uses its spectral-power definition when calculating the 500 km blur score. ([arXiv][17])

2. **ECMWF (2024), “IFS Documentation CY49R1—Part V: Ensemble Prediction System.”**
    Documents the ECMWF ensemble system. WM-3 uses the 51-member IFS ensemble and ensemble-member subsets to construct its RMSE-versus-blur comparison curves. This is technical documentation rather than a research paper.

---

## 13. Concise architecture summary

WeatherMesh-3 is a deterministic encoder-processor-decoder weather model operating globally at 0.25° resolution.

Its encoder compresses one physical weather state into a 2° latent grid using convolutional layers, ResNet blocks, geographic constants, and NATTEN. Separate 10-block NATTEN processors advance this latent state by either one or six hours. A greedy sequence of processor calls reaches any integer-hour lead time without repeatedly returning to physical space. A NATTEN/ResNet/deconvolution decoder reconstructs the requested physical forecast.

The operational model replaces the ERA5 encoder with separate IFS and GFS encoders whose latent outputs are added before processing.

The principal engineering contribution is not a new weather-variable representation or probabilistic objective. It is the combination of:

* Persistent latent-space rollout.
* Modular mixed-horizon processors.
* Multiple analysis-specific encoders.
* Neighborhood attention.
* Consumer-GPU-oriented long-rollout training through Matepoint.

---

## 14. Bottom line

The paper presents a compact and operationally motivated global weather model whose distinguishing feature is that forecasting occurs almost entirely in a shared latent space.

The strongest reported evidence is:

* Lower ERA5-verified RMSE than HRES on 689 of 690 evaluated field/lead combinations.
* A favorable accuracy-versus-blur trade-off on the displayed variables.
* A 14-day forecast in 12 seconds on one RTX 4090.
* Training and inference designed around comparatively inexpensive hardware.

The most important qualifications are:

* The operational evaluation covers only several months and 00Z initializations.
* The headline scorecard evaluates a subset of model outputs.
* ERA5 is used as the verification reference.
* MSE-related smoothing remains an acknowledged issue.
* The current model is deterministic.
* Several implementation details required for reproduction are omitted.
* No component-level ablation demonstrates how much each architectural contribution is responsible for the reported result.

The paper is therefore a useful architectural and operational reference, but not a complete implementation specification or a sufficient definition of a finished weather application.

[1]: https://arxiv.org/abs/2211.02556?utm_source=chatgpt.com "Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast"
[2]: https://arxiv.org/abs/2010.11929?utm_source=chatgpt.com "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
[3]: https://arxiv.org/abs/1512.03385?utm_source=chatgpt.com "Deep Residual Learning for Image Recognition"
[4]: https://arxiv.org/abs/2204.07143?utm_source=chatgpt.com "Neighborhood Attention Transformer"
[5]: https://arxiv.org/abs/2403.04690?utm_source=chatgpt.com "Faster Neighborhood Attention: Reducing the O(n^2) Cost of Self Attention at the Threadblock Level"
[6]: https://arxiv.org/abs/2103.14030?utm_source=chatgpt.com "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
[7]: https://arxiv.org/abs/2104.09864?utm_source=chatgpt.com "RoFormer: Enhanced Transformer with Rotary Position Embedding"
[8]: https://arxiv.org/abs/1802.09568?utm_source=chatgpt.com "Shampoo: Preconditioned Stochastic Tensor Optimization"
[9]: https://arxiv.org/abs/2309.06497?utm_source=chatgpt.com "A Distributed Data-Parallel PyTorch Implementation of the Distributed Shampoo Optimizer for Training Neural Networks At-Scale"
[10]: https://arxiv.org/abs/2208.05419?utm_source=chatgpt.com "FourCastNet: Accelerating Global High-Resolution Weather Forecasting using Adaptive Fourier Neural Operators"
[11]: https://arxiv.org/abs/2306.12873?utm_source=chatgpt.com "FuXi: A cascade machine learning forecasting system for 15-day global weather forecast"
[12]: https://arxiv.org/abs/2212.12794 "GraphCast: Learning skillful medium-range global weather forecasting"
[13]: https://arxiv.org/abs/2405.13063?utm_source=chatgpt.com "A Foundation Model for the Earth System"
[14]: https://arxiv.org/abs/2406.01465?utm_source=chatgpt.com "AIFS -- ECMWF's data-driven forecasting system"
[15]: https://arxiv.org/abs/2412.15832?utm_source=chatgpt.com "AIFS-CRPS: Ensemble forecasting using a model trained with a loss function based on the Continuous Ranked Probability Score"
[16]: https://arxiv.org/abs/2312.15796?utm_source=chatgpt.com "GenCast: Diffusion-based ensemble forecasting for medium-range weather"
[17]: https://arxiv.org/abs/2308.15560?utm_source=chatgpt.com "WeatherBench 2: A benchmark for the next generation of data-driven global weather models"
