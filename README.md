# Compression-Inference

**Target:** The aim is to explore the impact of different compression ratios and different number of inference steps on image quality.

---

## 1. Environment Setup

```shell
conda create --name sd3_m python=3.10
conda activate sd3_m
```

## 2. Activate Environment

Activate the created environment.

```shell
conda activate sd3_m
```

## 3. Install Required Packages

ubuntu==20.04  cuda==11.8
```shell
pip install torch==2.4.1
pip install sentence-transformers==3.1.1
pip install diffusers==0.30.3
pip install transformers==4.44.2
pip install accelerate==0.34.2
pip install protobuf==5.28.2
pip install sentencepiece==0.2.0
pip install openai-clip==1.0.1
pip install torchvision==0.19.1
pip install openpyxl==3.1.5
pip install pandas==2.3.0
pip install scikit-image==0.25.2
pip install matplotlib==3.10.3
```

Then you should get an env like:
```shell
Package                  Version    
------------------------ -----------
accelerate               0.34.2
certifi                  2025.6.15
charset-normalizer       3.4.2
contourpy                1.3.2
cycler                   0.12.1
diffusers                0.30.3
et_xmlfile               2.0.0
filelock                 3.18.0
fonttools                4.58.4
fsspec                   2025.5.1
ftfy                     6.3.1
hf-xet                   1.1.5
huggingface-hub          0.33.0
idna                     3.10
imageio                  2.37.0
importlib_metadata       8.7.0
Jinja2                   3.1.6
joblib                   1.5.1
kiwisolver               1.4.8
lazy_loader              0.4
MarkupSafe               3.0.2
matplotlib               3.10.3
mpmath                   1.3.0
networkx                 3.4.2
numpy                    2.2.6
nvidia-cublas-cu12       12.1.3.1
nvidia-cuda-cupti-cu12   12.1.105
nvidia-cuda-nvrtc-cu12   12.1.105
nvidia-cuda-runtime-cu12 12.1.105
nvidia-cudnn-cu12        9.1.0.70
nvidia-cufft-cu12        11.0.2.54
nvidia-curand-cu12       10.3.2.106
nvidia-cusolver-cu12     11.4.5.107
nvidia-cusparse-cu12     12.1.0.106
nvidia-nccl-cu12         2.20.5
nvidia-nvjitlink-cu12    12.9.86
nvidia-nvtx-cu12         12.1.105
openai-clip              1.0.1
openpyxl                 3.1.5
packaging                25.0
pandas                   2.3.0
pillow                   11.2.1
pip                      25.1
protobuf                 5.28.2
psutil                   7.0.0
pyparsing                3.2.3
python-dateutil          2.9.0.post0
pytz                     2025.2
PyYAML                   6.0.2
regex                    2024.11.6
requests                 2.32.4
safetensors              0.5.3
scikit-image             0.25.2
scikit-learn             1.7.0
scipy                    1.15.3
sentence-transformers    3.1.1
sentencepiece            0.2.0
setuptools               78.1.1
six                      1.17.0
sympy                    1.14.0
threadpoolctl            3.6.0
tifffile                 2025.5.10
tokenizers               0.19.1
torch                    2.4.1
torchvision              0.19.1
tqdm                     4.67.1
transformers             4.44.2
triton                   3.0.0       
typing_extensions        4.14.0
tzdata                   2025.2
urllib3                  2.5.0
wcwidth                  0.2.13
wheel                    0.45.1
zipp                     3.23.0
```
## 4. Locate and Modify StableDiffusion3Pipeline
Open `Experiment1.py` in your code editor.

Hold down the `ctrl` key if you are on Linux or Windows, or the `command` key if you are on MacOS, and click on StableDiffusion3Pipeline.

![image](/readme/Experiment1_code.png)

This will navigate to the file `pipeline_stable_diffusion_3.py`.

Replace `pipeline_stable_diffusion.py` with the file of the same name from this repository.

## 5. Explanation of Our Code Files

`pipeline_stable_diffusion.py`: 

In line 664, the method _compress_latents is defined to quantize latent values by rounding to a fixed number of decimal places, determined by the scale parameter.

In line 811, the parameter compress_at_step specifies at which denoising step the latent tensor will be compressed.

In line 1108, the latent compression is triggered when the current step index i equals compress_at_step, and it calls _compress_latents with the specified compress_scale.

`Experiment1.py`：

Initial prototype experiment.

Generates original and compressed images.

Tests various compression ratios and steps individually.

`Experiment2.py`：

Main batch experiment logic.

Sweeps across 12 compress steps × 9 ratios × 50 seeds = 5400 runs.

Saves images and records PSNR/SSIM into a CSV.

`fit_model.py`：

Post-processing and quality modeling.

Normalizes PSNR & SSIM, applies log mapping.

Fits sigmoid surface function.

Saves formula and surface visualization.

## 6. Explanation of Our Results

The data generated in Experiments 1-2 and fit_model are presented in dir：`result` (Coming soon).

Our generated image is available in:

Google Drive：Coming soon

Image naming rules：

```shell
step_ratio_seed_compressed.png

step:        The denoising step at which latent compression is applied.
ratio:       The compression ratio used (e.g., 0.75 means 75% precision retained).
seed:        The random seed used to initialize generation.
compressed:  Indicates this is the compressed version of the generated image.

eg：step10_ratio0.75_seed42_compressed.png
```
## 7. Acknowledge

[Stable Diffusion 3 Medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers/tree/main): It is a wonferful image generation model.

Thanks to all my partners!

If you have any confusion, please feel free to contact us!
