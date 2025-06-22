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
conda activate LVM
```

## 3. Install Required Packages

ubuntu==20.04  cuda==11.8
```shell
pip install torch==2.4.1
pip install torchvision==0.19.1
pip install diffusers==0.30.3
pip install transformers==4.44.2
pip install accelerate==0.34.2
pip install numpy==1.26.4
pip install pandas==2.2.2
pip install matplotlib==3.8.4
pip install tqdm==4.66.4
pip install scikit-learn==1.4.2
pip install scipy==1.13.1
pip install Pillow==10.3.0
pip install scikit-image==0.22.0
pip install joblib==1.4.2
pip install openpyxl==3.1.2
```

Then you should get an env like:
```shell
Package                  Version
------------------------ ----------
accelerate               0.34.2
certifi                  2024.8.30
charset-normalizer       3.3.2
diffusers                0.30.3
et-xmlfile               1.1.0
filelock                 3.16.1
fsspec                   2024.9.0
ftfy                     6.2.3
huggingface              0.0.1
huggingface-hub          0.25.0
idna                     3.10
importlib_metadata       8.5.0
Jinja2                   3.1.4
joblib                   1.4.2
MarkupSafe               2.1.5
mpmath                   1.3.0
networkx                 3.3
numpy                    2.1.1
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
nvidia-nvjitlink-cu12    12.6.68
nvidia-nvtx-cu12         12.1.105
openai-clip              1.0.1
openpyxl                 3.1.5
packaging                24.1
pillow                   10.4.0
pip                      24.2
protobuf                 5.28.2
psutil                   6.0.0
PyYAML                   6.0.2
regex                    2024.9.11
requests                 2.32.3
safetensors              0.4.5
scikit-learn             1.5.2
scipy                    1.14.1
sentence-transformers    3.1.1
sentencepiece            0.2.0
setuptools               75.1.0
sympy                    1.13.3
threadpoolctl            3.5.0
tokenizers               0.19.1
torch                    2.4.1
torchvision              0.19.1
tqdm                     4.66.5
transformers             4.44.2
triton                   3.0.0
typing_extensions        4.12.2
urllib3                  2.2.3
wcwidth                  0.2.13
wheel                    0.44.0
zipp                     3.20.2
```
## 4. Locate and Modify StableDiffusion3Pipeline
Open `Experiment1.py` in your code editor.

Hold down the `ctrl` key if you are on Linux or Windows, or the `command` key if you are on MacOS, and click on StableDiffusion3Pipeline.

![image](/readme/Experiment1.png)

This will navigate to the file `pipeline_stable_diffusion_3.py`.

Replace `pipeline_stable_diffusion.py` with the file of the same name from this repository.

## 5. Explanation of Our Code Files

`pipeline_stable_diffusion.py`: 




`Experiment1.py`：

Initial prototype experiment.
Generates original and compressed images
Tests various compression ratios and steps individually

`Experiment2.py`：

Main batch experiment logic.
Sweeps across 12 compress steps × 9 ratios × 50 seeds = 5400 runs
Saves images and records PSNR/SSIM into a CSV

`Experiment3.py`：

Post-processing and quality modeling.
Normalizes PSNR & SSIM, applies log mapping
Fits sigmoid surface function: 
Saves formula and surface visualization

## 6 Explanation of Our Results

Our generated image is available in:

xxx

