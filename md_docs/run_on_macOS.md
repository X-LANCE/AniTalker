
We have tested the environment on following devices:

- Macbook Pro M3 Max (128GB + 8TB), Sonoma 14.6.1
- Macbook Pro M1 Pro (16GB + 2TB), Sonoma 14.5

# 1. Project Download

```
git clone https://github.com/X-LANCE/AniTalker.git  
```

# 2. Dependencies Installation

```
conda create -n anitalker python==3.9.0 -c conda-forge 
conda activate anitalker 
conda install pytorch torchvision torchaudio -c pytorch 
conda install libffi
conda install -c conda-forge numpy tokenizers

# install espnet 
git clone https://github.com/espnet/espnet.git 
cd espnet 
pip install -e . 

# [Optional] You may install rust by the following script if you receive warnings.
# curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Finally, run: 
pip install -r requirements_macOS.txt 
```

If you still have trouble installing the environment, you can check [conda env file](../md_docs/mac_os_env_list/conda_environment.yml) or [pip env file](../md_docs/mac_os_env_list/pip_requirements.txt) for detailed version.

# 3. Model Download

```
# Prepare the Model  

cd AniTalker 
mkdir ckpts 
Go to https://huggingface.co/taocode/anitalker_ckpts/tree/main  
then download all six models in path ~/AniTalker/ckpts/ 

```
![](../assets/models_huggingface.png)

# 4. Run

```
 PYTORCH_ENABLE_MPS_FALLBACK=1 python ./code/demo.py \
    --infer_type 'hubert_audio_only' \
    --stage1_checkpoint_path 'ckpts/stage1.ckpt' \
    --stage2_checkpoint_path 'ckpts/stage2_audio_only_hubert.ckpt' \
    --test_image_path 'test_demos/portraits/monalisa.jpg' \
    --test_audio_path 'test_demos/audios/monalisa.wav' \
    --test_hubert_path 'test_demos/audios_hubert/monalisa.npy' \
    --result_path 'outputs/monalisa_hubert/'
```
- Macbook pro M3 Max (128GB + 8TB), Sonoma 14.6.1:

![](../assets/results_run_on_macOS_m3.png)

- Macbook pro M1 Pro (16GB + 2TB), Sonoma 14.5:

![](../assets/results_run_on_macOS_m1.jpg)

# 5. Modify log

- dependencies: requirements.txt
- use mps insted of cuda
- change float64 to float32
- PYTORCH_ENABLE_MPS_FALLBACK=1
