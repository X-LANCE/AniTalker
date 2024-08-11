
We have tested the environment on following devices:

- Macbook Pro M3 Max (128GB + 8TB), Sonoma 14.6.1
- Macbook Pro M1 Pro (16GB + 2TB), Sonoma 14.5

We don't have an Intel-based Mac on hand. If you happen to have one, we welcome you to submit the testing environment and results.

# 1. Project Download

```
git clone https://github.com/X-LANCE/AniTalker.git  
```

# 2. Dependencies Installation

```
# install pytorch env for mac os 
conda create -n anitalker python==3.9.0 -c conda-forge 
conda activate anitalker 
conda install pytorch torchvision torchaudio -c pytorch 

# install espnet 
git clone https://github.com/espnet/espnet.git 
cd espnet 
git checkout b10464
pip install -e . 


conda install -c conda-forge pytorch-lightning=1.6.5 torchmetrics=0.5.0 transformers=4.19.2 moviepy numpy tokenizers scipy tqdm libffi 

pip install python_speech_features


# [Optional] You may install rust by the following script if you receive warnings.
# curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

```

If you still have trouble installing the environment, you can check [conda env file](../md_docs/mac_os_env_list/conda_environment.yml) or [pip env file](../md_docs/mac_os_env_list/pip_requirements.txt) for detailed version.

# 3. Model Download

Please follow the instructions provided in the `README.md` file to download all the required models (including the hubert model). 

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
