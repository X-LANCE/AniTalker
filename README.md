
# AniTalker: Animate Vivid and Diverse Talking Faces through Identity-Decoupled Facial Motion Encoding


<p align="center">
  <a href="https://x-lance.github.io/AniTalker/">Demo</a> &nbsp;&nbsp;&nbsp; <a href="#">Paper (Uploading)</a> &nbsp;&nbsp;&nbsp; <a href="https://github.com/X-LANCE/AniTalker">Code</a>
</p>


![](docs/img/generated_result.png)


## Environment Installation

```shell
conda create -n anitalker python==3.9.0
conda activate anitalker
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install -r requirements.txt
```

## Model Zoo

Please download the checkpoint and place them into the folder ```ckpts```

## Run the demo

### Face facing forward

Keep pose_yaw, pose_pitch, pose_roll to zero.

![monalisa_facing_forward](assets/monalisa_facing_forward.gif)

Demo script:

```
python ./code/demo_audio_generation.py \
    --infer_type 'mfcc_pose_only' \
    --stage1_checkpoint_path 'ckpts/stage1.ckpt' \
    --stage2_checkpoint_path 'ckpts/stage2_pose_only.ckpt' \
    --test_image_path 'test_demos/portraits/monalisa.jpg' \
    --test_audio_path 'test_demos/audios/english_female.wav' \
    --result_path 'results/monalisa_frontal_face/' \
    --control_flag True \
    --seed 0 \
    --pose_yaw 0 \
    --pose_pitch 0 \
    --pose_roll 0 
```


### Adjust the orentation 

Chaning pose_yaw from `0` to `0.25`

![monalisa_turn_head_right](assets/monalisa_turn_head_right.gif)

Demo script:

```
python ./code/demo_audio_generation.py \
    --infer_type 'mfcc_pose_only' \
    --stage1_checkpoint_path 'ckpts/stage1.ckpt' \
    --stage2_checkpoint_path 'ckpts/stage2_pose_only.ckpt' \
    --test_image_path 'test_demos/portraits/monalisa.jpg' \
    --test_audio_path 'test_demos/audios/english_female.wav' \
    --result_path 'results/monalisa_turn_head_right/' \
    --control_flag True \
    --seed 0 \
    --pose_yaw 0.25 \
    --pose_pitch 0 \
    --pose_roll 0 
```


### Talking in Free-style

![monalisa_free_style](assets/monalisa_free_style.gif)

Demo script:

```
python ./code/demo_audio_generation.py \
    --infer_type 'mfcc_pose_only' \
    --stage1_checkpoint_path 'ckpts/stage1.ckpt' \
    --stage2_checkpoint_path 'ckpts/stage2_pose_only.ckpt' \
    --test_image_path 'test_demos/portraits/monalisa.jpg' \
    --test_audio_path 'test_demos/audios/english_female.wav' \
    --result_path 'results/monalisa_free_style/'
```


## Some Advice and Questions

<details><summary> 1. Using similar poses to the portrait</summary> Try to keep the generated face angle close to the original portrait angle to avoid potential deformation issues. For example, if the face starts by rotating to the left (viewed from the portrait face angle), it is better to use a value for yaw between -1 and 0 (-90 to 0 degrees) because when the difference in angle from the portrait is large, the face can appear distorted. </details><details>

<summary> 2. Utilizing algorithms to automatically extract or control using other faces' angles</summary> If you need to automate face control, you can use some pose extraction algorithms to do so, such as extracting the pose of another person to drive this portrait. The algorithms for extraction have been open sourced at <a href="https://github.com/liutaocode/talking_face_preprocessing?tab=readme-ov-file#face-orientation-angles">this link</a> </details>

<details><summary> 3. What are the differences between MFCC and Hubert features?</summary>Both `MFCC` and `Hubert` are front-end features for speech, used to extract audio signals. Since `Hubert` features require more environmental dependencies and occupy a lot of disk space, we have replaced this feature with a lightweight feature (MFCC) for everyone to use for quick inference. The rest of the code remains the same. We've observed that MFCC converges more easily but is slightly inferior in performance to Hubert. If you need to extract Hubert features, please refer to <a href="https://github.com/liutaocode/talking_face_preprocessing?tab=readme-ov-file#audio-feature-extraction">this link</a> </details>


## Citation

```
@INPROCEEDINGS{
}
```

## Ackonwlegements

We sincerely appreciate the contributions of numerous prior works that have paved the way for the development of AniTalker. 

Stage 1, which mainly involves training motion encoder and the rendering module, extensively utilizes resources from [LIA](https://github.com/wyhsirius/LIA). The second stage of diffusion training is based on [diffae](https://github.com/phizaz/diffae) and [espnet](https://espnet.github.io/espnet/_modules/espnet2/asr/encoder/conformer_encoder.html). For the computation of mutual information loss, we implement methods from [CLUB](https://github.com/Linear95/CLUB) and employ [AAM-softmax](https://github.com/TaoRuijie/ECAPA-TDNN) in the training of face recognition. Furthermore, we utilize the pretrained Hubert model provided by [TencentGameMate](https://github.com/TencentGameMate/chinese_speech_pretrain). Additionally, we've made the dataset preprocessing codes (such as pose and landmark extraction) available at [talking_face_preprocessing](https://github.com/liutaocode/talking_face_preprocessing). 