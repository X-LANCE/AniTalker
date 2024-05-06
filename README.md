
# AniTalker: Animate Vivid and Diverse Talking Faces through Identity-Decoupled Facial Motion Encoding

[Demo](https://x-lance.github.io/AniTalker/) 

[Paper]() [Uploading]

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

```
python ./code/demo_audio_generation.py \
    --infer_type 'mfcc_pose_only' \
    --stage1_checkpoint_path 'ckpts/stage1.ckpt' \
    --stage2_checkpoint_path 'ckpts/pose_only.ckpt' \
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

```
python ./code/demo_audio_generation.py \
    --infer_type 'mfcc_pose_only' \
    --stage1_checkpoint_path 'ckpts/stage1.ckpt' \
    --stage2_checkpoint_path 'ckpts/pose_only.ckpt' \
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


```
python ./code/demo_audio_generation.py \
    --infer_type 'mfcc_pose_only' \
    --stage1_checkpoint_path 'ckpts/stage1.ckpt' \
    --stage2_checkpoint_path 'ckpts/pose_only.ckpt' \
    --test_image_path 'test_demos/portraits/monalisa.jpg' \
    --test_audio_path 'test_demos/audios/english_female.wav' \
    --result_path 'results/monalisa_free_style/'
```


## Best Practise

<details><summary>Using similar poses to the portrait</summary>For example, if the face starts by rotating to the left (viewed from the portrait face angle), it is better to use a value for yaw between -1 and 0 (-90 to 0 degrees) because when the difference in angle from the portrait is large, the face can appear distorted. To automatically extract the angles of a portrait face, you can refer to this repo (https://github.com/liutaocode/talking_face_preprocessing), where we have used the same extraction algorithm.</details>


## 

There is a minor variation from the paper in that we use `MFCC` as the audio feature extractor instead of `Hubert` for two primary reasons: (1) the setup for the inference environment is simpler and quicker, and (2) we believe the slightly inferior results can reduce the risk of model misuse.



## Citation

```
@INPROCEEDINGS{
}
```

## Ackonwlegements

Stage 1, which mainly involves training motion encoder and the rendering module, extensively utilizes resources from [LIA](https://github.com/wyhsirius/LIA). The second stage of diffusion training is based on [diffae](https://github.com/phizaz/diffae) and [espnet](https://espnet.github.io/espnet/_modules/espnet2/asr/encoder/conformer_encoder.html). For the computation of mutual information loss, we implement methods from [CLUB](https://github.com/Linear95/CLUB) and employ [AAM-softmax](https://github.com/TaoRuijie/ECAPA-TDNN) in the training of face recognition. Furthermore, we utilize the pretrained Hubert model provided by [TencentGameMate](https://github.com/TencentGameMate/chinese_speech_pretrain). Additionally, we've made the dataset preprocessing codes (such as pose and landmark extraction) available at [talking_face_preprocessing](https://github.com/liutaocode/talking_face_preprocessing). We are grateful to the authors for sharing their exceptional work.