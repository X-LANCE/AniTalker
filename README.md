
# AniTalker: Animate Vivid and Diverse Talking Faces through Identity-Decoupled Facial Motion Encoding

<p align="center">
  <a href="https://x-lance.github.io/AniTalker/">Demo</a> &nbsp;&nbsp;&nbsp; <a href="https://arxiv.org/abs/2405.03121">Paper</a> &nbsp;&nbsp;&nbsp; <a href="https://github.com/X-LANCE/AniTalker">Code</a>
</p>


![](docs/img/generated_result.png)

* The weights and code are being organized, and we will make them public as soon as possible.

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
    --result_path 'results/monalisa_case1/' \
    --control_flag True \
    --seed 0 \
    --pose_yaw 0 \
    --pose_pitch 0 \
    --pose_roll 0 
```


### Adjust the orentation 

Changing pose_yaw from `0` to `0.25`

![monalisa_turn_head_right](assets/monalisa_turn_head_right.gif)

Demo script:

```
python ./code/demo.py \
    --infer_type 'mfcc_pose_only' \
    --stage1_checkpoint_path 'ckpts/stage1.ckpt' \
    --stage2_checkpoint_path 'ckpts/stage2_pose_only.ckpt' \
    --test_image_path 'test_demos/portraits/monalisa.jpg' \
    --test_audio_path 'test_demos/audios/english_female.wav' \
    --result_path 'results/monalisa_case2/' \
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
python ./code/demo.py \
    --infer_type 'mfcc_pose_only' \
    --stage1_checkpoint_path 'ckpts/stage1.ckpt' \
    --stage2_checkpoint_path 'ckpts/stage2_pose_only.ckpt' \
    --test_image_path 'test_demos/portraits/monalisa.jpg' \
    --test_audio_path 'test_demos/audios/english_female.wav' \
    --result_path 'results/monalisa_case3/'
```

### More Scripts

See [MORE_SCRIPTS](MORE_SCRIPTS.md)


## Some Advice and Questions

<details><summary>1. Using similar poses to the portrait (Best Practice)</summary>
To avoid potential deformation issues, it is recommended to keep the generated face angle close to the original portrait angle. For instance, if the face in the portrait is initially rotated to the left, it is advisable to use a value for yaw between -1 and 0 (-90 to 0 degrees). When the difference in angle from the portrait is significant, the generated face may appear distorted.
</details>

<details><summary>2. Utilizing algorithms to automatically extract or control using other faces' angles</summary>
If you need to automate face control, you can employ pose extraction algorithms to achieve this, such as extracting the pose of another person to drive the portrait. The algorithms for extraction have been open-sourced and can be found at <a href="https://github.com/liutaocode/talking_face_preprocessing?tab=readme-ov-file#face-orientation-angles">this link</a>.
</details>

<details><summary>3. What are the differences between MFCC and Hubert features?</summary>
Both `MFCC` and `Hubert` are front-end features for speech, used to extract audio signals. However, `Hubert` features require more environmental dependencies and occupy a significant amount of disk space. To facilitate quick inference for everyone, we have replaced this feature with a lightweight alternative (MFCC). The rest of the code remains unchanged. We have observed that MFCC converges more easily but may be inferior in terms of expressiveness compared to Hubert. If you need to extract Hubert features, please refer to <a href="https://github.com/liutaocode/talking_face_preprocessing?tab=readme-ov-file#audio-feature-extraction">this link</a>. Considering the highly lifelike nature of the generated results, we currently do not plan to release the weights based on Hubert.
</details>


## Citation

```
@misc{liu2024anitalker,
      title={AniTalker: Animate Vivid and Diverse Talking Faces through Identity-Decoupled Facial Motion Encoding}, 
      author={Tao Liu and Feilong Chen and Shuai Fan and Chenpeng Du and Qi Chen and Xie Chen and Kai Yu},
      year={2024},
      eprint={2405.03121},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgments

We would like to express our sincere gratitude to the numerous prior works that have laid the foundation for the development of AniTalker.

Stage 1, which primarily focuses on training the motion encoder and the rendering module, heavily relies on resources from [LIA](https://github.com/wyhsirius/LIA). The second stage of diffusion training is built upon [diffae](https://github.com/phizaz/diffae) and [espnet](https://espnet.github.io/espnet/_modules/espnet2/asr/encoder/conformer_encoder.html). For the computation of mutual information loss, we implement methods from [CLUB](https://github.com/Linear95/CLUB) and utilize [AAM-softmax](https://github.com/TaoRuijie/ECAPA-TDNN) in the training of face recognition. Moreover, we leverage the pretrained Hubert model provided by [TencentGameMate](https://github.com/TencentGameMate/chinese_speech_pretrain).

Additionally, we employ [3DDFA_V2](https://github.com/cleardusk/3DDFA_V2) to extract head pose and [torchlm](https://github.com/DefTruth/torchlm) to obtain face landmarks, which are used to calculate face location and scale. We have already open-sourced the code usage for these preprocessing steps at [talking_face_preprocessing](https://github.com/liutaocode/talking_face_preprocessing). We acknowledge the importance of building upon existing knowledge and are committed to contributing back to the research community by sharing our findings and code.


## Disclaimer

1. This library's code is not a formal product, and we have not tested all use cases; therefore, it cannot be directly offered to end-service customers.

2. The main purpose of making our code public is to facilitate academic demonstrations and communication. Any use of this code to spread harmful information is strictly prohibited.

3. Please use this library in compliance with the terms specified in the license file and avoid improper use.

4. When using the code, please follow and abide by local laws and regulations.

5. During the use of this code, you will bear the corresponding responsibility. Our company (AISpeech Ltd.) is not responsible for the generated results.
