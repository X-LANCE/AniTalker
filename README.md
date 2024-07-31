
# AniTalker: Animate Vivid and Diverse Talking Faces through Identity-Decoupled Facial Motion Encoding

<p align="center">
  <a href="https://x-lance.github.io/AniTalker/">Demo</a> &nbsp;&nbsp;&nbsp; <a href="https://arxiv.org/abs/2405.03121">Paper</a> &nbsp;&nbsp;&nbsp; <a href="https://github.com/X-LANCE/AniTalker">Code</a>
</p>


An updated version of the paper will be uploaded later

![](docs/img/generated_result.png)

[Overall Pipeline](md_docs/overall_pipeline.md)

## Updates

- [2024.07.31] Added hubert feature extraction code and environment

## Environment Installation

```shell
conda create -n anitalker python==3.9.0
conda activate anitalker
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install -r requirements.txt
```

[Windows Installation Tutorial](https://www.youtube.com/watch?v=x1ZP3e830DY) Thanks to [nitinmukesh](https://github.com/nitinmukesh)

## Model Zoo

Please download the checkpoint from [URL](https://huggingface.co/taocode/anitalker_ckpts/tree/main) and place them into the folder ```ckpts```

[中文用户] For Chinese users, we recommend you visit [here](https://pan.baidu.com/s/1gqTPmoJ3QwKbGkqgMXM3Jw?pwd=antk) to download.

```
ckpts/
├── chinese-hubert-large
├──── config.json
├──── preprocessor_config.json
├──── pytorch_model.bin
├── stage1.ckpt
├── stage2_pose_only_mfcc.ckpt
├── stage2_full_control_mfcc.ckpt
├── stage2_audio_only_hubert.ckpt
├── stage2_pose_only_hubert.ckpt
└── stage2_full_control_hubert.ckpt
```

**Model Description:**

| Stage | Model Name | Audio-only Inference | Addtional Control Signal | 
| --- | --- | --- | --- |
| First stage | stage1.ckpt | - | Motion Encoder & Image Renderer | 
| Second stage (Hubert) | stage2_audio_only_hubert.ckpt | yes | - | 
| Second stage (Hubert) | stage2_pose_only_hubert.ckpt | yes | Head Pose |
| Second stage (Hubert) | stage2_full_control_hubert.ckpt | yes | Head Pose/Location/Scale | 
| Second stage (MFCC) | stage2_pose_only_mfcc.ckpt | yes |  Head Pose | 
| Second stage (MFCC) | stage2_full_control_mfcc.ckpt | yes | Head Pose/Location/Scale | 

- `stage1.ckpt` is trained on a single image video dataset, aiming to learn the transfer of actions. After training, it utilizes the Motion Encoder (for extracting identity-independent motion) and Image Renderer.
- The models starting with `stage2` are trained on a video dataset with audio, and unless otherwise specified, are trained from scratch.
- `stage2_audio_only_hubert.ckpt` inputs audio features as Hubert, without any control signals. Suitable for scenes with faces oriented forward, compared to controllable models, it requires less parameter adjustment to achieve satisfactory results. [We recommend starting with this model]
- `stage2_pose_only_hubert.ckpt` is similar to `stage2_pose_only_mfcc.ckpt`, the difference being that the audio features are Hubert. Compared to the audio_only model, it includes pose control signals.
- `stage2_more_controllable_hubert.ckpt` is similar to `stage2_more_controllable_mfcc.ckpt`, but uses Hubert for audio features. 
- `stage2_pose_only_mfcc.ckpt` inputs audio features as MFCC, and includes pose control signals (yaw, pitch, roll angles).  [The performance of the MFCC model is poor and not recommended for use.]
- `stage2_more_controllable_mfcc.ckpt` inputs audio features as MFCC, and adds control signals for face location and face scale in addition to pose.

**Quick Guide:**

* If you want to quickly experience the entire algorithm process, please use the model with the MFCC suffix.
* If you desire better performance and are willing to endure greater resource consumption and deployment complexity, please use the model with the Hubert suffix.
* If you need more control, please use the model with the controllable suffix. Controllable models often have better expressiveness but requiring more parameter adjustment.
* Considering usability and model performance, we recommend using `stage2_audio_only_hubert.ckpt`.
* All stage2 models can also be generated solely by audio if the control flag is disabled.

## Run the demo

[Explanation of Parameters for demo.py](md_docs/config.md)


### Main Inference Scripts (Hubert, Better Result 💪) - Recommended

```
python ./code/demo.py \
    --infer_type 'hubert_audio_only' \
    --stage1_checkpoint_path 'ckpts/stage1.ckpt' \
    --stage2_checkpoint_path 'ckpts/stage2_audio_only_hubert.ckpt' \
    --test_image_path 'test_demos/portraits/monalisa.jpg' \
    --test_audio_path 'test_demos/audios/monalisa.wav' \
    --test_hubert_path 'test_demos/audios_hubert/monalisa.npy' \
    --result_path 'outputs/monalisa_hubert/' 
```

[See More Hubert Cases](md_docs/more_hubert_cases_audio_only.md)

- The generated video of this sample will be saved to [outputs/monalisa_hubert/monalisa-monalisa.mp4](outputs/monalisa_hubert/monalisa-monalisa.mp4).

- For Pose Controllable Hubert Cases, see [more_hubert_cases_pose_only](md_docs/more_hubert_cases_pose_only.md).

- For Pose/Face Controllable Hubert Cases, see [more_hubert_cases_more_control](md_docs/more_hubert_cases_more_control.md).


| Source Img | Results           | 
|------------|--------------------------|
|<img src="test_demos/portraits/monalisa.jpg" width="200" ></img> | <img src="assets/monalisa-monalisa.gif" width="200" ></img> | 


### Main Inference Scripts (MFCC, Faster 🚀) - Not Recommended

[Note] The Hubert model is our default model. For environment convenience, we provide an MFCC version, but we found that the utilization rate of the Hubert model is not high, and people still use MFCC more often. MFCC has poorer results. This goes against our original intention, so we have deprecated this model. We recommend you start testing with the hubert_audio_only model. Thanks.

[Upgrade for Early Users] Re-download the checkpoint with the Hubert model into the ckpts directory and additionally install `pip install transformers==4.19.2`. When the code does not detect the Hubert path, it will automatically extract it and provide extra instructions on how to resolve any errors encountered.

<details><summary>Still Show Original MFCC Scripts</summary>
```
python ./code/demo.py \
    --infer_type 'mfcc_pose_only' \
    --stage1_checkpoint_path 'ckpts/stage1.ckpt' \
    --stage2_checkpoint_path 'ckpts/stage2_pose_only_mfcc.ckpt' \
    --test_image_path 'test_demos/portraits/monalisa.jpg' \
    --test_audio_path 'test_demos/audios/monalisa.wav' \
    --result_path 'outputs/monalisa_mfcc/' \
    --control_flag \
    --seed 0 \
    --pose_yaw 0.25 \
    --pose_pitch 0 \
    --pose_roll 0 
```
</details>

### Face Super-resolution (Optional)

The purpose is to upscale the resolution from 256 to 512 and address the issue of blurry rendering.


Please install addtional environment here:

```
pip install facexlib
pip install tb-nightly -i https://mirrors.aliyun.com/pypi/simple
pip install gfpgan

# Ignore the following warning:
# espnet 202301 requires importlib-metadata<5.0, but you have importlib-metadata 7.1.0 which is incompatible.
```

Then enable the option `--face_sr` in your scripts. The first time will download the weights of gfpgan.


## Best Practice 

<details><summary>1. Using similar poses to the portrait</summary>
[This recommendation is applicable only to models with pose control] To avoid potential deformation issues, primarily caused by 2D-wrapping, it is recommended to keep the generated face angle close to the original portrait angle or allow for only slight angle variations. For instance, if the face in the portrait is initially rotated to the left, it is advisable to use a face rotated to the left to obtain the best results. Specifically, a face rotated to the left is equivalent to adjusting the yaw value to be between -1 and 0 (representing a change of -90 to 0 yaw degrees). Particularly for the HDTF dataset, it is recommended to keep pose_yaw, pose_pitch, and pose_roll at 0, as this dataset mainly consists of frontal faces.
</details>

<details><summary>2. Keep the head centered in the frame</summary>
Our model was trained by first detecting the face and then estimating the head's position, with most heads located in the center of the frame. Therefore, please keep the head as centered as possible (this is different from face alignment). Refer to the specific cropping positions in the [facial cropping code](https://github.com/liutaocode/talking_face_preprocessing?tab=readme-ov-file#facial-part-cropping). Placing the head in other positions or having an overly large face might cause some distortions.
</details>

<details><summary>3. Using English Speech</summary>
Our model is primarily trained on English speech content, with minimal exposure to other languages. Therefore, one of the best practice is to use English audio for driving the model. If other languages are used, we have observed issues such as the generated lips moving slightly or causing image distortion and deformation.
</details>

<details><summary>4. Enhancing Immersion</summary>
We have found that visually focusing straight ahead can greatly enhance the immersion of the generated video. Since our model does not model gaze, it sometimes results in a vacant or unfocused look, which can lead to poor immersion. We suggest controlling the signal to ensure that the gaze is directed forward as much as possible. Additionally, it might be beneficial to refer to some gaze modeling algorithms, such as [vasa-1](https://www.microsoft.com/en-us/research/project/vasa-1/) and [PD-FGC](https://github.com/Dorniwang/PD-FGC-inference), recommended by [tanshuai0219](https://github.com/tanshuai0219), for potential improvements in gaze modeling.

</details>

## TODO 

- [ ] Consider adding automatic calibration for the first frame, taking into account that the initial position and shape of the face have a certain impact on the results.

We welcome any contributions to the repository.

## Questions

<details><summary>1. What are the differences between MFCC and Hubert features?</summary>
Both `MFCC` and `Hubert` are types of front-end features used in speech analysis to extract characteristics from audio signals. `Hubert` features, while robust, require extensive environmental dependencies and consume a considerable amount of disk space. Consequently, for efficiency and to ensure quick inference accessibility for all users, we have substituted `Hubert` features with the more lightweight `MFCC`. Although `MFCC` features are easier to converge, they are less expressive and perform less effectively in cross-language inference compared to `Hubert`. Additionally, artifacts such as jitter, excessive smoothness, or overly exaggerated expressions may occur in Text-to-Speech (TTS) audio and silent segments.
</details>

<details><summary>2. How to apply to higher resolution?</summary>
The base resolution of our generated output is only 256×256. If you require higher resolution (e.g., 512×512), you can refer to the super-resolution module in <a href="https://github.com/OpenTalker/SadTalker">SadTalker</a>. By incorporating the super-resolution module at the end of the aforementioned pipeline, you can achieve higher resolution output. We also integrate this module in our code.
</details>

<details><summary> 3. How to automatically extract or control using other faces' angles? </summary>
If you need to automate face control, you can employ pose extraction algorithms to achieve this, such as extracting the pose of another person to drive the portrait. The algorithms for extraction pipeline have been open-sourced and can be found at <a href="https://github.com/liutaocode/talking_face_preprocessing?tab=readme-ov-file#face-orientation-angles">this link</a>.
</details>

<details><summary>4. Will other resources be released in the future?</summary>
Due to the potential risks and ethical concerns associated with life-like models, we have currently decided not distributing additional resources, such as training scripts, and other checkpoint. We apologize for any inconvenience this decision may cause.
</details>

## Model Bias / Limitations

Regarding the checkpoints provided by this library, the issues we encountered while testing various audio clips and images have revealed model biases. These biases are primarily due to the training dataset or the model capacity, including but not limited to the following:

- The dataset processes the face and its surrounding areas, not involving the full body or upper body.
- The dataset predominantly contains English, with limited instances of non-English or dialects.
- The dataset uses relatively ideal conditions in processing, not accounting for dramatic changes.
- The dataset primarily focuses on speech content at a normal speaking pace, not considering different speech rates or non-speech scenarios.
- The dataset has been exposed only to specific demographics, potentially causing biases against different ethnic groups or age groups.
- Rendering models are unable to model multi-view objects. Sometimes they fail to separate characters from the background, especially accessories and hairstyles of the characters which cannot be effectively isolated.

**Please generate content carefully based on the above considerations.**

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

Stage 1, which primarily focuses on training the motion encoder and the rendering module, heavily relies on resources from [LIA](https://github.com/wyhsirius/LIA). The second stage of diffusion training is built upon [diffae](https://github.com/phizaz/diffae) and [espnet](https://espnet.github.io/espnet/_modules/espnet2/asr/encoder/conformer_encoder.html). For the computation of mutual information loss, we implement methods from [CLUB](https://github.com/Linear95/CLUB) and utilize [AAM-softmax](https://github.com/TaoRuijie/ECAPA-TDNN) in the training of identity network. Moreover, we leverage the pretrained Hubert model provided by [TencentGameMate](https://github.com/TencentGameMate/chinese_speech_pretrain) and mfcc feature from [MFCC](https://github.com/jameslyons/python_speech_features).

Additionally, we employ [3DDFA_V2](https://github.com/cleardusk/3DDFA_V2) to extract head pose and [torchlm](https://github.com/DefTruth/torchlm) to obtain face landmarks, which are used to calculate face location and scale. We have already open-sourced the code usage for these preprocessing steps at [talking_face_preprocessing](https://github.com/liutaocode/talking_face_preprocessing). We acknowledge the importance of building upon existing knowledge and are committed to contributing back to the research community by sharing our findings and code.


## Disclaimer

```
1. This library's code is not a formal product, and we have not tested all use cases; therefore, it cannot be directly offered to end-service customers.

2. The main purpose of making our code public is to facilitate academic demonstrations and communication. Any use of this code to spread harmful information is strictly prohibited.

3. Please use this library in compliance with the terms specified in the license file and avoid improper use.

4. When using the code, please follow and abide by local laws and regulations.

5. During the use of this code, you will bear the corresponding responsibility. Our company (AISpeech Ltd.) is not responsible for the generated results.
```