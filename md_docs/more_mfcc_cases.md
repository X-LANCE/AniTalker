## More MFCC cases

### Storytelling (Chinese)

```
python ./code/demo.py \
    --infer_type 'mfcc_pose_only' \
    --stage1_checkpoint_path 'ckpts/stage1.ckpt' \
    --stage2_checkpoint_path 'ckpts/stage2_pose_only_mfcc.ckpt' \
    --test_image_path 'test_demos/portraits/sad.jpg' \
    --test_audio_path 'test_demos/audios/lianliru.wav' \
    --test_hubert_path 'test_demos/audios_hubert/lianliru.npy' \
    --result_path 'outputs/lianliru_mfcc/' \
    --control_flag True \
    --seed 0 \
    --pose_yaw -0.0427 \
    --pose_pitch -0.0536 \
    --pose_roll 0.0434
```

- The generated video of this sample will be saved to [outputs/lianliru_mfcc/sad-lianliru.mp4](../outputs/lianliru_mfcc/sad-lianliru.mp4).
- 'lianliru.wav' is from [StoryTTS](https://github.com/X-LANCE/StoryTTS) dataset.

## Einstein

```
python ./code/demo.py \
    --infer_type 'mfcc_pose_only' \
    --stage1_checkpoint_path 'ckpts/stage1.ckpt' \
    --stage2_checkpoint_path 'ckpts/stage2_pose_only_mfcc.ckpt' \
    --test_image_path 'test_demos/portraits/Einstein.png' \
    --test_audio_path 'test_demos/audios/english_male.mp3' \
    --test_hubert_path 'test_demos/audios_hubert/english_male.npy' \
    --result_path 'outputs/Einstein_mfcc/' \
    --control_flag True \
    --seed 0 \
    --pose_yaw 0.0277 \
    --pose_pitch 0.0252 \
    --pose_roll 0.0308
```


- The generated video of this sample will be saved to [outputs/Einstein_mfcc/Einstein-english_male.mp4](../outputs/Einstein_mfcc/Einstein-english_male.mp4).
- Image of `Einstein.png` is from [GAIA](https://gaiavatar.github.io/gaia/)
- There is a bad case occurring in the middle of the generated video; this may be caused by the lack of robustness of MFCC.


## Long Story Generation

```
python ./code/demo.py \
    --infer_type 'mfcc_pose_only' \
    --stage1_checkpoint_path 'ckpts/stage1.ckpt' \
    --stage2_checkpoint_path 'ckpts/stage2_pose_only_mfcc.ckpt' \
    --test_image_path 'test_demos/portraits/cartoon_girl.png' \
    --test_audio_path 'test_demos/audios/mars.wav' \
    --test_hubert_path 'test_demos/audios_hubert/mars.npy' \
    --result_path 'outputs/cartoon_girl_mars_story_mfcc/' \
    --control_flag True \
    --seed 0 \
    --pose_yaw 0.0302 \
    --pose_pitch 0.164 \
    --pose_roll 0.0415


```
- The generated video of this sample will be saved to [outputs/cartoon_girl_mars_story_mfcc/cartoon_girl-mars.mp4](../outputs/cartoon_girl_mars_story_mfcc/cartoon_girl-mars.mp4).

## Statue

```
python ./code/demo.py \
    --infer_type 'mfcc_pose_only' \
    --stage1_checkpoint_path 'ckpts/stage1.ckpt' \
    --stage2_checkpoint_path 'ckpts/stage2_pose_only_mfcc.ckpt' \
    --test_image_path 'test_demos/portraits/statue.jpg' \
    --test_audio_path 'test_demos/audios/statue.wav' \
    --test_hubert_path 'test_demos/audios_hubert/statue.npy' \
    --result_path 'outputs/statue_mfcc/' \
    --control_flag True \
    --seed 0 \
    --pose_yaw -0.0363 \
    --pose_pitch 0.0123 \
    --pose_roll -0.0031

```
- The generated video of this sample will be saved to [outputs/statue_mfcc/statue-statue.mp4](../outputs/statue_mfcc/statue-statue.mp4).

