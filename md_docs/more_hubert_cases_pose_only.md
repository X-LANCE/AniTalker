## More hubert cases (Pose-controllable Model)

**Features of this model include:**
- The driving signals require one image plus an audio segment.
- You can also adjust the pose_yaw, pose_pitch, and pose_roll.
- It offers moderate visual stability.
- The expressiveness is better.

### Storytelling (Chinese)

```
python ./code/demo.py \
    --infer_type 'hubert_pose_only' \
    --stage1_checkpoint_path 'ckpts/stage1.ckpt' \
    --stage2_checkpoint_path 'ckpts/stage2_pose_only_hubert.ckpt' \
    --test_image_path 'test_demos/portraits/sad.jpg' \
    --test_audio_path 'test_demos/audios/lianliru.wav' \
    --test_hubert_path 'test_demos/audios_hubert/lianliru.npy' \
    --result_path 'outputs/lianliru_hubert_with_pose/' \
    --control_flag \
    --seed 0 \
    --pose_yaw 0 \
    --pose_pitch 0 \
    --pose_roll 0 \
    --face_sr
```

- The generated video of this sample will be saved to [outputs/lianliru_hubert_with_pose/girl-lianliru.mp4](../outputs/lianliru_hubert_with_pose/girl-lianliru.mp4).
- 'lianliru.wav' is from [StoryTTS](https://github.com/X-LANCE/StoryTTS) dataset.

## Einstein

```
python ./code/demo.py \
    --infer_type 'hubert_pose_only' \
    --stage1_checkpoint_path 'ckpts/stage1.ckpt' \
    --stage2_checkpoint_path 'ckpts/stage2_pose_only_hubert.ckpt' \
    --test_image_path 'test_demos/portraits/Einstein.png' \
    --test_audio_path 'test_demos/audios/english_male.mp3' \
    --test_hubert_path 'test_demos/audios_hubert/english_male.npy' \
    --result_path 'outputs/Einstein_hubert_pose/' \
    --control_flag \
    --seed 0 \
    --pose_yaw 0 \
    --pose_pitch 0 \
    --pose_roll 0 \
    --face_sr
```


- The generated video of this sample will be saved to [outputs/Einstein_hubert_pose/Einstein-english_male.mp4](../outputs/Einstein_hubert_pose/Einstein-english_male.mp4).
- Image of `Einstein.png` is from [GAIA](https://gaiavatar.github.io/gaia/)


## Long Story Generation

```
python ./code/demo.py \
    --infer_type 'hubert_pose_only' \
    --stage1_checkpoint_path 'ckpts/stage1.ckpt' \
    --stage2_checkpoint_path 'ckpts/stage2_pose_only_hubert.ckpt' \
    --test_image_path 'test_demos/portraits/cartoon_girl.png' \
    --test_audio_path 'test_demos/audios/mars.wav' \
    --test_hubert_path 'test_demos/audios_hubert/mars.npy' \
    --result_path 'outputs/cartoon_girl_mars_story_hubert_pose/' \
    --control_flag \
    --seed 0 \
    --pose_yaw 0 \
    --pose_pitch 0 \
    --pose_roll 0 \
    --face_sr

```
- The generated video of this sample will be saved to [outputs/cartoon_girl_mars_story_hubert_pose/cartoon_girl-mars.mp4](../outputs/cartoon_girl_mars_story_hubert_pose/cartoon_girl-mars.mp4).

## Statue

```
python ./code/demo.py \
    --infer_type 'hubert_pose_only' \
    --stage1_checkpoint_path 'ckpts/stage1.ckpt' \
    --stage2_checkpoint_path 'ckpts/stage2_pose_only_hubert.ckpt' \
    --test_image_path 'test_demos/portraits/statue.jpg' \
    --test_audio_path 'test_demos/audios/statue.wav' \
    --test_hubert_path 'test_demos/audios_hubert/statue.npy' \
    --result_path 'outputs/statue_hubert_pose/' \
    --control_flag \
    --seed 0 \
    --pose_yaw 0 \
    --pose_pitch 0 \
    --pose_roll 0 \
    --face_sr

```
- The generated video of this sample will be saved to [outputs/statue_hubert_pose/statue-statue.mp4](../outputs/statue_hubert_pose/statue-statue.mp4).



