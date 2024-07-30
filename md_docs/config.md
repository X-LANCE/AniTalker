# Explanation of Parameters for demo.py

| index | Name | Type  | Description | 
| --- | --- | --- | --- |
| 1 | infer_type | String | Single choices: ['mfcc_pose_only', 'mfcc_full_control', 'hubert_audio_only', 'hubert_pose_only'] | 
| 2 | test_image_path | String | Path to the portrait (.jpg or .png) | 
| 3 | test_audio_path | String | Path to the driven audio (.wav or .mp3)  | 
| 4 | test_hubert_path | String | Path to the Hubert feature of the driven audio (.npy). Not needed for MFCC model  | 
| 5 | result_path | String | The result will be saved to this folder  | 
| 6 | stage1_checkpoint_path | String | The model path for the first stage. Fixed to ./ckpts/stage1.ckpt in our experiment  | 
| 7 | stage2_checkpoint_path | String | The model path for the second stage. This model will change with infer_type, see the table below for specific relationships | 
| 8 | seed | integer | The seed of the second model to control diversity.  | 
| 9 | control_flag | boolean | Whether to enable control signals, does not work for audio-only models.  | 
| 10 | pose_yaw | float | Yaw angle for head pose. Already normalized, ranging from -1 to 1, representing -90° to 90°, only effective when control_flag is enabled and for pose designated models | 
| 11 | pose_pitch | float | Pitch angle for head pose. Already normalized, ranging from -1 to 1, representing -90° to 90°, only effective when control_flag is enabled and for pose designated models | 
| 12 | pose_roll | float | Roll angle for head pose. Already normalized, ranging from -1 to 1, representing -90° to 90°, only effective when control_flag is enabled and for pose designated models | 
| 13 | pose_driven_path | str | [Optional] Path to pose numpy, shape is (T, 3). You can check the following code https://github.com/liutaocode/talking_face_preprocessing to extract the yaw, pitch and roll. | 
| 14 | face_location | float | x-coordinate of the nose (screen coordinate). Already normalized, ranging from 0 to 1, representing from the leftmost to the rightmost of the screen, with most values centered around 0.5 (i.e., centered), effective when control_flag is enabled and only for models with the more_controllable identifier | 
| 15 | face_scale | float | Size of the face (or distance from the camera). Already normalized, ranging from 0 to 1, representing the size of the face, effective when control_flag is enabled and only for models with the more_controllable identifier | 
| 16 | step_T | integer | Number of diffusion denoising steps, default 50.  | 
| 17 | image_size | integer | Image size, currently only supports 256 in our experiment | 
| 18 | motion_dim | integer | Dimension of motion, currently only supports 20 in our experiment | 

Current model mapping table:

| index | infer_type | stage2_checkpoint_name | 
| --- | --- | --- | 
| 1 | mfcc_pose_only | stage2_pose_only_mfcc.ckpt | 
| 2 | mfcc_full_control | stage2_more_controllable_mfcc.ckpt | 
| 3 | hubert_audio_only | stage2_audio_only_hubert.ckpt | 
| 4 | hubert_pose_only | stage2_pose_only_hubert.ckpt |