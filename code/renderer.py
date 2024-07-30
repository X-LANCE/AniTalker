from config import *

def render_condition(
    conf: TrainConfig,
    model,
    sampler, start, motion_direction_start, audio_driven, \
        face_location, face_scale, \
        yaw_pitch_roll, noisyT, control_flag,
):
    if conf.train_mode == TrainMode.diffusion:
        assert conf.model_type.has_autoenc()

        return sampler.sample(model=model,
                              noise=noisyT,
                              model_kwargs={
                                  'motion_direction_start': motion_direction_start, 
                                  'yaw_pitch_roll': yaw_pitch_roll,
                                  'start': start,
                                  'audio_driven': audio_driven,
                                  'face_location': face_location,
                                  'face_scale': face_scale,
                                  'control_flag': control_flag
                              })
    else:
        raise NotImplementedError()
