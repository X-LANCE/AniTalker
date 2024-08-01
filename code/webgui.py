import gradio as gr
import os
from demo import main
def run_lia_model(infer_type, test_image_path, test_audio_path, test_hubert_path, result_path, stage1_checkpoint_path, stage2_checkpoint_path, seed, control_flag, pose_yaw, pose_pitch, pose_roll, face_location, pose_driven_path, face_scale, step_T, image_size, device, motion_dim, decoder_layers, face_sr):
    class Args:
        pass
    
    args = Args()
    args.infer_type = infer_type
    args.test_image_path = test_image_path
    args.test_audio_path = test_audio_path
    args.test_hubert_path = test_hubert_path
    args.result_path = result_path
    args.stage1_checkpoint_path = stage1_checkpoint_path
    args.stage2_checkpoint_path = stage2_checkpoint_path
    args.seed = seed
    args.control_flag = control_flag
    args.pose_yaw = pose_yaw
    args.pose_pitch = pose_pitch
    args.pose_roll = pose_roll
    args.face_location = face_location
    args.pose_driven_path = pose_driven_path
    args.face_scale = face_scale
    args.step_T = step_T
    args.image_size = image_size
    args.device = device
    args.motion_dim = motion_dim
    args.decoder_layers = decoder_layers
    args.face_sr = face_sr
    main(args)
    return os.path.join(result_path, f"{os.path.splitext(os.path.basename(test_image_path))[0]}-{os.path.splitext(os.path.basename(test_audio_path))[0]}.mp4")

# List of checkpoint paths and inference types
stage1_checkpoint_paths = ['./ckpts/stage1.ckpt', './ckpts/another_stage1.ckpt']
stage2_checkpoint_paths = ['./ckpts/pose_only.ckpt', './ckpts/another_stage2.ckpt']
infer_types = ['mfcc_pose_only', 'mfcc_full_control', 'hubert_pose_only', 'hubert_audio_only', 'hubert_full_control']

# Define Gradio inputs
inputs = [
    gr.Dropdown(label="Inference Type", choices=infer_types, value='mfcc_pose_only'),
    gr.Image(type="filepath", label="Reference Image"),
    gr.Audio(type="filepath", label="Input Audio"),
    gr.Textbox(label="Test Hubert Path", value='./test_demos/audios_hubert/english_female.npy'),
    gr.Textbox(label="Result Path", value='./results/'),
    gr.Dropdown(label="Stage 1 Checkpoint Path", choices=stage1_checkpoint_paths, value='./ckpts/stage1.ckpt'),
    gr.Dropdown(label="Stage 2 Checkpoint Path", choices=stage2_checkpoint_paths, value='./ckpts/pose_only.ckpt'),
    gr.Slider(label="Seed", minimum=0, maximum=1000, step=1, value=0),
    gr.Checkbox(label="Control Flag", value=False),
    gr.Slider(label="Pose Yaw", minimum=-1, maximum=1, step=0.01, value=0.25),
    gr.Slider(label="Pose Pitch", minimum=-1, maximum=1, step=0.01, value=0),
    gr.Slider(label="Pose Roll", minimum=-1, maximum=1, step=0.01, value=0),
    gr.Slider(label="Face Location", minimum=0, maximum=1, step=0.01, value=0.5),
    gr.Textbox(label="Pose Driven Path", value='xxx'),
    gr.Slider(label="Face Scale", minimum=0, maximum=1, step=0.01, value=0.5),
    gr.Slider(label="Step T", minimum=0, maximum=100, step=1, value=50),
    gr.Slider(label="Image Size", minimum=0, maximum=512, step=1, value=256),
    gr.Textbox(label="Device", value='cuda:0'),
    gr.Slider(label="Motion Dimension", minimum=0, maximum=100, step=1, value=20),
    gr.Slider(label="Decoder Layers", minimum=0, maximum=10, step=1, value=2),
    gr.Checkbox(label="Face Super Resolution", value=False)
]

outputs = gr.Video()

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            uploaded_img = gr.Image(type="filepath", label="Reference Image")
            uploaded_audio = gr.Audio(type="filepath", label="Input Audio")
        with gr.Column():
            result_video = gr.Video()
    submit_button = gr.Button("Run")
    infer_type = gr.Dropdown(label="Inference Type", choices=infer_types, value='hubert_audio_only')
    stage1_checkpoint_path = gr.Dropdown(label="Stage 1 Checkpoint Path", choices=stage1_checkpoint_paths, value='./ckpts/stage1.ckpt')
    stage2_checkpoint_paths.extend([
        './ckpts/stage2_audio_only_hubert.ckpt',
        './ckpts/stage2_full_control_hubert.ckpt',
        './ckpts/stage2_full_control_mfcc.ckpt',
        './ckpts/stage2_pose_only_hubert.ckpt',
        './ckpts/stage2_pose_only_mfcc.ckpt'
    ])
    stage2_checkpoint_path = gr.Dropdown(
        label="Stage 2 Checkpoint Path",
        choices=stage2_checkpoint_paths,
        value='./ckpts/stage2_audio_only_hubert.ckpt'
    )
    result_path = gr.Textbox(label="Result Path", value='./results/')
    test_hubert_path = gr.Textbox(label="Hubert Path", value='')
    seed = gr.Slider(label="Seed", minimum=0, maximum=1000, step=1, value=0)
    control_flag = gr.Checkbox(label="Control Flag", value=False)
    pose_yaw = gr.Slider(label="Pose Yaw", minimum=-1, maximum=1, step=0.01, value=0.25)
    pose_pitch = gr.Slider(label="Pose Pitch", minimum=-1, maximum=1, step=0.01, value=0)
    pose_roll = gr.Slider(label="Pose Roll", minimum=-1, maximum=1, step=0.01, value=0)
    face_location = gr.Slider(label="Face Location", minimum=0, maximum=1, step=0.01, value=0.5)
    pose_driven_path = gr.Textbox(label="Pose Driven Path", value='xxx')
    face_scale = gr.Slider(label="Face Scale", minimum=0, maximum=1, step=0.01, value=0.5)
    step_T = gr.Slider(label="Step T", minimum=0, maximum=100, step=1, value=50)
    image_size = gr.Slider(label="Image Size", minimum=0, maximum=512, step=1, value=256)
    device = gr.Textbox(label="Device", value='cuda:0')
    motion_dim = gr.Slider(label="Motion Dimension", minimum=0, maximum=100, step=1, value=20)
    decoder_layers = gr.Slider(label="Decoder Layers", minimum=0, maximum=10, step=1, value=2)
    face_sr = gr.Checkbox(label="Face Super Resolution", value=False)
    submit_button.click(run_lia_model, inputs=[
        infer_type, uploaded_img, uploaded_audio, test_hubert_path, result_path, stage1_checkpoint_path, stage2_checkpoint_path, seed, control_flag, pose_yaw, pose_pitch, pose_roll, face_location, pose_driven_path, face_scale, step_T, image_size, device, motion_dim, decoder_layers, face_sr
    ], outputs=result_video)


demo.launch()

