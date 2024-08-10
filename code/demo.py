from LIA_Model import LIA_Model
import torch
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import argparse
import numpy as np
from torchvision import transforms
from templates import *
import argparse
import shutil
from moviepy.editor import *
import librosa
import python_speech_features
import importlib.util
import time

def check_package_installed(package_name):
    package_spec = importlib.util.find_spec(package_name)
    if package_spec is None:
        print(f"{package_name} is not installed.")
        return False
    else:
        print(f"{package_name} is installed.")
        return True

def frames_to_video(input_path, audio_path, output_path, fps=25):
    image_files = [os.path.join(input_path, img) for img in sorted(os.listdir(input_path))]
    clips = [ImageClip(m).set_duration(1/fps) for m in image_files]
    video = concatenate_videoclips(clips, method="compose")

    audio = AudioFileClip(audio_path)
    final_video = video.set_audio(audio)
    final_video.write_videofile(output_path, fps=fps, codec='libx264', audio_codec='aac')

def load_image(filename, size):
    img = Image.open(filename).convert('RGB')
    img = img.resize((size, size))
    img = np.asarray(img)
    img = np.transpose(img, (2, 0, 1))  # 3 x 256 x 256
    return img / 255.0

def img_preprocessing(img_path, size):
    img = load_image(img_path, size)  # [0, 1]
    img = torch.from_numpy(img).unsqueeze(0).float()  # [0, 1]
    imgs_norm = (img - 0.5) * 2.0  # [-1, 1]
    return imgs_norm

def saved_image(img_tensor, img_path):
    toPIL = transforms.ToPILImage()
    img = toPIL(img_tensor.detach().cpu().squeeze(0))  # 使用squeeze(0)来移除批次维度
    img.save(img_path)

def main(args):
    frames_result_saved_path = os.path.join(args.result_path, 'frames')
    os.makedirs(frames_result_saved_path, exist_ok=True)
    test_image_name = os.path.splitext(os.path.basename(args.test_image_path))[0]
    audio_name = os.path.splitext(os.path.basename(args.test_audio_path))[0]
    predicted_video_256_path = os.path.join(args.result_path,  f'{test_image_name}-{audio_name}.mp4')
    predicted_video_512_path = os.path.join(args.result_path,  f'{test_image_name}-{audio_name}_SR.mp4')
 
    #======Loading Stage 1 model=========
    lia = LIA_Model(motion_dim=args.motion_dim, fusion_type='weighted_sum')
    lia.load_lightning_model(args.stage1_checkpoint_path)
    lia.to(args.device)
    #============================

    conf = ffhq256_autoenc()
    conf.seed = args.seed
    conf.decoder_layers = args.decoder_layers
    conf.infer_type = args.infer_type
    conf.motion_dim = args.motion_dim
    
    if args.infer_type == 'mfcc_full_control':
        conf.face_location=True
        conf.face_scale=True
        conf.mfcc = True

    elif args.infer_type == 'mfcc_pose_only':
        conf.face_location=False
        conf.face_scale=False
        conf.mfcc = True

    elif args.infer_type == 'hubert_pose_only':
        conf.face_location=False
        conf.face_scale=False
        conf.mfcc = False

    elif args.infer_type == 'hubert_audio_only':
        conf.face_location=False
        conf.face_scale=False
        conf.mfcc = False

    elif args.infer_type == 'hubert_full_control':
        conf.face_location=True
        conf.face_scale=True
        conf.mfcc = False

    else:
        print('Type NOT Found!')
        exit(0)
        
    if not os.path.exists(args.test_image_path):
        print(f'{args.test_image_path} does not exist!')
        exit(0)
    
    if not os.path.exists(args.test_audio_path):
        print(f'{args.test_audio_path} does not exist!')
        exit(0)
    
    img_source = img_preprocessing(args.test_image_path, args.image_size).to(args.device)
    one_shot_lia_start, one_shot_lia_direction, feats = lia.get_start_direction_code(img_source, img_source, img_source, img_source)


    #======Loading Stage 2 model=========
    model = LitModel(conf)
    state = torch.load(args.stage2_checkpoint_path, map_location='cpu')
    model.load_state_dict(state, strict=True)
    model.ema_model.eval()
    model.ema_model.to(args.device);
    #=================================
    
    
    #======Audio Input=========
    if conf.infer_type.startswith('mfcc'):
        # MFCC features
        wav, sr = librosa.load(args.test_audio_path, sr=16000)
        input_values = python_speech_features.mfcc(signal=wav, samplerate=sr, numcep=13, winlen=0.025, winstep=0.01)
        d_mfcc_feat = python_speech_features.base.delta(input_values, 1)
        d_mfcc_feat2 = python_speech_features.base.delta(input_values, 2)
        audio_driven_obj = np.hstack((input_values, d_mfcc_feat, d_mfcc_feat2))
        frame_start, frame_end = 0, int(audio_driven_obj.shape[0]/4)
        audio_start, audio_end = int(frame_start * 4), int(frame_end * 4) # The video frame is fixed to 25 hz and the audio is fixed to 100 hz
        
        audio_driven = torch.Tensor(audio_driven_obj[audio_start:audio_end,:]).unsqueeze(0).float().to(args.device)
        
    elif conf.infer_type.startswith('hubert'):
        # Hubert features
        if not os.path.exists(args.test_hubert_path):
            
            if not check_package_installed('transformers'):
                print('Please install transformers module first.')
                exit(0)
            hubert_model_path = 'ckpts/chinese-hubert-large'
            if not os.path.exists(hubert_model_path):
                print('Please download the hubert weight into the ckpts path first.')
                exit(0)
            print('You did not extract the audio features in advance, extracting online now, which will increase processing delay')

            start_time = time.time()

            # load hubert model
            from transformers import Wav2Vec2FeatureExtractor, HubertModel
            audio_model = HubertModel.from_pretrained(hubert_model_path).to(args.device)
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_model_path)
            audio_model.feature_extractor._freeze_parameters()
            audio_model.eval()

            # hubert model forward pass
            audio, sr = librosa.load(args.test_audio_path, sr=16000)
            input_values = feature_extractor(audio, sampling_rate=16000, padding=True, do_normalize=True, return_tensors="pt").input_values
            input_values = input_values.to(args.device)
            ws_feats = []
            with torch.no_grad():
                outputs = audio_model(input_values, output_hidden_states=True)
                for i in range(len(outputs.hidden_states)):
                    ws_feats.append(outputs.hidden_states[i].detach().cpu().numpy())
                ws_feat_obj = np.array(ws_feats)
                ws_feat_obj = np.squeeze(ws_feat_obj, 1)
                ws_feat_obj = np.pad(ws_feat_obj, ((0, 0), (0, 1), (0, 0)), 'edge') # align the audio length with video frame
            
            execution_time = time.time() - start_time
            print(f"Extraction Audio Feature: {execution_time:.2f} Seconds")

            audio_driven_obj = ws_feat_obj
        else:
            print(f'Using audio feature from path: {args.test_hubert_path}')
            audio_driven_obj = np.load(args.test_hubert_path)

        frame_start, frame_end = 0, int(audio_driven_obj.shape[1]/2)
        audio_start, audio_end = int(frame_start * 2), int(frame_end * 2) # The video frame is fixed to 25 hz and the audio is fixed to 50 hz
        
        audio_driven = torch.Tensor(audio_driven_obj[:,audio_start:audio_end,:]).unsqueeze(0).float().to(args.device)
    #============================
    
    # Diffusion Noise
    noisyT = th.randn((1,frame_end, args.motion_dim)).to(args.device)
    
    #======Inputs for Attribute Control=========
    if os.path.exists(args.pose_driven_path):
        pose_obj = np.load(args.pose_driven_path)

       
        if len(pose_obj.shape) != 2:
            print('please check your pose information. The shape must be like (T, 3).')
            exit(0)
        if pose_obj.shape[1] != 3:
            print('please check your pose information. The shape must be like (T, 3).')
            exit(0)

        if pose_obj.shape[0] >= frame_end:
            pose_obj = pose_obj[:frame_end,:]
        else:
            padding = np.tile(pose_obj[-1, :], (frame_end - pose_obj.shape[0], 1))
            pose_obj = np.vstack((pose_obj, padding))
            
        pose_signal = torch.Tensor(pose_obj).unsqueeze(0).to(args.device)/ 90 # 90 is for normalization here
    else:
        yaw_signal = torch.zeros(1, frame_end, 1).to(args.device) + args.pose_yaw
        pitch_signal = torch.zeros(1, frame_end, 1).to(args.device) + args.pose_pitch
        roll_signal = torch.zeros(1, frame_end, 1).to(args.device) + args.pose_roll
        pose_signal = torch.cat((yaw_signal, pitch_signal, roll_signal), dim=-1)
    
    pose_signal = torch.clamp(pose_signal, -1, 1)

    face_location_signal = torch.zeros(1, frame_end, 1).to(args.device) + args.face_location
    face_scae_signal = torch.zeros(1, frame_end, 1).to(args.device) + args.face_scale
    #===========================================

    start_time = time.time()

    #======Diffusion Denosing Process=========
    generated_directions = model.render(one_shot_lia_start, one_shot_lia_direction, audio_driven, face_location_signal, face_scae_signal, pose_signal, noisyT, args.step_T, control_flag=args.control_flag)
    #=========================================
    
    execution_time = time.time() - start_time
    print(f"Motion Diffusion Model: {execution_time:.2f} Seconds")

    generated_directions = generated_directions.detach().cpu().numpy()
    
    start_time = time.time()
    #======Rendering images frame-by-frame=========
    for pred_index in tqdm(range(generated_directions.shape[1])):
        ori_img_recon = lia.render(one_shot_lia_start, torch.Tensor(generated_directions[:,pred_index,:]).to(args.device), feats)
        ori_img_recon = ori_img_recon.clamp(-1, 1)
        wav_pred = (ori_img_recon.detach() + 1) / 2
        saved_image(wav_pred, os.path.join(frames_result_saved_path, "%06d.png"%(pred_index)))
    #==============================================
    
    execution_time = time.time() - start_time
    print(f"Renderer Model: {execution_time:.2f} Seconds")

    frames_to_video(frames_result_saved_path, args.test_audio_path, predicted_video_256_path)
    
    shutil.rmtree(frames_result_saved_path)
    
    
    # Enhancer
    # Code is modified from https://github.com/OpenTalker/SadTalker/blob/cd4c0465ae0b54a6f85af57f5c65fec9fe23e7f8/src/utils/face_enhancer.py#L26

    if args.face_sr and check_package_installed('gfpgan'):
        from face_sr.face_enhancer import enhancer_list
        import imageio

        # Super-resolution
        imageio.mimsave(predicted_video_512_path+'.tmp.mp4', enhancer_list(predicted_video_256_path, method='gfpgan', bg_upsampler=None), fps=float(25))
        
        # Merge audio and video
        video_clip = VideoFileClip(predicted_video_512_path+'.tmp.mp4')
        audio_clip = AudioFileClip(predicted_video_256_path)
        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(predicted_video_512_path, codec='libx264', audio_codec='aac')
        
        os.remove(predicted_video_512_path+'.tmp.mp4')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer_type', type=str, default='mfcc_pose_only', help='mfcc_pose_only or mfcc_full_control')
    parser.add_argument('--test_image_path', type=str, default='./test_demos/portraits/monalisa.jpg', help='Path to the portrait')
    parser.add_argument('--test_audio_path', type=str, default='./test_demos/audios/english_female.wav', help='Path to the driven audio')
    parser.add_argument('--test_hubert_path', type=str, default='./test_demos/audios_hubert/english_female.npy', help='Path to the driven audio(hubert type). Not needed for MFCC')
    parser.add_argument('--result_path', type=str, default='./results/', help='Type of inference')
    parser.add_argument('--stage1_checkpoint_path', type=str, default='./ckpts/stage1.ckpt', help='Path to the checkpoint of Stage1')
    parser.add_argument('--stage2_checkpoint_path', type=str, default='./ckpts/pose_only.ckpt', help='Path to the checkpoint of Stage2')
    parser.add_argument('--seed', type=int, default=0, help='seed for generations')
    parser.add_argument('--control_flag', action='store_true', help='Whether to use control signal or not')
    parser.add_argument('--pose_yaw', type=float, default=0.25, help='range from -1 to 1 (-90 ~ 90 angles)')
    parser.add_argument('--pose_pitch', type=float, default=0, help='range from -1 to 1 (-90 ~ 90 angles)')
    parser.add_argument('--pose_roll', type=float, default=0, help='range from -1 to 1 (-90 ~ 90 angles)')
    parser.add_argument('--face_location', type=float, default=0.5, help='range from 0 to 1 (from left to right)')
    parser.add_argument('--pose_driven_path', type=str, default='xxx', help='path to pose numpy, shape is (T, 3). You can check the following code https://github.com/liutaocode/talking_face_preprocessing to extract the yaw, pitch and roll.')
    parser.add_argument('--face_scale', type=float, default=0.5, help='range from 0 to 1 (from small to large)')
    parser.add_argument('--step_T', type=int, default=50, help='Step T for diffusion denoising process')
    parser.add_argument('--image_size', type=int, default=256, help='Size of the image. Do not change.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for computation')
    parser.add_argument('--motion_dim', type=int, default=20, help='Dimension of motion. Do not change.')
    parser.add_argument('--decoder_layers', type=int, default=2, help='Layer number for the conformer.')
    parser.add_argument('--face_sr', action='store_true', help='Face super-resolution (Optional). Please install GFPGAN first')



    args = parser.parse_args()

    # macOS Config
    # Check if MPS is available
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        args.device = torch.device("mps")
        print("MPS backend is available.")

    main(args)