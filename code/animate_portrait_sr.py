import os
import torch
import sys
from tqdm import tqdm
from glob import glob
import cv2
import numpy as np
from os.path import join
from torch.utils.data import Dataset, DataLoader
import argparse
import argparse
import os
import numpy as np
import librosa
import torch
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, HubertModel
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

# 输入一张照片， 一段音频，出来一个视频
def animate_portrait(pic_path, wav_path,outfile_path):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HubertModel.from_pretrained("chinese-hubert-large").to(device)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("chinese-hubert-large")
    model.feature_extractor._freeze_parameters()
    model.eval()
        
    audio, sr = librosa.load(wav_path, sr=16000)
    input_values = feature_extractor(audio, sampling_rate=16000, padding=True, do_normalize=True, return_tensors="pt").input_values
    input_values = input_values.to(device)
    ws_feats = []
    with torch.no_grad():
        outputs = model(input_values, output_hidden_states=True)
        for i in range(len(outputs.hidden_states)):
            ws_feats.append(outputs.hidden_states[i].detach().cpu().numpy())
        ws_feat_obj = np.array(ws_feats)
        ws_feat_obj = np.squeeze(ws_feat_obj, 1)

        # if args.padding_to_align_audio or True:

        ws_feat_obj = np.pad(ws_feat_obj, ((0, 0), (0, 1), (0, 0)), 'edge')

        # save_audio_path_1 = save_audio_path.replace('audio', 'audio_hubert', 1)
        
        # dir_path = os.path.dirname(save_audio_path_1)
        # os.makedirs(dir_path, exist_ok=True)
        
        # np.save(save_audio_path_1, ws_feat_obj)
        
        # append_string_to_file('output_g2.txt', save_audio_path_1)
    
    ## 得到了音频的特征
    lia = LIA_Model(size=512, motion_dim=20, fusion_type="weighted_sum", enable_lmd=True, toFlow3D=False)
    
    # if args.stage1_checkpoint_path != "":
    lia.load_lightning_model('./ModelData/epoch=00005.ckpt')
    lia.to('cuda:0')
    #============================
    
    # conf = ffhq256_autoenc()
    conf = autoenc_base()
    conf.seed = 0
    conf.decoder_layers = 8
    conf.infer_type = 'hubert_pose_only'
    conf.motion_dim = 20
    if conf.infer_type == 'hubert_pose_only':
        conf.face_location=False
        conf.face_scale=False
        conf.mfcc = False
    elif conf.infer_type == 'hubert_full_control':
        conf.face_location=True
        conf.face_scale=True
        conf.mfcc = False
    else:
        print('Type NOT Found!')
        exit(0)
    #===========================preprocess============================
    temp_dir = './pre_process_temp'
    crop_res_dir = './pre_process_crop'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    file_name = os.path.basename(pic_path)
    dst_path = os.path.join(temp_dir, file_name)
    shutil.copy(pic_path, dst_path)
    temp_cmd = f'python extract_cropped_faces.py   --from_dir_prefix "./pre_process_temp"   --output_dir_prefix "./pre_process_crop"   --expanded_ratio 0.6'
    try:
        os.system(temp_cmd)
    except:
        print('Error in preprocess!')
    
    source_path = os.path.join(crop_res_dir, file_name)
    pic_path = source_path
    # shutil.move(source_path, pic_path)
    shutil.rmtree(temp_dir)
    # shutil.rmtree(crop_res_dir)
    start_time = time.time()
    #===========================inference============================
    img_source = img_preprocessing(pic_path, 512).to('cuda:0')
    one_shot_lia_start, one_shot_lia_direction, feats = lia.get_start_direction_code(img_source, img_source, img_source, img_source)
    
    model = LitModel(conf)
    model = model.load_from_checkpoint('./ModelData/epoch=239-step=23249.ckpt', conf=conf, strict=False)
    model.ema_model.eval()
    model.ema_model.to('cuda:0')
    
    if conf.infer_type.startswith('hubert'):
        # MFCC hubert features
        # Please extract features to test_hubert_path first.
        # audio_driven_obj = np.load(args.test_hubert_path)
        audio_driven_obj = ws_feat_obj
        frame_start, frame_end = 0, int(audio_driven_obj.shape[1]/2)
        audio_start, audio_end = int(frame_start * 2), int(frame_end * 2) # The video frame is fixed to 25 hz and the audio is fixed to 50 hz
        
        audio_driven = torch.Tensor(audio_driven_obj[:,audio_start:audio_end,:]).unsqueeze(0).float().to('cuda:0')
    
    # Diffusion Noise
    noisyT = th.randn((1,frame_end, 20)).to('cuda:0')
    
    #======Inputs for Attribute Control=========
    yaw_signal = torch.zeros(1, frame_end, 1).to('cuda:0') 
    pitch_signal = torch.zeros(1, frame_end, 1).to('cuda:0') 
    roll_signal = torch.zeros(1, frame_end, 1).to('cuda:0')
    pose_signal = torch.cat((yaw_signal, pitch_signal, roll_signal), dim=-1)

    face_location_signal = torch.zeros(1, frame_end, 1).to('cuda:0')
    face_scae_signal = torch.zeros(1, frame_end, 1).to('cuda:0')
    #===========================================
    
    #======Diffusion Denosing Process=========
    generated_directions = model.render(one_shot_lia_start, one_shot_lia_direction, audio_driven, face_location_signal, face_scae_signal, pose_signal, noisyT, 50, control_flag=False)
    #=========================================
  
    generated_directions = generated_directions.detach().cpu().numpy()
    frames_result_saved_path = os.path.join("temp", 'frames')
    os.makedirs(frames_result_saved_path, exist_ok=True)
    #======Rendering images frame-by-frame=========
    for pred_index in tqdm(range(generated_directions.shape[1])):
        ori_img_recon = lia.render(one_shot_lia_start, torch.Tensor(generated_directions[:,pred_index,:]).to('cuda:0'), feats)
        ori_img_recon = ori_img_recon.clamp(-1, 1)
        wav_pred = (ori_img_recon.detach() + 1) / 2
        saved_image(wav_pred, os.path.join(frames_result_saved_path, "%06d.png"%(pred_index)))
    #==============================================
    #predicted_video_path = os.path.join("temp",  f'{os.path.basename(pic_path).split(".")[0]}-{os.path.basename(wav_path).split(".")[0]}.mp4')
    frames_to_video(frames_result_saved_path, wav_path, outfile_path)
    
    shutil.rmtree(frames_result_saved_path)
    shutil.rmtree(crop_res_dir)
    print(f'Time Cost: {time.time() - start_time}')
    #===================SR=========================
    from face_sr.face_enhancer import enhancer_list
    import imageio
    imageio.mimsave(outfile_path+'.tmp.mp4', enhancer_list(outfile_path, method='gfpgan', bg_upsampler=None), fps=float(25))
    video_clip = VideoFileClip(outfile_path+'.tmp.mp4')
    audio_clip = AudioFileClip(outfile_path)
    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile(outfile_path, codec='libx264', audio_codec='aac')
    print(f'after SR Time Cost: {time.time() - start_time}')
    os.remove(outfile_path+'.tmp.mp4')
    # return predicted_video_path


if __name__ == '__main__':
    animate_portrait("./test_demos/portraits/teeth.png", "./test_demos/audios/trump_audio.mp3","./result0712/trump.mp4")