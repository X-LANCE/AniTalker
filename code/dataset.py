import os
import librosa
from PIL import Image
from torchvision import transforms
import python_speech_features
import random
import os
import numpy as np
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
from PIL import Image

class LatentDataLoader(object):

    def __init__(
        self, 
        window_size, 
        frame_jpgs, 
        lmd_feats_prefix, 
        audio_prefix, 
        raw_audio_prefix,
        motion_latents_prefix, 
        pose_prefix, 
        db_name, 
        video_fps=25, 
        audio_hz=50, 
        size=256, 
        mfcc_mode=False,
    ):
        self.window_size = window_size
        self.lmd_feats_prefix = lmd_feats_prefix
        self.audio_prefix = audio_prefix
        self.pose_prefix = pose_prefix
        self.video_fps = video_fps
        self.audio_hz = audio_hz
        self.db_name = db_name
        self.raw_audio_prefix = raw_audio_prefix
        self.mfcc_mode = mfcc_mode
        

        self.transform = torchvision.transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
        )
    
        self.data = []
        for db_name in [ 'VoxCeleb2', 'HDTF' ]:
            db_png_path = os.path.join(frame_jpgs, db_name)
            for clip_name in tqdm(os.listdir(db_png_path)):

                item_dict = dict()
                item_dict['clip_name'] = clip_name
                item_dict['frame_count'] =  len(list(os.listdir(os.path.join(frame_jpgs, db_name, clip_name))))
                item_dict['hubert_path'] = os.path.join(audio_prefix, db_name, clip_name +".npy")
                item_dict['wav_path'] = os.path.join(raw_audio_prefix, db_name, clip_name +".wav")
                
                item_dict['yaw_pitch_roll_path'] = os.path.join(pose_prefix, db_name, 'raw_videos_pose_yaw_pitch_roll', clip_name +".npy")
                if not os.path.exists(item_dict['yaw_pitch_roll_path']):
                    print(f"{db_name}'s {clip_name} miss yaw_pitch_roll_path")
                    continue

                item_dict['yaw_pitch_roll'] = np.load(item_dict['yaw_pitch_roll_path'])
                item_dict['yaw_pitch_roll'] = np.clip(item_dict['yaw_pitch_roll'], -90, 90) / 90.0
                
                if not os.path.exists(item_dict['wav_path']):
                    print(f"{db_name}'s {clip_name} miss wav_path")
                    continue 
                
                if not os.path.exists(item_dict['hubert_path']):
                    print(f"{db_name}'s {clip_name} miss hubert_path")
                    continue 
                
                
                if self.mfcc_mode:
                    wav, sr = librosa.load(item_dict['wav_path'], sr=16000)
                    input_values = python_speech_features.mfcc(signal=wav,samplerate=sr,numcep=13,winlen=0.025,winstep=0.01)
                    d_mfcc_feat = python_speech_features.base.delta(input_values, 1)
                    d_mfcc_feat2 = python_speech_features.base.delta(input_values, 2)
                    input_values = np.hstack((input_values, d_mfcc_feat, d_mfcc_feat2))
                    item_dict['hubert_obj'] = input_values
                else:
                    item_dict['hubert_obj'] = np.load(item_dict['hubert_path'], mmap_mode='r')
                item_dict['lmd_path'] = os.path.join(lmd_feats_prefix, db_name, clip_name +".txt")
                item_dict['lmd_obj_full'] = self.read_landmark_info(item_dict['lmd_path'], upper_face=False)

                motion_start_path = os.path.join(motion_latents_prefix, db_name, 'motions', clip_name +".npy")
                motion_direction_path = os.path.join(motion_latents_prefix, db_name, 'directions',  clip_name +".npy")
                
                if not os.path.exists(motion_start_path):
                    print(f"{db_name}'s {clip_name} miss motion_start_path")
                    continue
                if not os.path.exists(motion_direction_path):
                    print(f"{db_name}'s {clip_name} miss motion_direction_path")
                    continue

                item_dict['motion_start_obj'] = np.load(motion_start_path)
                item_dict['motion_direction_obj'] = np.load(motion_direction_path)                

                if self.mfcc_mode:
                    min_len = min(
                        item_dict['lmd_obj_full'].shape[0],
                        item_dict['yaw_pitch_roll'].shape[0],
                        item_dict['motion_start_obj'].shape[0],
                        item_dict['motion_direction_obj'].shape[0],
                        int(item_dict['hubert_obj'].shape[0]/4),
                        item_dict['frame_count']
                    )
                    item_dict['frame_count'] = min_len
                    item_dict['hubert_obj'] =  item_dict['hubert_obj'][:min_len*4,:]
                else:
                    min_len = min(
                        item_dict['lmd_obj_full'].shape[0],
                        item_dict['yaw_pitch_roll'].shape[0],
                        item_dict['motion_start_obj'].shape[0],
                        item_dict['motion_direction_obj'].shape[0],
                        int(item_dict['hubert_obj'].shape[1]/2),
                        item_dict['frame_count']
                    )
                
                    item_dict['frame_count'] = min_len
                    item_dict['hubert_obj'] =  item_dict['hubert_obj'][:, :min_len*2, :]
                
                if min_len < self.window_size * self.video_fps + 5:
                    continue

        print('Db count:', len(self.data))

    def get_single_image(self, image_path):
        img_source = Image.open(image_path).convert('RGB')
        img_source = self.transform(img_source)
        return img_source

    def get_multiple_ranges(self, lists, multi_ranges):
        # Ensure that multi_ranges is a list of tuples
        if not all(isinstance(item, tuple) and len(item) == 2 for item in multi_ranges):
            raise ValueError("multi_ranges must be a list of (start, end) tuples with exactly two elements each")
        extracted_elements = [lists[start:end] for start, end in multi_ranges]
        flat_list = [item for sublist in extracted_elements for item in sublist]
        return flat_list
    
    
    def read_landmark_info(self, lmd_path, upper_face=True):
        with open(lmd_path, 'r') as file:
            lmd_lines = file.readlines()
        lmd_lines.sort()
        
        total_lmd_obj = []
        for i, line in enumerate(lmd_lines):
            # Split the coordinates and filter out any empty strings
            coords = [c for c in line.strip().split(' ') if c]
            coords = coords[1:] # do not include the file name in the first row
            lmd_obj = []
            if upper_face:
                # Ensure that the coordinates are parsed as integers
                for coord_pair in self.get_multiple_ranges(coords, [(0, 3), (14, 27), (36, 48)]): # 28个
                    x, y = coord_pair.split('_')
                    lmd_obj.append((int(x)/512, int(y)/512))
            else:
                for coord_pair in coords:
                    x, y = coord_pair.split('_')
                    lmd_obj.append((int(x)/512, int(y)/512))
            total_lmd_obj.append(lmd_obj)
        
        return np.array(total_lmd_obj, dtype=np.float32)

    def calculate_face_height(self, landmarks):
        forehead_center = (landmarks[ :, 21, :] + landmarks[:, 22, :]) / 2
        chin_bottom = landmarks[:, 8, :]
        distances = np.linalg.norm(forehead_center - chin_bottom, axis=1, keepdims=True)
        return distances
            
    def __getitem__(self, index):
        
        data_item = self.data[index]
        hubert_obj = data_item['hubert_obj']
        frame_count = data_item['frame_count']
        lmd_obj_full = data_item['lmd_obj_full']
        yaw_pitch_roll = data_item['yaw_pitch_roll']
        motion_start_obj = data_item['motion_start_obj']
        motion_direction_obj = data_item['motion_direction_obj']
 
        frame_end_index = random.randint(self.window_size * self.video_fps + 1, frame_count - 1)
        frame_start_index = frame_end_index - self.window_size * self.video_fps
        frame_hint_index = frame_start_index - 1
        
        audio_start_index = int(frame_start_index * (self.audio_hz / self.video_fps))
        audio_end_index = int(frame_end_index * (self.audio_hz / self.video_fps))
        
        if self.mfcc_mode:
            audio_feats = hubert_obj[audio_start_index:audio_end_index, :]
        else:
            audio_feats = hubert_obj[:, audio_start_index:audio_end_index, :]

        lmd_obj_full = lmd_obj_full[frame_hint_index:frame_end_index, :]
        
        yaw_pitch_roll = yaw_pitch_roll[frame_start_index:frame_end_index, :]

        motion_start = motion_start_obj[frame_hint_index]
        motion_direction_start = motion_direction_obj[frame_hint_index]
        motion_direction = motion_direction_obj[frame_start_index:frame_end_index, :]



        return {
            'motion_start': motion_start,
            'motion_direction': motion_direction,
            'audio_feats': audio_feats,
            'face_location': lmd_obj_full[1:, 30, 0], # '1:' means taking the first frame as the driven frame. '30' is the noise location, '0' means x coordinate
            'face_scale': self.calculate_face_height(lmd_obj_full[1:,:,:]),
            'yaw_pitch_roll': yaw_pitch_roll,
            'motion_direction_start': motion_direction_start,
        }
  
    def __len__(self):
        return len(self.data)
    