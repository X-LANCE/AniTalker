import torch
from torch import nn
from model.base import BaseModule
from espnet.nets.pytorch_backend.conformer.encoder import Encoder as ConformerEncoder
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, motion_dim, output_dim, num_layers=2, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_size=motion_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.lstm(x)
        return self.fc(x)

class DiffusionPredictor(BaseModule):
    def __init__(self, conf):
        super(DiffusionPredictor, self).__init__()
        
        self.infer_type = conf.infer_type
        
        self.initialize_layers(conf)
        print(f'infer_type: {self.infer_type}')
        
    def create_conformer_encoder(self, attention_dim, num_blocks):
        return ConformerEncoder(
            idim=0, attention_dim=attention_dim, attention_heads=2, linear_units=attention_dim,
            num_blocks=num_blocks, input_layer=None, dropout_rate=0.2, positional_dropout_rate=0.2,
            attention_dropout_rate=0.2, normalize_before=False, concat_after=False,
            positionwise_layer_type="linear", positionwise_conv_kernel_size=3, macaron_style=True,
            pos_enc_layer_type="rel_pos", selfattention_layer_type="rel_selfattn", use_cnn_module=True,
            cnn_module_kernel=13)

    def initialize_layers(self,  conf, mfcc_dim=39, hubert_dim=1024, speech_layers=4, speech_dim=512, decoder_dim=1024, motion_start_dim=512, HAL_layers=25):

        self.conf = conf
        # Speech downsampling
        if self.infer_type.startswith('mfcc'):
            # from 100 hz to 25 hz
            self.down_sample1 = nn.Conv1d(mfcc_dim, 256, kernel_size=3, stride=2, padding=1)
            self.down_sample2 = nn.Conv1d(256, speech_dim, kernel_size=3, stride=2, padding=1)
        elif self.infer_type.startswith('hubert'):
            # from 50 hz to 25 hz
            self.down_sample1 = nn.Conv1d(hubert_dim, speech_dim, kernel_size=3, stride=2, padding=1)
            
            self.weights = nn.Parameter(torch.zeros(HAL_layers))
            self.speech_encoder = self.create_conformer_encoder(speech_dim, speech_layers)
        else:
            print('infer_type not supported')
            
        # Encoders & Deocoders
        self.coarse_decoder = self.create_conformer_encoder(decoder_dim, conf.decoder_layers)

        # LSTM predictors for Variance Adapter
        if self.infer_type != 'hubert_audio_only':
            self.pose_predictor = LSTM(speech_dim, 3)
            self.pose_encoder = LSTM(3, speech_dim)

        if 'full_control' in self.infer_type:
            self.location_predictor = LSTM(speech_dim, 1)
            self.location_encoder = LSTM(1, speech_dim)
            self.face_scale_predictor = LSTM(speech_dim, 1)
            self.face_scale_encoder = LSTM(1, speech_dim)
        
        # Linear transformations
        self.init_code_proj = nn.Sequential(nn.Linear(motion_start_dim, 128))
        self.noisy_encoder = nn.Sequential(nn.Linear(conf.motion_dim, 128))
        self.t_encoder = nn.Sequential(nn.Linear(1, 128))
        self.encoder_direction_code = nn.Linear(conf.motion_dim, 128)
        
        self.out_proj = nn.Linear(decoder_dim, conf.motion_dim)

    
    def forward(self, initial_code, direction_code, seq_input_vector, face_location, face_scale, yaw_pitch_roll, noisy_x, t_emb, control_flag=False):
        
        if self.infer_type.startswith('mfcc'):
            x = self.mfcc_speech_downsample(seq_input_vector)
        elif self.infer_type.startswith('hubert'):
            norm_weights = F.softmax(self.weights, dim=-1)
            weighted_feature = (norm_weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * seq_input_vector).sum(dim=1)
            x = self.down_sample1(weighted_feature.transpose(1,2)).transpose(1,2)
            x, _ = self.speech_encoder(x, masks=None)
        predicted_location, predicted_scale, predicted_pose = face_location, face_scale, yaw_pitch_roll
        if self.infer_type != 'hubert_audio_only':
            print(f'pose controllable. control_flag: {control_flag}')
            x, predicted_location, predicted_scale, predicted_pose = self.adjust_features(x, face_location, face_scale, yaw_pitch_roll, control_flag)
        concatenated_features = self.combine_features(x, initial_code, direction_code, noisy_x, t_emb) # initial_code and direction_code serve as a motion guide extracted from the reference image. This aims to tell the model what the starting motion should be.
        outputs = self.decode_features(concatenated_features)
        return outputs, predicted_location, predicted_scale, predicted_pose

    def mfcc_speech_downsample(self, seq_input_vector):
        x = self.down_sample1(seq_input_vector.transpose(1,2))
        return self.down_sample2(x).transpose(1,2)

    def adjust_features(self, x, face_location, face_scale, yaw_pitch_roll, control_flag):
        predicted_location,  predicted_scale = 0, 0
        if 'full_control' in self.infer_type:
            print(f'full controllable. control_flag: {control_flag}')
            x_residual, predicted_location = self.adjust_location(x, face_location, control_flag)
            x = x + x_residual

            x_residual, predicted_scale = self.adjust_scale(x, face_scale, control_flag)
            x = x + x_residual

        x_residual, predicted_pose= self.adjust_pose(x, yaw_pitch_roll, control_flag)
        x = x + x_residual
        return x, predicted_location, predicted_scale, predicted_pose

    def adjust_location(self, x, face_location, control_flag):
        if control_flag:
            predicted_location = face_location
        else:
            predicted_location = self.location_predictor(x)
        return self.location_encoder(predicted_location), predicted_location

    def adjust_scale(self, x, face_scale, control_flag):
        if control_flag:
            predicted_face_scale = face_scale
        else:
            predicted_face_scale = self.face_scale_predictor(x)
        return self.face_scale_encoder(predicted_face_scale), predicted_face_scale

    def adjust_pose(self, x, yaw_pitch_roll, control_flag):
        if control_flag:
            predicted_pose = yaw_pitch_roll
        else:
            predicted_pose = self.pose_predictor(x)
        return self.pose_encoder(predicted_pose), predicted_pose

    def combine_features(self, x, initial_code, direction_code, noisy_x, t_emb):
        init_code_proj = self.init_code_proj(initial_code).unsqueeze(1).repeat(1, x.size(1), 1)
        noisy_feature = self.noisy_encoder(noisy_x)
        t_emb_feature = self.t_encoder(t_emb.unsqueeze(1).float()).unsqueeze(1).repeat(1, x.size(1), 1)
        direction_code_feature = self.encoder_direction_code(direction_code).unsqueeze(1).repeat(1, x.size(1), 1)
        return torch.cat((x, direction_code_feature, init_code_proj, noisy_feature, t_emb_feature), dim=-1)

    def decode_features(self, concatenated_features):
        outputs, _ = self.coarse_decoder(concatenated_features, masks=None)
        return self.out_proj(outputs)