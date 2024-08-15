
## Windows Installation

Step 1: Clone repository
```
git clone https://github.com/X-LANCE/AniTalker/
```

Step 2: Navigate inside cloned repository

```
cd AniTalker
```

Step 3: Create virtual environment using conda
```
conda create -n anitalker python==3.9.0
```

Step 4: Activate virtual environment
```
conda activate anitalker
```

Step 5: Install dependencies
```
pip install -r requirements_windows.txt
```

Step 6: Download checkpoints
```
git lfs install
```
```
git clone https://huggingface.co/taocode/anitalker_ckpts ckpts
```

Step 7: Download additional files for auto-cropping on source image
```
wget -O code/data_preprocess/shape_predictor_68_face_landmarks.dat https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat

wget -O code/data_preprocess/M003_template.npy https://raw.githubusercontent.com/tanshuai0219/EDTalk/main/data_preprocess/M003_template.npy

```

Step 8: Launch WebUI
```
python code/webgui.py
```