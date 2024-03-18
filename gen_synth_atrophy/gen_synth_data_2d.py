import os
import freesurfer as fs
import numpy as np
from pathlib import Path


data_3d_dir = '/space/azura/2/users/kl021/Data/Synth_OASIS-2_3d_talairach'
data_2d_dir = '/space/azura/2/users/kl021/Data/Synth_OASIS-2_2d_talairach'
if not os.path.exists(data_2d_dir):  os.mkdir(data_2d_dir)

subject_names = sorted([str(f).split('/')[-1] for f in Path(data_3d_dir).iterdir()])
data_3d_paths = [os.path.join(data_3d_dir, sid) for sid in subject_names]
data_2d_paths = [os.path.join(data_2d_dir, sid) for sid in subject_names]


for sid in subject_names:
    print(sid)
    subject_3d_dir = os.path.join(data_3d_dir, sid)
    subject_2d_dir = os.path.join(data_2d_dir, sid)
    if not os.path.exists(subject_2d_dir):  os.mkdir(subject_2d_dir)

    vol_names = os.listdir(subject_3d_dir)
    for vid in vol_names:
        vol_3d = fs.Volume.read(os.path.join(subject_3d_dir, vid)).data
        vol_2d = np.squeeze(vol_3d[:, :, 112])
        fs.Image(data=vol_2d).write(os.path.join(subject_2d_dir, vid))
