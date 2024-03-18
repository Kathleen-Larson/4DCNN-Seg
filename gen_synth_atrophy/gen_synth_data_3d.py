#!/usr/bin/env python3
import site,sys
print(site.getsitepackages())
print(sys.path)

import socket, os
import numpy as np
import glob, copy

#from tqdm import tqdm
import tensorflow as tf

import glob, copy
import freesurfer as fs
import neurite as ne
import neurite_sandbox as nes
import voxelmorph as vxm
import voxelmorph_sandbox as vxms
from pathlib import Path

from gen_model import gen_synth_atrophy as gsa


## Log set-up
log = 'gen_synth_atrophy/log.txt'
f = open(log, 'w')
f.write(f'SynthId SampleID\n')
f.close()


## Path set-up
sample_data_dir = '/autofs/cluster/vxmdata1/FS_Slim/proc/cleaned'
subject_paths = [f for f in Path(sample_data_dir).iterdir()]
sample_names = [str(f).split('/')[-1].split('.')[0] for f in subject_paths if f.stem.endswith('MR1')]
n_sample_subjects = len(sample_names)

n_synth_subjects = 1000
synth_dir = '/space/azura/2/users/kl021/Data/Synth_OASIS-2_3d_talairach'
synth_paths = [os.path.join(synth_dir, 'subject%4.4d' % s) \
               for s in range(n_synth_subjects)]
if not os.path.exists(synth_dir):  os.mkdir(synth_dir)


## Sample data set-up
sample_label_paths = [os.path.join(sample_data_dir, s, 'samseg_talairach.mgz') for s in sample_names]
sample_label_vols = [fs.Volume.read(path).data for path in sample_label_paths]
inshape = sample_label_vols[0].shape


## Label set-up
lut = fs.lookups.default()
lesion_label = lut.search('WM-hypointensities')[0]
unknown_label = lut.search('Unknown')[0]
left_wm = lut.search('Left-Cerebral-White-Matter')[0]
right_wm = lut.search('Right-Cerebral-White-Matter')[0]
left_gm = lut.search('Left-Cerebral-Cortex')[0]
right_gm = lut.search('Right-Cerebral-Cortex')[0]
left_hippo = lut.search('Left-Hippocampus')[0]
right_hippo = lut.search('Right-Hippocampus')[0]
left_amy = lut.search('Left-Amygdala')[0]
right_amy = lut.search('Right-Amygdala')[0]
left_caudate = lut.search('Left-Caudate')[0]
right_caudate = lut.search('Right-Caudate')[0]
left_vent = lut.search('Left-Lateral-Ventricle')[0]
right_vent = lut.search('Right-Lateral-Ventricle')[0]
left_inf_lat_vent = lut.search('Left-Inf-Lat-Ven')[0]
right_inf_lat_vent = lut.search('Right-Inf-Lat-Ven')[0]
left_thalamus = lut.search('Left-Thalamus')[0]
right_thalamus = lut.search('Right-Thalamus')[0]
left_accumbens = lut.search('Left-Accumbens')[0]
right_accumbens = lut.search('Right-Accumbens')[0]

perc_change = -0.4
slist = {left_hippo : [perc_change, left_vent, left_inf_lat_vent, unknown_label],
         left_amy : [perc_change, left_vent, left_inf_lat_vent, unknown_label],
         left_caudate : [perc_change, left_vent, lesion_label],
         lesion_label : [10, left_gm, left_wm, left_caudate],
         left_wm : [perc_change, left_vent, left_inf_lat_vent, lesion_label],
         left_thalamus : [perc_change, left_vent],
         left_accumbens  : [perc_change, left_vent],
         left_thalamus : [perc_change, left_vent],
         left_accumbens  : [perc_change, left_vent]
}

left = [1,2,3,4,5,6,7,8,9,10,11,12,13,17,18,19,20,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,73,78,81,83,96]
right = [40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,74,79,82,84,97]

labels_in = np.unique(np.array(sample_label_vols[0], dtype=np.int32))
labels_in_dict = {}
for i in range(len(labels_in)):
    if np.isin(labels_in[i], right):
        j = np.where(right==labels_in[i])[0][0]
        labels_in_dict[labels_in[i]] = left[j]
    else:
        labels_in_dict[labels_in[i]] = labels_in[i]


## Initialize functions
t_max = 5 #baseline, 6mo, 1yr, 18mo, 2yr
label_to_image_args = dict(warp_max=1, blur_max=.5, bias_max=.25,
                           noise_min=0.05, noise_max=0.15,
                           zero_background=1.0, clip_max=2800)
synth_image_seeds = dict(mean=1, warp=2, blur=5, bias=4, gamma=3)

gen_model_init = gsa.GenSynthAtrophy(inshape=inshape,
                                     labels_in=labels_in_dict,
                                     labels_out=None,
                                     synth_image_seeds=synth_image_seeds,
                                     structure_list={},
                                     max_lesions_to_insert=5,
                                     insert_labels=[left_wm],
                                     lesion_label=lesion_label,
                                     subsample_atrophy=1.0,
                                     label_to_image_args=label_to_image_args,
)
gen_model = gsa.GenSynthAtrophy(inshape=inshape,
                                labels_in=labels_in_dict,
                                labels_out=None,
                                synth_image_seeds=synth_image_seeds,
                                structure_list=slist,
                                max_lesions_to_insert=5,
                                insert_labels=[left_wm],
                                lesion_label=lesion_label,
                                subsample_atrophy=1.0,
                                label_to_image_args=label_to_image_args,
)


## Iterate through number of synth subjects
for spath in synth_paths[681:]:
    sname = str(spath).split('/')[-1].split('.')[0]
    print(sname)
    sspath = os.path.join(spath, sname)
    if not os.path.exists(spath):  os.mkdir(spath)
    
    sample_idx = np.random.randint(0, len(sample_names))
    sample_label_vol = sample_label_vols[sample_idx]

    f = open(log, 'a')
    f.write(f'{sname} {sample_names[sample_idx]}\n')
    f.close()

    for t in range(t_max):
        input_label_vol = sample_label_vol if t==0 else output_label
        output_image_path = sspath + '.time' + str(t) + '.intensity.mgz'
        output_label_path = sspath + '.time' + str(t) + '.labels.mgz'
        
        with tf.device('/gpu:0'):
            model = gen_model_init if t==0 else gen_model
            input_label_vol = input_label_vol[np.newaxis, ...]
            pred = gen_model.predict(input_label_vol)
            
        output_image = np.squeeze(pred[0])
        fs.Volume(data=output_image).write(output_image_path)

        output_label = np.squeeze(pred[1])
        output_label_edited = np.multiply(output_label, np.array(output_label > 1, dtype=output_label.dtype))
        fs.Volume(data=output_label_edited).write(output_label_path)

