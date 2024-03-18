In-progress repository for 4D longitudinal semgentation with synthetic atrophy data

Current workflow:

1) Synthetic atrophy data generated w/ gen_synth_atrophy.gen_synth_data_3d.py
   - Uses cleaned OASIS-2 data and samseg label map
   - Atrophy generated w/ a modified version of SynthAtrophyPair from neurite-sandbox
     - Starts w/ unatrophied label map L_r --> generates synthetic intensity image I_s,0 using neurite_sandbox.models.labels_to_image --> then generates new corresponding label L_s,0 from I_s,0
     - Takes L_s,0 --> induces atrophy w/ voxelmorph_sandbox.layers.ResizeLabels --> generates I_s,1 and L_s,1
     - Iterate to get desired number of synthetic timepoints
   - Can also extract 2D slices from the synth atrophy data w/ gen_synth_atrophy/gen_synth_data_2d.py

2) Longitudnal segmentation w/ 2D or 3D images
   - Model:  unet4d.py : uses 4D convolutions for conv layers and for down/upsampling
     - No max pooling (at least for now, because I couldn't do efficient pooling over 4d)
     - 4D conv operations w/ models/conv4d.py or models/.conv4d_transpose.py
   - segment__base_Xd.sh (X=2 or 3) will run the training script (train.py) w/ desired input parameters
     - RUN_TYPE variable (edit w/in script, not as input arg) specifies whether to run locally or as a slurm job on mlsc
     - All params not specified as an input will default to values in options.py
   - train.py sets up training
   - models/segment.py is the wrapper that runs the training/validation/testing loops (similar to how pytorch lightning has a LightningModule class)
     - I could have put all this in train.py but I thought it was cleaner this way



To-do:

1) a lot
2) ideally the atrophy generation would be done on the fly, but that will require pytorch implementations of all the scripts used to synthesize the data