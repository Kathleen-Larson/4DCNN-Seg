#!/bin/bash

#  SLURM STUFF
#SBATCH --account=lcn
#SBATCH --partition=dgx-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --gpus=1
#SBATCH --time=0-150:00:00
#SBATCH --job-name=4DCNN

set -x

#run_type=slurm
run_type=debug

WDIR=/space/azura/1/users/kl021/Code/4DCNN
DATA_DIR=$WDIR/data/results/synth_training

train=$WDIR/train_unet_classifier.py

output_str_base='classifier_testing'

function call-train(){
    if [ $run_type == "slurm" ] ; then
        JOB_ID=$SLURM_ARRAY_TASK_ID
        n_workers=4
    else
        JOB_ID=$1
        n_workers=8
    fi

    # Define inputs
    aug_config='configs/augmentation/aug_config__sigma_21.txt'
    input_data_files='configs/data/data_config__OAS1_samseg_44_talairach.csv'

    batch_size=1
    n_data_splits=3
    start_aug_on=0
    steps_per_epoch=2 #000
    max_n_epochs=1 #200
    n_dataset_samples=10
    
    apply_robust_normalization=0
    crop_patch_size=160
    n_levels=4
    use_residuals=0

    activ_fn='ELU'
    conv_size=3
    pool_size=2
    loss_fns=('mean_dice_loss_yesbackground')
    lr_start=0.0001
    network='UNet3D_long'
    optim='Adam'
    weight_decay=0.000001

    # Output dir
    output_dir=${DATA_DIR}'/'${output_str_base}
    mkdir -p $output_dir

    # Run
    python3 $train \
            --activation_function $activ_fn \
	    --apply_robust_normalization $apply_robust_normalization \
            --aug_config $aug_config \
            --batch_size $batch_size \
            --conv_window_size $conv_size \
            --crop_patch_size $crop_patch_size \
            --input_data_files $input_data_files \
            --loss_fns ${loss_fns[@]} \
            --lr_start $lr_start \
            --max_n_epochs $max_n_epochs \
            --network $network \
            --n_workers $n_workers \
            --n_data_splits $n_data_splits \
	    --n_dataset_samples $n_dataset_samples \
            --optim $optim \
            --output_dir $output_dir \
            --pool_window_size $pool_size \
            --start_aug_on $start_aug_on \
            --steps_per_epoch $steps_per_epoch \
            --weight_decay $weight_decay
}



function main(){
    if [ $run_type == 'slurm' ] ; then
        OUT_DIR=slurm_outputs/${output_str_base}
        mkdir -p ${OUT_DIR}

        sbatch --array=0 --output=${OUT_DIR}/${output_str}.out $0 call-train
        #sbatch --array=$n_start-$n_jobs --output=slurm_outputs/${output_str_base}_%a.out $0 call-train
    else
        call-train
    fi
}



if [[ $1 ]] ; then
    command=$1
    echo $1
    shift
    $command $@
else
    main
fi
