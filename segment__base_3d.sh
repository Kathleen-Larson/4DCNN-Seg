#!/bin/bash

### SLURM STUFF
#SBATCH --account=lcn
#SBATCH --partition=rtx8000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=48G
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
#SBATCH --job-name=4Dseg_test_3D

set -x

WDIR=/space/azura/1/users/kl021/Code/4DCNN
X=3 # number of image dims (should be 2 or 3)
#n_jobs=$(ls $WDIR/configs/temporal/temporal_config_*.txt | wc -l)


### Job running / slurm stuff
#xrun_type=slurm
run_type=debug

n_workers=4

n_start=0
n_jobs=1 #$(ls $WDIR/configs/temporal/temporal_config_*.txt | wc -l) # to run as job array
let n_jobs=$n_jobs-1

output_str="testing_"$X"d__full_res"


## Call this to run the training script
train=train.py

function call-train(){
    if [ $run_type == "slurm" ] ; then
        JOB_ID=$SLURM_ARRAY_TASK_ID
	n_workers=1
    else
        n_workers=1
        JOB_ID=$1
    fi

    # Define non-default inputs
    aug_config="data_utils/augmentation_parameters.txt" # contains params for data augmentation
    data_config="data_utils/Synth_OASIS-2_"$X"d.csv" # contains path to data
    data_loader="synth_"$X"d" # calls loader for 2d or 3d synth data
    lr_start=0.0001
    lr_param=0.95
    lr_scheduler='ConstantLR'
    max_n_epochs=3
    metrics_train="MeanDice"
    metrics_valid="MeanDice"
    metrics_test="MeanDice"
    network="UNet"$X"D_long"
    n_samples=100 # number of samples to use from dataset (also number of steps per epoch)
    
    output_dir=$WDIR"/data/results/"$output_str
    mkdir -p $output_dir
    
    #model_state_path=${output_dir}"/model_best" # only use if running model from previous state
    
    # Run train
    python3 $train \
	    --aug_config $aug_config \
            --data_config $data_config \
	    --data_loader $data_loader \
            --max_n_epochs $max_n_epochs \
	    --lr_start $lr_start \
	    --lr_param $lr_param \
	    --lr_scheduler $lr_scheduler \
            --metrics_test $metrics_test \
            --metrics_train $metrics_train \
            --metrics_valid $metrics_valid \
	    --network $network \
	    --n_samples $n_samples \
	    --n_workers $n_workers \
	    --output_dir $output_dir
	    #--model_state_path $model_state_path
}



function main(){
    if [ $run_type == 'slurm' ] ; then
        sbatch --array=0 \
	       --output=slurm_outputs/$output_str.out \
	       $0 call-train
    else
        call-train 0
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
