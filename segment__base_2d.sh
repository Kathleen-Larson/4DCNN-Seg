#!/bin/bash

### SLURM STUFF
#SBATCH --account=lcnrtx
#SBATCH --partition=lcnrtx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=5G
#SBATCH --gpus=1
#SBATCH --time=3-00:00:00
#SBATCH --job-name=4Dseg_test_2d

WDIR=/space/azura/1/users/kl021/Code/4DCNN
#n_jobs=$(ls $WDIR/configs/temporal/temporal_config_*.txt | wc -l)

set -x

run_type=slurm
#run_type=debug

if [ $run_type == "slurm" ] ; then
    n_workers=8
else
    n_workers=8
fi

loss_fn_list=("dice_loss" "cce_loss" "mse_loss" "mean_dice_loss")

n_start=0
n_jobs=${#loss_fn_list[@]}
let n_jobs=$n_jobs-1

X=2 # number of image dims
output_str="testing_2d__full_res"

function call-train(){
    if [ $run_type == "slurm" ] ; then
        JOB_ID=$SLURM_ARRAY_TASK_ID
    else
        JOB_ID=$1
    fi

    # Define non-default inputs
    data_config="data_utils/Synth_OASIS-2_"$X"d.csv"
    data_loader="synth_"$X"d"
    loss_fn=${loss_fn_list[$JOB_ID]}
    lr_start=0.0001
    lr_param=0.95
    lr_scheduler='ConstantLR'
    max_n_epochs=1000
    metrics_train="MeanDice"
    metrics_valid="MeanDice"
    metrics_test="MeanDice"
    network="UNet"$X"D_long"
    n_samples=300

    #load_model_state=last
    output_dir=$WDIR"/data/results/"${output_str}"/"${loss_fn}"__"${lr_scheduler}"__lr_1e-4__crop_224"
    mkdir -p $output_dir

    # Run train
    python3 train__preload_data.py \
            --data_config $data_config \
	    --data_loader $data_loader \
            --max_n_epochs $max_n_epochs \
	    --loss_fn $loss_fn \
	    --lr_start $lr_start \
	    --lr_param $lr_param \
	    --lr_scheduler $lr_scheduler \
            --network $network \
	    --n_samples $n_samples \
	    --n_workers $n_workers \
	    --output_dir $output_dir \
	    #--load_model_state $load_model_state
	    
}



function main(){
    if [ $run_type == 'slurm' ] ; then
        sbatch --array=$n_start-$n_jobs \
	       --output=slurm_outputs/${output_str}_%a.out \
	       --cpus-per-task=$n_workers \
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
