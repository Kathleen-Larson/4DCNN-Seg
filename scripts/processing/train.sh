#!/bin/bash

if [ "$#" -lt 1 ] ; then
    echo "Must input config.yaml file name"
    exit
fi

# Parse any specified flags from command line
flags=()
while [[ $# -gt 0 && "$1" != *.yaml && "$1" != *.yml ]]; do
    flags+=("$1")
    shift
done
   
# Parse config file
config=$1
shift
if [ ! -f $config ] ; then
    echo $config "does not exist :("
    exit
fi

configId=$(basename $config | rev | cut -d '.' -f 2- | rev )
echo 'config='$config

# Run
subdir=$(echo "$(dirname "$config")" | sed 's|.*configs/||')
output_dir=data/results/${subdir}/${configId}
echo "output_dir="$output_dir
mkdir -p $output_dir

train=main/synth_ucnet.py

set -x
python3 $train --config $config --output_dir $output_dir --num_workers 1 --use_cuda "${flags[@]}"
