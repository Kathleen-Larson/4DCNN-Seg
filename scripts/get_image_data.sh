#!/bin/bash

DATA_DIR='/autofs/cluster/vxmdata1/FS_Slim/proc/cleaned'
DATASET='OASIS_OAS1'
IMG_LIST=$(ls ${DATA_DIR}/${DATASET}'_'*'_MR1/samseg_talairach.mgz' )

SYM_DIR='data/input/OAS1_samseg_44_talairach'

## Copy data
if false ; then
    for fname in ${IMG_LIST[@]} ; do
	subject_id=$(echo $fname | rev | cut -d '/' -f 2 | rev | cut -d '_' -f 1-3)
	sym_fname=${SYM_DIR}'/'${subject_id}'.mgz'
	ln -s $fname $sym_fname
    done
fi


## Make data config
if true ; then
    csv_file='configs/data/data_config__OAS1_samseg_44_talairach.csv'
    echo -n > $csv_file
    
    for fname in $(ls $SYM_DIR) ; do
	echo ${SYM_DIR}'/'${fname} >> $csv_file
    done
fi
    



