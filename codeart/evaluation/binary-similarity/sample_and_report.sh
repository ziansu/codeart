#!/bin/bash -x

function sample_and_report_one(){
    proj_name=$1
    ckpt_name=$2
    pool_size=$3
    python3 sample_and_report.py \
    --source_file output/$proj_name-src_$ckpt_name \
    --target_file output/$proj_name-tgt_$ckpt_name \
    --source_id_file cache/binary_clone_detection/$proj_name-query.id \
    --target_id_file cache/binary_clone_detection/$proj_name-pool.id \
    --pool_size $pool_size |tee report_$ckpt_name-$proj_name-pool$pool_size.txt
}

function sample_and_report_pools(){
    proj_name=$1
    ckpt_name=$2
    sample_and_report_one $proj_name $ckpt_name 32
    sample_and_report_one $proj_name $ckpt_name 50
    sample_and_report_one $proj_name $ckpt_name 100
    sample_and_report_one $proj_name $ckpt_name 200
    sample_and_report_one $proj_name $ckpt_name 300
    sample_and_report_one $proj_name $ckpt_name 500    
}

function sample_and_report_all(){
    ckpt_name=$1
    sample_and_report_pools coreutilsh $ckpt_name
    sample_and_report_pools binutilsh $ckpt_name
    sample_and_report_pools libcurlh $ckpt_name
    sample_and_report_pools libmagickh $ckpt_name
    sample_and_report_pools opensslh $ckpt_name
    sample_and_report_pools libsqlh $ckpt_name
    sample_and_report_pools puttyh $ckpt_name
}


sample_and_report_all ckpt-4k

