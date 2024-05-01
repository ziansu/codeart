#!/bin/bash -x

function encode(){
    model="--model_name_or_path $1"
    python3 inference.py $model \
    --masking_enable_global_memory_patterns true \
    --masking_enable_bridge_patterns false \
    --masking_enable_graph_patterns true \
    --masking_enable_local_patterns true \
    --with_transitive_closure true \
    --position_embedding_type mixed \
    --max_relative_position_embeddings 8 \
    --normalize_embed true \
    --batch_size 48 \
    --source_file cache/binary_clone_detection/$3-query.jsonl \
    --target_file cache/binary_clone_detection/$3-pool.jsonl \
    --source_embed_save_file output/$3-src_$2.npy \
    --target_embed_save_file output/$3-tgt_$2.npy \
    --zero_shot false \
    --top_k 1
}


function encode_benchmarks(){
    encode $1 $2 coreutilsh
    encode $1 $2 binutilsh
    encode $1 $2 libcurlh
    encode $1 $2 libmagickh
    encode $1 $2 opensslh
    encode $1 $2 libsqlh
    encode $1 $2 puttyh
}


mkdir -p output

encode_benchmarks ../save/codeart-binsim/checkpoint-4000 ckpt-4k
