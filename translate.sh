#!/usr/bin/bash
set -e

model_root_dir=checkpoints
task=iwslt-de2en
model_dir_tag=001

gpu=0


data_dir=iwslt14.tokenized.de-en
ensemble=5
batch_size=128
beam=8
length_penalty=1.6
src_lang=de
tgt_lang=en
sacrebleu_set=

model_dir=$model_root_dir/$task/$model_dir_tag

checkpoint=checkpoint_best.pt

if [ -n "$ensemble" ]; then
        if [ ! -e "$model_dir/last$ensemble.ensemble.pt" ]; then
                PYTHONPATH=`pwd` python3 average_checkpoints.py --inputs $model_dir --output $model_dir/last$ensemble.ensemble.pt --num-epoch-checkpoints $ensemble
        fi
        checkpoint=last$ensemble.ensemble.pt
fi

output=$model_dir/translation.log

export CUDA_VISIBLE_DEVICES=$gpu

python3 generate.py \
data-bin/$data_dir \
--path $model_dir/$checkpoint \
--gen-subset $who \
--batch-size $batch_size \
--beam $beam \
--lenpen $length_penalty \
--output $model_dir/hypo.txt \
--quiet \
--remove-bpe $use_cpu | tee $output