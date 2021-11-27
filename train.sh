#! /usr/bin/bash
set -e
device=0

task=iwslt-de2en
tag=001

arch=fixup_dense_relative_transformer_t2t_iwslt_de_en
share_embedding=1
share_decoder_input_output_embed=0
criterion=label_smoothed_cross_entropy
fp16=1
lr=0.0015
warmup=8000
max_tokens=2048
update_freq=2
weight_decay=0.0001
keep_last_epochs=30
max_epoch=60
data_dir=iwslt14.tokenized.de-en
src_lang=de
tgt_lang=en


save_dir=checkpoints/$task/$tag

if [ ! -d $save_dir ]; then
        mkdir -p $save_dir
fi


gpu_num=`echo "$device" | awk '{split($0,arr,",");print length(arr)}'`

cmd="python3 -u train.py data-bin/$data_dir
  --distributed-world-size $gpu_num -s $src_lang -t $tgt_lang
  --arch $arch
  --optimizer adam --clip-norm 0.0
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $warmup
  --lr $lr --min-lr 1e-09
  --weight-decay $weight_decay
  --criterion $criterion --label-smoothing 0.1
  --max-tokens $max_tokens
  --update-freq $update_freq
  --no-progress-bar
  --log-interval 100
  --ddp-backend no_c10d 
  --save-dir $save_dir
  --keep-last-epochs $keep_last_epochs
  --tensorboard-logdir $save_dir
  --encoder-layer 50
  --decoder-layer 6
  --adam-betas '(0.9, 0.98)'
  --share-all-embeddings
  --fp16"

export CUDA_VISIBLE_DEVICES=$device
cmd="nohup "${cmd}" > $save_dir/train.log 2>&1 &"
eval $cmd
tail -f $save_dir/train.log
