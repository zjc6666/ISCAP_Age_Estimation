#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;


cuda_cmd="slurm.pl --quiet --exclude=node0[3-7]"
decode_cmd="slurm.pl --quiet --exclude=node0[3-5]"
cmd="slurm.pl --quiet --exclude=node0[3-6]"
# general configuration
backend=pytorch
steps=1
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
nj=20
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot
vocab_size=2000
bpemode=bpe
# feature configuration
do_delta=false

train_config=conf/train_Transformer_regression.yaml
lm_config=conf/espnet_lm.yaml
decode_config=conf/espnet_decode.yaml
preprocess_config=conf/espnet_specaug.yaml

# rnnlm related
lm_resume=         # specify a snapshot file to resume LM training
lmtag=0             # tag for managing LMs

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
n_average=5

# exp tag
tag="base" # tag for managing experiments.

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
# set -u
set -o pipefail

. utils/parse_options.sh || exit 1;
. path.sh

steps=$(echo $steps | perl -e '$steps=<STDIN>;  $has_format = 0;
  if($steps =~ m:(\d+)\-$:g){$start = $1; $end = $start + 10; $has_format ++;}
        elsif($steps =~ m:(\d+)\-(\d+):g) { $start = $1; $end = $2; if($start == $end){}
        elsif($start < $end){ $end = $2 +1;}else{die;} $has_format ++; }
      if($has_format > 0){$steps=$start;  for($i=$start+1; $i < $end; $i++){$steps .=":$i"; }}
      print $steps;' 2>/dev/null)  || exit 1

if [ ! -z "$steps" ]; then
#  echo $steps
  for x in $(echo $steps|sed 's/[,:]/ /g'); do
     index=$(printf "%02d" $x);
    # echo $index
     declare step$index=1
  done
fi

data=data
exp=exp
train_set="train_vol_sp"
valid_set="valid"
recog_set="test"
basename=wav2vec2-libri960-model-finetune-48epoch # 48 means use sre08/10 data to train wav2vec2 model 48 epochs
feats=ark_layer9
if [ ! -z $step01 ]; then
   echo "## Step 01: Extracting wav2vec2 features and cmvn"
   # You need to extract wav2vec2 features first, use "ISCAP_Age_Estimation/self-supervised-speech-recognition/"
   compute-cmvn-stats scp:$data/${train_set}/${basename}/${feats}.scp $data/${train_set}/${basename}/${feats}_cmvn.ark
   echo "## Step01 Extracting filter-bank features and cmvn Done"
fi

if [ ! -z $step02 ]; then
   echo "## Step02: Generate label file and dump features for track2:E2E"
   for x in ${train_set} ;do
      dump.sh --cmd "$cmd" --nj $nj  --do_delta false \
         $data/$x/${basename}/${feats}.scp $data/${train_set}/${basename}/${feats}_cmvn.ark $data/$x/${basename}/${feats}_dump/log $data/$x/${basename}/${feats}_dump
  done

   for x in ${valid_set} $recog_set;do  
       dump.sh --cmd "$cmd" --nj $nj  --do_delta false \
          $data/$x/${basename}/${feats}.scp $data/${train_set}/${basename}/${feats}_cmvn.ark $data/$x/${basename}/${feats}_dump_${train_set}/log $data/$x/${basename}/${feats}_dump_${train_set}
   done
   echo "## Step02 Generate label file and dump features for track2:E2E Done"   
fi

bpe_set=$train_set
bpe_model=$data/lang/train/${train_set}_${bpemode}_${vocab_size}
dict=$data/lang_1char/sre0810_all.txt
if [ ! -z $step03 ]; then
    echo "## Stage 03: Dictionary Preparation" 
    mkdir -p $data/lang_1char/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 $data/all/text --trans_type char | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

   [ -d $data/lang/$train_set ] || mkdir -p $data/lang/$train_set || exit;
   echo "<unk> 1" > ${dict}

   awk '{$1=""; print}' $data/$bpe_set/text | sed -r 's#^ ##g' > $data/lang/$train_set/${train_set}_input.txt

   spm_train --input=$data/lang/$train_set/${train_set}_input.txt --vocab_size=${vocab_size} --model_type=${bpemode} --model_prefix=${bpe_model} --input_sentence_size=100000000
   spm_encode --model=${bpe_model}.model --output_format=piece < $data/lang/$train_set/${train_set}_input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}

   echo "## Stage 03: Dictionary Preparation Done"
fi

if [ ! -z $step04 ]; then
    echo "## Stage 04: Make Json Labels Done"
    # make json labels
    data2json.sh --nj $nj --cmd "${cmd}" --feat $data/${train_set}/${basename}/${feats}_dump/feats.scp --trans_type char \
       $data/${train_set} ${dict} > ${data}/${train_set}/${basename}/${train_set}_${bpemode}_${vocab_size}_${feats}.json

    for x in $valid_set $recog_set;do 
       data2json.sh --nj $nj --cmd "${cmd}" --feat $data/${x}/${basename}/${feats}_dump_${train_set}/feats.scp --trans_type char \
          $data/$x ${dict} > ${data}/$x/${basename}/${train_set}_${bpemode}_${vocab_size}_${feats}.json
    done
    echo "## Stage 04: Make Json Labels Done"
fi


if [ ! -z $step05 ]; then
    echo "## Stage 05: Decoding"
    nj=50
    for expname in best-model; do
    expdir=${expname}
    for recog_set in $valid_set $recog_set; do
    recog_model=best-model/model.last5.avg.best
    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}

        feat_recog_dir=$data/$rtask/${basename}
        echo $feat_recog_dir 
        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/${train_set}_${bpemode}_${vocab_size}_${feats}.json
        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog_for_regression.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/${train_set}_${bpemode}_${vocab_size}_${feats}.JOB.json \
            --result-label ${expdir}/${decode_dir}/age_pre.JOB.json \
            --model ${expdir}/results/${recog_model}
        cat ${expdir}/${decode_dir}/age_pre.*.json > ${expdir}/${decode_dir}/prediction.txt

        python estimate_rmse_mae_age.py ${expdir}/${decode_dir}/prediction.txt $data/${rtask}/utt2age $data/all/utt2spk $data/all/spk2gender  > ${expdir}/${decode_dir}/results.txt
        python3 estimate_rmse_mae_age_entireRecording.py ${expdir}/${decode_dir}/prediction.txt $data/${rtask}/utt2age $data/all/utt2spk $data/all/spk2gender  > ${expdir}/${decode_dir}/results_recording.txt

    )
    done
    done
    done
    echo "##  Stage 05: Decoding Finished"
fi  
