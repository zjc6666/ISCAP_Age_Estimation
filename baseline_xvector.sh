#!/bin/bash

# Build a kaldi x-vector baseline for age estimation
# Modify by Zhang Jicheng(extract x-vector) July 6, 2021

nj=10
steps=100
train_stage=-11
use_gpu=wait
remove_egs=false
cmd="slurm.pl --quiet --exclude=node0[3-6]"
echo 
echo "$0 $@"
echo

. path.sh

. parse_options.sh || exit 1

steps=$(echo $steps | perl -e '$steps=<STDIN>;  $has_format = 0;
  if($steps =~ m:(\d+)\-$:g){$start = $1; $end = $start + 10; $has_format ++;}
        elsif($steps =~ m:(\d+)\-(\d+):g) { $start = $1; $end = $2; if($start == $end){}
        elsif($start < $end){ $end = $2 +1;}else{die;} $has_format ++; }
      if($has_format > 0){$steps=$start;  for($i=$start+1; $i < $end; $i++){$steps .=":$i"; }}
      print $steps;' 2>/dev/null)  || exit 1

if [ ! -z "$steps" ]; then
  for x in $(echo $steps|sed 's/[,:]/ /g'); do
  index=$(printf "%02d" $x);
  declare step$index=1
  done
fi

# data location and project location
source_data=data-spk
project_dir=exp-xvector
train_set=train_vol_sp # train
test_set=test
dev_set=valid
exp_dir=$project_dir/exp
target_data=$project_dir/data
[ -d $exp_dir ] || mkdir -p $exp_dir
[ -d $target_data ] || mkdir -p $target_data

if [ ! -z $step01 ]; then
    echo "## Step01: Extract Feature"
    for i in $train_set $dev_set $test_set; do
        data=$source_data/$i/mfcc-pitch
        log=$data/log
        feat=$data/data 
        utils/data/copy_data_dir.sh $source_data/$i $data || exit 1;
        steps/make_mfcc_pitch.sh --pitch-config conf/pitch.conf --mfcc-config conf/sre-mfcc-20.conf \
                                --nj $nj --cmd "$cmd" $data $log $feat || exit 1;
        steps/compute_cmvn_stats.sh $data $log $feat || exit 1
        sid/compute_vad_decision.sh --nj $nj --cmd "$cmd" \
                                --vad-config conf/vad-5.0.conf $data $log $feat || exit 1
    done
    echo "## Step01: Extract Feature Done"
fi


## Kaldi x-vector model training
# Training (preprocess -> get_egs -> training -> extract_xvectors)
outputname=kaldi_xvector
nnet_dir=$exp_dir/${outputname}
[ -d $nnet_dir ] || mkdir -p $nnet_dir
sleep_time=3
model_limit=8
echo -e "SleepTime=$sleep_time\nLimit=$model_limit" > $nnet_dir/control.conf

egs_dir=$exp_dir/${outputname}/egs
train=$source_data/${train_set}

no_sil=$target_data/${train_set}_no_sil
# Now we prepare the features to generate examples for xvector training.
if [ ! -z $step02 ]; then
  [ -d "${no_sil}" ] && rm -rf ${no_sil}
  [ -d "${exp_dir}/train_no_sil" ] && rm -rf ${exp_dir}/train_no_sil
  subtools/kaldi/sid/nnet3/xvector/prepare_feats_for_egs.sh --nj $nj --cmd "$cmd" \
    ${train}/mfcc-pitch ${no_sil} $exp_dir/${train_set}_no_sil
  echo "${no_sil}"
  utils/fix_data_dir.sh ${no_sil}
  echo "## Step02: Prepare feats for egs Done"
fi
num_pdfs=
min_chunk=60
max_chunk=80

if [ ! -z $step03 ]; then
    subtools/removeUtt.sh ${no_sil} $max_chunk
    echo "## Steps03: Remove utts Done"
fi

if [ ! -z $step04 ]; then

    mkdir -p $nnet_dir
    num_pdfs=$(awk '{print $2}' $train/utt2spk | sort | uniq -c | wc -l)

    subtools/kaldi/sid/nnet3/xvector/get_egs.sh --cmd "$cmd" \
    --nj 20 \
    --stage 0 \
    --frames-per-iter 5500000000 \
    --frames-per-iter-diagnostic 100000 \
    --min-frames-per-chunk $min_chunk \
    --max-frames-per-chunk $max_chunk \
    --num-diagnostic-archives 3 \
    --num-repeats 1000 \
    "${no_sil}" $egs_dir
    echo "## Step04: Get egs Done"
fi

if [ ! -z $step05 ]; then
    num_targets=$(wc -w $egs_dir/pdf2num | awk '{print $1}')
    feat_dim=$(cat $egs_dir/info/feat_dim)

    max_chunk_size=10000
    min_chunk_size=25

    mkdir -p $nnet_dir/configs
	cat <<EOF > $nnet_dir/configs/network.xconfig
	# please note that it is important to have input layer with the name=input
	# The frame-level layers
	input dim=${feat_dim} name=input
 	spec-augment-layer name=spec-augment freq-max-proportion=0.3 time-zeroed-proportion=0.1 time-mask-max-frames=20 include-in-init=true
	relu-batchnorm-layer name=tdnn1 input=Append(-2,-1,0,1,2) dim=512
	relu-batchnorm-layer name=tdnn2 dim=512
	relu-batchnorm-layer name=tdnn3 input=Append(-2,0,2) dim=512
	relu-batchnorm-layer name=tdnn4 dim=512
	relu-batchnorm-layer name=tdnn5 input=Append(-3,0,3) dim=512
	relu-batchnorm-layer name=tdnn6 dim=512
	relu-batchnorm-layer name=tdnn7 input=Append(-4,0,4) dim=512
	relu-batchnorm-layer name=tdnn8 dim=512
	relu-batchnorm-layer name=tdnn9 dim=512
	relu-batchnorm-layer name=tdnn10 dim=1500

	stats-layer name=stats config=mean+stddev(0:1:1:${max_chunk_size})

	# This is where we usually extract the embedding (aka xvector) from.
	relu-batchnorm-layer name=embedding1 dim=512 input=stats

	# This is where another layer the embedding could be extracted
	# from, but usually the previous one works better.
	relu-batchnorm-layer name=embedding2 dim=512
	output-layer name=output include-log-softmax=true dim=${num_targets}
EOF

	steps/nnet3/xconfig_to_configs.py \
          --xconfig-file $nnet_dir/configs/network.xconfig \
          --config-dir $nnet_dir/configs
	cp $nnet_dir/configs/final.config $nnet_dir/nnet.config

	# These three files will be used by sid/nnet3/xvector/extract_xvectors.sh
	echo "output-node name=output input=embedding1.affine" > $nnet_dir/extract.config
	echo "$max_chunk_size" > $nnet_dir/max_chunk_size
	echo "$min_chunk_size" > $nnet_dir/min_chunk_size
fi
if [ ! -z $step06 ]; then
    echo "## Step06: Train X-Vector Network"
    dropout_schedule='0,0@0.20,0.1@0.50,0'
    srand=123
	
    subtools/kaldi/steps/nnet3/train_raw_dnn.py --stage=$train_stage \
      --cmd="$cmd" \
      --trainer.optimization.proportional-shrink 10 \
      --trainer.optimization.momentum=0.5 \
      --trainer.optimization.num-jobs-initial=2 \
      --trainer.optimization.num-jobs-final=4 \
      --trainer.optimization.initial-effective-lrate=0.005 \
      --trainer.optimization.final-effective-lrate=0.0005 \
      --trainer.optimization.minibatch-size=128 \
      --trainer.srand=$srand \
      --trainer.max-param-change=2 \
      --trainer.num-epochs=10 \
      --trainer.dropout-schedule="$dropout_schedule" \
      --trainer.shuffle-buffer-size=1000 \
      --egs.frames-per-eg=1 \
      --egs.dir="$egs_dir" \
      --cleanup.remove-egs $remove_egs \
      --cleanup.preserve-model-interval=10 \
      --use-gpu=wait \
      --dir=$nnet_dir  || exit 1;
    echo "## Step06: Train X-Vector Network Done"
fi

if [ ! -z $step07 ]; then
    echo "## Step07: Extract xvectors of several datasets"
    nj=4
    gpu=true
    cache=3000

    for x in ${train_set} $test_set $dev_set;do 
        subtools/kaldi/sid/nnet3/xvector/extract_xvectors_sre.sh  --cmd "$cmd" \
	    --use-gpu $gpu --nj $nj --cache-capacity $cache $nnet_dir ${source_data}/$x/mfcc-pitch $nnet_dir/$x
        echo "## layer embeddings of $x extracted done."
    done

    echo "## Step07: Extract xvectors of several datasets Done"
fi

# 使用train_vol_sp去训练分类器 
train=("train_vol_sp train_vol_sp") # ("train train")
enroll=("train_vol_sp train_vol_sp") # ("train train")
test=("valid test" )
num=${#enroll[@]}
clad=100
# Get score
if [ ! -z $step08 ]; then
    echo $num
    for ((i=0;i<$num;i++));do
        for meth in lr ;do
            ./scripts/classifier_score.sh --nj $nj --steps 1-11 --eval false --source_data $source_data \
                --trainset ${train[i]} --vectordir $nnet_dir --enrollset ${enroll[i]} --testset ${test[i]} \
                --lda true --clda $clad --submean true --score $meth --metric "eer"
            ./scripts/classifier_score.sh --nj $nj --steps 1-11 --eval false --source_data $source_data --trainset ${train[i]} \
                --vectordir $nnet_dir --enrollset ${enroll[i]} --testset ${test[i]} --lda true --clda $clad --submean true --score $meth --metric "Cavg" || exit;
            echo "#LOG:: getScore use $meth Done!"
        done
    done
    echo "## Step08::getScore use lr Done!"
fi

# print Score
if [ ! -z $step09 ]; then
    echo "##############      Print Score   ######################"
    for ((i=0;i<$num;i++));do
        for meth in lr ;do
            echo ${test[i]} " && " $meth " && clda=" $clad
            if [ -f "$nnet_dir/${test[i]}/score/${meth}_${enroll[i]}_${test[i]}_lda${clad}_submean_norm.Cavg" ];then
                cat $nnet_dir/${test[i]}/score/${meth}_${enroll[i]}_${test[i]}_lda${clad}_submean_norm.Cavg
            fi
            if [ -f "$nnet_dir/${test[i]}/score/${meth}_${enroll[i]}_${test[i]}_lda${clad}_submean_norm.eer" ];then
                cat $nnet_dir/${test[i]}/score/${meth}_${enroll[i]}_${test[i]}_lda${clad}_submean_norm.eer
            fi

            if [ -f "$nnet_dir/${test[i]}/score/${meth}_${enroll[i]}_${test[i]}_lda${clad}_submean_norm.score" ];then
                python3 scripts/calcute_accuracy.py --p 1 --score $nnet_dir/${test[i]}/score/${meth}_${enroll[i]}_${test[i]}_lda${clad}_submean_norm.score
                python3 scripts/calcute_accuracy.py --p 0 --score $nnet_dir/${test[i]}/score/${meth}_${enroll[i]}_${test[i]}_lda${clad}_submean_norm.score > $nnet_dir/${test[i]}/score/${meth}_${enroll[i]}_${test[i]}_lda${clad}_submean_norm.acc

                python3 scripts/get_predict.py $nnet_dir/${test[i]}/score/${meth}_${enroll[i]}_${test[i]}_lda${clad}_submean_norm.score $nnet_dir/${test[i]}/score/${meth}_${enroll[i]}_${test[i]}_lda${clad}_submean_norm_predict.txt
            fi
        done
    done
fi
if [ ! -z $step10 ]; then
    echo "## Statistic MAE ans RMSE"
    for ((i=0;i<$num;i++));do
        python3 estimate_rmse_mae_age.py $nnet_dir/${test[i]}/score/${meth}_${enroll[i]}_${test[i]}_lda${clad}_submean_norm_predict.txt $data/${test[i]}/utt2age $data/all/utt2spk $data/all/spk2gender  > $nnet_dir/${test[i]}/score/results.txt
        python3 estimate_rmse_mae_age_entireRecording.py $nnet_dir/${test[i]}/score/${meth}_${enroll[i]}_${test[i]}_lda${clad}_submean_norm_predict.txt $data/${rtask}/utt2age $data/all/utt2spk $data/all/spk2gender  > $nnet_dir/${test[i]}/score/results_recording.txt
    done
    echo "MAE for ${test[i]} set"
    tail $nnet_dir/${test[i]}/score/results.txt
    echo "RMAE for ${test[i]} set"
    tail $nnet_dir/${test[i]}/score/results_recording.txt
    echo "## Step10: Statistic MAE and RMSE Done"
fi


