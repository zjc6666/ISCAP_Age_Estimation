#!/bin/bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

steps=1

set -e
# set -u
set -o pipefail
. utils/parse_options.sh || exit 1;

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

data=data-org
tgt=data
train_set="train"
valid_set="valid"
recog_set="test"

if [ ! -z $step01 ]; then
   echo "## Generate new wav file for sre08/10 dataset"
   for x in $train_set $valid_set $recog_set; do
      python scripts/generate_new_wav.py data-org/$x/wav.scp data-org/$x/segments /home3/maison2/zjc/data/NIST_SRE_Corpus/$x/ > data-org/$x/generate_cmd.sh
      bash $data/$x/generate_cmd.sh
      mkdir $tgt/$x -p
      cp $data/$x/{text,utt2spk,utt2age} $tgt/$x
      path="/home3/maison2/zjc/data/NIST_SRE_Corpus/"$x
      cat $data/$x/utt2spk | awk -v p="$path" '{print $1 " "p"/"$1".wav"}' > $tgt/$x/wav.scp
      cat $tgt/$x/wav.scp | cut -d ' ' -f2 > $tgt/$x/wav_path
      utils/fix_data_dir.sh $tgt/$x
      utils/validate_data_dir.sh --no-feats $tgt/$x
   done

   echo "## Generate new wav file for sre08/10 dataset Done"
fi

if [ ! -z $step02 ]; then
   echo "## Do speed perturb for train set"
   utils/perturb_data_dir_speed.sh 1.1 $tgt/$train_set $data/train_1.1
   utils/perturb_data_dir_speed.sh 0.9 $tgt/$train_set $data/train_0.9
   # generate new wav file
   for x in 1.1 0.9; do
      python scripts/generate_new_wav2.py $data/train_$x/wav.scp /home3/maison2/zjc/data/NIST_SRE_Corpus/train_$x/ > $data/train_$x/generate_cmd.sh
      bash $data/train_$x/generate_cmd.sh
      mkdir $tgt/train_$x -p
      cp $data/train_$x/{text,utt2spk} $tgt/train_$x
      path="/home3/maison2/zjc/data/NIST_SRE_Corpus/train_"$x
      cat $data/train_$x/wav.scp | awk -v p="$path" '{print $1 " "p"/"$1".wav"}' > $tgt/train_$x/wav.scp
      cat $data/train/utt2age | awk -v sp="$x" '{print "sp"sp"-"$0}' > $tgt/train_$x/utt2age
      utils/fix_data_dir.sh $tgt/train_$x
      utils/validate_data_dir.sh --no-feats ${tgt}/train_$x
   done

   # form train_vol_sp
   utils/combine_data.sh $tgt/train_vol_sp $tgt/train $tgt/train_1.1 $tgt/train_0.9
   cat $tgt/train/utt2age $tgt/train_1.1/utt2age $tgt/train_0.9/utt2age > $tgt/train_vol_sp/utt2age
   utils/fix_data_dir.sh $tgt/train_vol_sp
   utils/validate_data_dir.sh --no-feats  $tgt/train_vol_sp
   echo "## Do speed perturb for train set Done"
fi
data_spk="data-spk"
if [ ! -z $step03 ]; then
   echo "## Step03: Form a new format file in order to run the xvector system"
   for x in $train_set $valid_set $recog_set train_vol_sp; do
      mkdir ${data_spk}/$x -p
      cat $tgt/$x/utt2age | awk '{print $2"-"$1 " " $2}' > ${data_spk}/$x/utt2spk
      paste ${data_spk}/$x/utt2spk ${tgt}/$x/wav.scp -d ' ' | cut -d ' ' -f1,4- > ${data_spk}/$x/new_wav.scp
      mv ${data_spk}/$x/new_wav.scp ${data_spk}/$x/wav.scp
      utils/fix_data_dir.sh $data_spk/$x
      utils/validate_data_dir.sh --no-feats --no-text  ${data_spk}/$x
   done
fi
