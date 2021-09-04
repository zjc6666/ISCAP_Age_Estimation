## ISCAP - Identifying Speaker Characteristics through Audio Profiling - Age Estimation

## Introduction
This project is mainly for age estimation. The model architecture is transformer encoder structure.
While proposing to use such a network structure, two different features are mainly used in the experiment, namely fbank and wav2vec2 feature.
## Installation
### Setting up kaldi environment
```
git clone -b 5.4 https://github.com/kaldi-asr/kaldi.git kaldi
cd kaldi/tools/; make; cd ../src; ./configure; make
```
### Setting up espnet environment
```
git clone -b v.0.9.7 https://github.com/espnet/espnet.git
cd espnet/tools/        # change to tools folder
ln -s {kaldi_root}      # Create link to Kaldi. e.g. ln -s home/theanhtran/kaldi/
```
### Set up Conda environment
```
./setup_anaconda.sh anaconda espnet 3.7.9   # Create a anaconda environmetn - espnet with Python 3.7.9
make TH_VERSION=1.8.0 CUDA_VERSION=10.2     # Install Pytorch and CUDA
. ./activate_python.sh; python3 check_install.py  # Check the installation
conda install torchvision==0.9.0 torchaudio==0.8.0 -c pytorch
```
### Set your own execution environment
Open ISCAP_Age_Estimation/path.sh file, change $MAIN_ROOT$ to your espnet directory, 
```
e.g. MAIN_ROOT=/home/theanhtran/espnet
```
## Prepare data
Because wav2vec2 model training can only be sent to the absolute path of audio files, not in the form of pipes.\
So the first step is to regenerate a new wav file using the original wav.scp and segments files.\
You can find the following three scripts in prepare.sh by directly executing ```bash scripts/prepare_ Data.sh```
### Denerate new wav file for train set
```
## parameter 1: wav.scp file, parameter 2: segments file, parameter 3: Storage address of the new audio file
## Generate execute command
python scripts/generate_new_wav.py data-org/train/wav.scp data-org/train/segments /home3/maison2/zjc/data/NIST_SRE_Corpus/train/ > data-org/train/generate_cmd.sh
bash data-org/train/generate_cmd.sh
mkdir data/train -p
cp data-org/train/{text,utt2spk,utt2age} data/train
cat data-org/train/wav.scp | awk '{print $1 " /home3/maison2/zjc/data/NIST_SRE_Corpus/train/"$1".wav"}' > data/train/wav.scp
```
### Denerate new wav file for valid set
```
## parameter 1: wav.scp file, parameter 2: segments file, parameter 3: Storage address of the new audio file
## Generate execute command
python scripts/generate_new_wav.py data-org/valid/wav.scp data-org/valid/segments /home3/maison2/zjc/data/NIST_SRE_Corpus/valid/ > data-org/valid/generate_cmd.sh
bash data-org/valid/generate_cmd.sh
mkdir data/valid -p
cp data-org/valid/{text,utt2spk,utt2age} data/valid
cat data-org/valid/wav.scp | awk '{print $1 " /home3/maison2/zjc/data/NIST_SRE_Corpus/valid/"$1".wav"}' > data/valid/wav.scp
```
### Denerate new wav file for test set
```
## parameter 1: wav.scp file, parameter 2: segments file, parameter 3: Storage address of the new audio file
## Generate execute command
python scripts/generate_new_wav.py data-org/test/wav.scp data-org/test/segments /home3/maison2/zjc/data/NIST_SRE_Corpus/test/ > data-org/test/generate_cmd.sh
bash data-org/test/generate_cmd.sh
mkdir data/test -p
cp data-org/test/{text,utt2spk,utt2age} data/test
cat data-org/test/wav.scp | awk '{print $1 " /home3/maison2/zjc/data/NIST_SRE_Corpus/test/"$1".wav"}' > data/test/wav.scp
```
### Do speed perturb for train


## How to run Age Estimation systems
Two different features are mainly used in the experiment, namely fbank and wav2vec2 feature.

### The role of each step of the script
```
## step01: Extracting filter-bank features and cmvn
## step02: Generate label file and dump features
## step03: Dictionary Preparation
## step04: Make Json Labels (generate json file)
## step05: Network training (Train our age estimate system)
## step06: Decoding(get estimation age)
```

### Using fank feature to run age estimation system
```
## --steps: You can modify this variable to control the part to be executed
## --nj: Number of jobs for the task
bash run_transformer_age_estimation.sh --steps 1,2,3,4,5,6 --nj 10
```

### Using wav2vec2 feature to run age estimation system
```
bash run_transformer_age_estimation_wav2vec2.sh --steps 1,2,3,4,5,6 --nj 10
```

### Notations
You can change the ```train_set``` variable in the above two scripts(run_transformer_age_estimation.sh, run_transformer_age_estimation_wav2vec2.sh) to select the data you want to use (train or train_vol_sp).

