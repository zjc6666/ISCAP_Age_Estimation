# ISCAP_Age_Estimation

This is a wrapper version of [wav2vec 2.0 framework](https://github.com/pytorch/fairseq/tree/master/examples/wav2vec), which attempts to build an accurate speech recognition models with small amount of transcribed data.
But we use this model to extractor new feature for our training.

## Installation

## Usage

### 1.prepare data 

First of all, take our training data set as an unlabeled data set to train the wav2 vec2 model. \
Of course, if you have other real unlabeled data sets to participate in the training, it is also encouraged.\
create wav_path file, like ```../data/train/wav_path```
You also can use ../prepare_data.sh to prepare and generate new wav file.
 - In wav_path file, it's path must be the path of the real audio file. You cannot use the pipeline command.

### 2. download pretrained wav2vec2 model

Instead of training from scratch, we download and use english wav2vec model for weight initialization. This pratice can be apply to all languages.
```
wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt
```

### 3. use new data to finetune orignal wav2vec2 model

```
python3 pretrain.py --fairseq_path ./fairseq/ --audio_path ../data/train/wav_path --init_model wav2vec_small.pt
```
If you have a large amount of data to train wav2vec2 model, you can also consider not using the downloaded model for initialization.
```
python3 pretrain.py --fairseq_path ./fairseq/ --audio_path ../data/train/wav_path
```
After the training, you can find an ```outputs``` folder in the current directory. You can find the location of your final trained model according to the training time.
```
egs: 
outputs/2021-09-02/00-04-52/checkpoints/checkpoint_best.pt
```
For sre08/10 data, You can train only 50 epochs.

### 4. extract wav2vec2 feature from different wav2vec2 encoder layers

Because the encoder of wav2vec2 has many layers, but the features extracted from the last layer are not the best to train our speaker analysis system. \
Therefore, we can train our system by extracting features from different layers to explore which layer has the best effect.\
Our experiments show that if the wav2vec2 model have 12 encoder layers, the best results can be obtained by extracting features from layer 7 or 9.\
You can use the following commands to extract features:

```
# --wav-path: The wav.scp file which you want to extract the dataset
# --layer 9: extractor feature from layer 9
# --out-dir: Storage path of extracted features
# --model: Trained model path, you can find in your outputs folder
python extract_wav2vec2_fairseq2.py --wav-path ../data/train/wav.scp --out-dir ../data/train/wav2vec2-libri960-model-finetune-48epoch --model wav2vec2_model/libri960_basemodel_sre0810_finetune_48epoch.pt --layer 9
```

## Reference:
Paper: wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations: https://arxiv.org/abs/2006.11477
