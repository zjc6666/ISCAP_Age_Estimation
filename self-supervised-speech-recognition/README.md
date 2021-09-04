# ISCAP_Age_Estimation

This is a wrapper version of [wav2vec 2.0 framework](https://github.com/pytorch/fairseq/tree/master/examples/wav2vec), which attempts to build an accurate speech recognition models with small amount of transcribed data.
But we use this model to extractor new feature for our training.

## Installation

## Usage

### 1.prepare data 

First of all, take our training data set as an unlabeled data set to train the wav2 vec2 model. \
Of course, if you have other real unlabeled data sets to participate in the training, it is also encouraged.\
create wav_path file, like ```../data/train/wav_path```
 - In wav_path file, it's path must be the path of the real audio file. You cannot use the pipeline command.

### 2. download pretrained wav2vec2 model



### 3. use new data to finetune orignal wav2vec2 model

### 4. extract wav2vec2 feature from different wav2vec2 encoder layers


