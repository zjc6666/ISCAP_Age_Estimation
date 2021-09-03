#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Helper script to pre-compute embeddings for a wav2letter++ dataset
"""

import argparse
import glob
import os
from shutil import copy
import h5py
import soundfile as sf
import numpy as np
import torch
from torch import nn
import tqdm
import numpy

import kaldiio
# from models.wav2vec.wav2vec2 import Wav2Vec2Model
import fairseq
import logging
from kaldiio import WriteHelper


def read_audio(fname):
    """ Load an audio file and return PCM along with the sample rate """
    wav, sr = sf.read(fname)
    # assert sr == 16e3
    return wav, 16e3

class PretrainedWav2VecModel(nn.Module):
    def __init__(self, fname):
        super().__init__()
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([fname])
        # checkpoint = torch.load(fname)
        # self.args = checkpoint["args"]
        # print(self.args)
        # model = Wav2Vec2Model.build_model(self.args, None)
        # model.load_state_dict(checkpoint["model"])

        model = model[0]
        # print(model)
        model.eval()

        self.model = model

    def forward(self, x):
        padding_mask =None 
        mask = False
        with torch.no_grad():
            # x = self.model.encoder.extractAllLayerFeatures(x, padding_mask=padding_mask)
            x = self.model.exrtactEncoderFeature(x, padding_mask=padding_mask)
            logging.warning("## layer length: " + str(x[0].shape))
        return x

class Prediction():
    """ Lightweight wrapper around a fairspeech embedding model """

    def __init__(self, fname, gpu=0):
        self.gpu = gpu
        self.model = PretrainedWav2VecModel(fname).cuda(gpu)

    def __call__(self, x):
        x = torch.from_numpy(x).float().cuda(self.gpu)
        # print("##### x.size()===>", x.size())
        with torch.no_grad():
            z = self.model(x.unsqueeze(0))
        return z
        # return z.squeeze(0).cpu().numpy()

def ExtraceEmbedding(layer, wav_path, model_path, out_ark_dir, use_feat=False, gpu=0):
    embedding_dict = {}
    model = Prediction(model_path, gpu)
    f = open(wav_path, 'r')
    lines = f.readline()
    feat_scp_path_list = []

    layer = int(layer)
    # dir_name = "ark_layer" + str(layer)
    # path = os.path.join(out_ark_dir, dir_name)
    path = out_ark_dir
    if not os.path.exists(path):
        os.makedirs(path)
    feat_scp_path = "{}.scp".format(os.path.join(path, "ark_layer" + str(layer)))
    feat_ark_path = "{}.ark".format(os.path.join(path, "ark_layer" + str(layer)))
    if os.path.exists(feat_ark_path):
        os.remove(feat_ark_path)
    with WriteHelper('ark,scp:' + feat_ark_path + "," + feat_scp_path) as writer:
        while(lines):
            utt_name = lines.split()[0]
            path = lines.split()[1]
            logging.warning(lines)
            wav, sr = read_audio(path)
            feature = model(wav)
            writer(utt_name, feature[layer-1].squeeze(0).cpu().numpy())
            lines = f.readline()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('-t', '--type', choices = ['ark_npy', 'npy_ark'], required = True, help = "choice a transfer type.")
    parser.add_argument('--wav-path',  default="", type=str, required = True)
    parser.add_argument('--out-dir',  default="", type=str, required = True)
    parser.add_argument('--model',  default="", type=str, required = True)
    parser.add_argument('--layer',  default=3, type=int, required = True)
    args = parser.parse_args()
    # model_path = "/home/maison2/lid/zjc/w2021/wav2vec2/wav2vec/wav2vec2_base_no_finetuning.pt"
    res_dict = ExtraceEmbedding(args.layer, args.wav_path, args.model, args.out_dir, use_feat=False, gpu=0)
