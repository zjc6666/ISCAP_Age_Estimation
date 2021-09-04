#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import json
import logging
import codecs
import numpy as np
import sys
import math

def utt2label(file):
    f = open(file, 'r')
    line = f.readline()
    utt2h = {}
    while line:
        row = line.split()
        utt2h[row[0]] = float(row[1])
        line = f.readline()
    f.close()
    return utt2h

def genMap(file):
    f = open(file, 'r')
    line = f.readline()
    m = {}
    while line:
        row = line.split()
        m[row[0]] = row[1]
        line = f.readline()
    f.close()
    return m

predicted = sys.argv[1] ### Contains features information
reference = sys.argv[2] # utt2age
utt2spkFile = sys.argv[3] # utt2spk
spk2genderFile = sys.argv[4] # spk2gender

utt2spk = genMap(utt2spkFile)
spk2gender = genMap(spk2genderFile)

# you may also want to remove whitespace characters like `\n` at the end of each line
pred_utt2h = utt2label(predicted)
ref_utt2h = utt2label(reference)
mse_m = 0
mae_m = 0
count_m = 0

mse_f = 0
mae_f = 0
count_f = 0

mse_all = 0
mae_all = 0
count_all = 0
for utt in pred_utt2h.keys():
    pred_val = pred_utt2h[utt]
    ref_val = ref_utt2h[utt]
    spk = utt2spk[utt]
    gender = spk2gender[spk]
    if gender.lower() == 'm' :
        mse_m = mse_m +  (pred_val - ref_val) * (pred_val - ref_val)
        mae_m = mae_m +  abs(pred_val - ref_val)
        count_m = count_m + 1
    elif gender.lower() == 'f':
        mse_f = mse_f +  (pred_val - ref_val) * (pred_val - ref_val)
        mae_f = mae_f +  abs(pred_val - ref_val)
        count_f = count_f + 1
    else:
        print('Unexpected gender ' + utt[0])
        sys.exit(1)
    mse_all = mse_all +  (pred_val - ref_val) * (pred_val - ref_val)
    mae_all = mae_all +  abs(pred_val - ref_val)
    count_all = count_all + 1
mse_m = mse_m/count_m
mae_m = mae_m/count_m
rmse_m = math.sqrt(mse_m)

mse_f = mse_f/count_f
mae_f = mae_f/count_f
rmse_f = math.sqrt(mse_f)

mse_all = mse_all/count_all
mae_all = mae_all/count_all
rmse_all = math.sqrt(mse_all)

print('=====Number of male utt = ' + str(count_m))
print('Root Mean Square Error (RMSE) for male = ' + str(rmse_m))
print('Mean Absolute Error (MAE) for male = ' + str(mae_m))

print('=====Number of female utt = ' + str(count_f))
print('Root Mean Square Error (RMSE) for female = ' + str(rmse_f))
print('Mean Absolute Error (MAE) for female = ' + str(mae_f))

print('=====Number of all utt = ' + str(count_all))
print('Root Mean Square Error (RMSE) for all = ' + str(rmse_all))
print('Mean Absolute Error (MAE) for all = ' + str(mae_all))


