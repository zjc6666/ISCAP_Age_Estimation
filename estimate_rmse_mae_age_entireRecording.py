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
import statistics

def getRecording(utt):
    uinf = utt.split("_")
    return "_".join(uinf[:-2])

# return: 句子的中位数， 句子的性别（句子指的是segments之前的句子）
def utt2label(file, utt2spk, spk2gender): 
    f = open(file, 'r')
    line = f.readline()
    reco2predict = {}
    reco2gender = {}
    while line:
        row = line.split()
        recording = getRecording(row[0])
        spk = utt2spk[row[0]]
        gender = spk2gender[spk]
        print('utt = ' + row[0] + ', gender = ' + str(gender) + ', prediction = ' + row[1])
        reco2gender[recording] = gender
        if recording in reco2predict:
            reco2predict[recording].append(float(row[1]))
        else:
            listPred = [float(row[1])]
            reco2predict[recording] = listPred
        line = f.readline()
    f.close()
    reco2finalPred = {}
    count = 1
    for recording in reco2predict:
        listPred = reco2predict[recording]
        #print('Consider recording ' + recording + ', which has prd = ' + str(listPred))
        #count += 1
        #if count > 10:
        #    sys.exit(1)
        finalPred = statistics.median(listPred)
        print('Consider recording ' + recording + ' which has ' + str(len(listPred)) + ' segments. max = ' + str(max(listPred)) + ', min = ' + str(min(listPred)) + ', median = ' + str(finalPred))
        reco2finalPred[recording] = finalPred
    return reco2finalPred, reco2gender

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
reference = sys.argv[2]
utt2spkFile = sys.argv[3]
spk2genderFile = sys.argv[4]

utt2spk = genMap(utt2spkFile)
spk2gender = genMap(spk2genderFile)

# you may also want to remove whitespace characters like `\n` at the end of each line
reco2pred, reco2gender = utt2label(predicted, utt2spk, spk2gender)
print('############ Finish prediction. Now load the reference ##############')
reco2ref, reco2gender = utt2label(reference, utt2spk, spk2gender)
mse_m = 0
mae_m = 0
count_m = 0

mse_f = 0
mae_f = 0
count_f = 0

mse_all = 0
mae_all = 0
count_all = 0
for rec in reco2pred.keys():
    pred_val = reco2pred[rec]
    ref_val = reco2ref[rec]
    gender = reco2gender[rec]
    if gender.lower() == 'm' :
        mse_m = mse_m +  (pred_val - ref_val) * (pred_val - ref_val)
        mae_m = mae_m +  abs(pred_val - ref_val)
        count_m = count_m + 1
    elif gender.lower() == 'f':
        mse_f = mse_f +  (pred_val - ref_val) * (pred_val - ref_val)
        mae_f = mae_f +  abs(pred_val - ref_val)
        count_f = count_f + 1
    else:
        print('Unexpected gender ')
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


