#!/usr/bin/env python3
# encoding: utf-8

import os
import logging
import sys

def generateNewWavFile(wav, path):
  wavFile = open(wav, 'r')

  lines = wavFile.readline()
  while(lines):
    line = lines.split()
    cmd = ' '.join(line[1:-1])
    cmd = cmd + " > " + path + "/" + line[0] + ".wav"
    print(cmd)
    lines = wavFile.readline()
    # os.system("bash -c " + cmd)

wav = sys.argv[1]
path = sys.argv[2]

generateNewWavFile(wav, path)

