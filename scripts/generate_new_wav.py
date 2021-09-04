#!/usr/bin/env python3
# encoding: utf-8
# write by Jicheng Zhang, for read model's log to write files for draw curve

import os
import logging
import sys

def generateNewWavFile(wav, segment, path):
  wavFile = open(wav, 'r')
  segFile = open(segment, 'r')

  wavDict = {}
  lines = wavFile.readline()
  while(lines):
    line = lines.split()
    con = ' '.join(line[1:])
    wavDict[line[0]] = con
    lines = wavFile.readline()

  lines = segFile.readline()
  while(lines):
    line = lines.split()
    newName = line[0]
    oldName = line[1]
    start = float(line[2])
    end = float(line[3])
    cmd = wavDict[oldName] + " sox -t wav - -t wav " + path + "/" + newName + ".wav trim " + str(start) + " " + str(end - start)
    # cmd = wavDict[oldName] + "trim " + str(start) + " " + str(end - start) + "> " + path + "/" + newName + ".wav"
    print(cmd)
    # os.system("bash -c " + cmd)

    lines = segFile.readline()
  


wav = sys.argv[1]
segment  = sys.argv[2]
path = sys.argv[3]

generateNewWavFile(wav, segment, path)

