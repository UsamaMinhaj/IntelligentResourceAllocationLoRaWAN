#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:05:43 2019

@author: tuyenta
"""
from lora.utils import print_params, sim
import logging

nrNodes = int(1000)
nrIntNodes = int(nrNodes)
nrBS = int(1)
initial = "RANDOM"
radius = float(4500)
distribution = [0.1, 0.1, 0.2, 0.2, 0.2, 0.2]
# distribution = [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6]
avgSendTime = int(10 * 60 * 1000)  # original 4*60*1000
horTime = int(6000)  # original 2e7 (6000 for just 4 readings)
packetLength = int(50)
sfSet = [7, 8, 9, 10, 11, 12]
freqSet = [867100, 868100, 867500]
CRSet = [5, 7]
powSet = [14, 17, 11, 20, 8, 5, 23]

powSet.sort()

captureEffect = True
interSFInterference = True
info_mode = 'NO'

# Mobility
mobility = False
mobile_cal = False and mobility
mobile_nodes = [x for x in range(1, nrNodes + 1, 2)]
velocity = float(50 / 3.6)  # km per hour -> m/s

# learning algorithm
freqAlgo = 'Uniform'  # Random or RL or Uniform
algo = 'exp4'
power_algo = 'DL'
adr = 1  # Type of ADR algorithm

if power_algo == 'DL' or algo == 'ADR':  # If power has to be determined using Deep Learning then powSet has only one
    # choice (14 is just arbitrary selected)
    powSet = [14]

if algo == 'exp4o':
    freqAlgo = 'Uniform'
    CRSet = [5]
    power_algo = 'DL'

if algo == 'ADR':
    freqAlgo = 'Uniform'

# folder
print(powSet)
exp_name = 'test_2'
logdir = 'Sim_2'
learning_rate2 = 0.05

# print simulation parameters
print("\n=================================================")
print_params(nrNodes, nrIntNodes, nrBS, initial, radius, distribution, avgSendTime, horTime, packetLength,
             sfSet, freqSet, powSet, captureEffect, interSFInterference, info_mode, algo)
assert initial in ["UNIFORM", "RANDOM"], "Initial mode must be UNIFORM, or RANDOM."
assert info_mode in ["NO", "PARTIAL", "FULL"], "Initial mo de must be NO, PARTIAL, or FULL."
assert algo in ["exp3", "exp3s", "exp4", "ADR", "exp4o", "Random"], "Learning algorithm must be exp3 or exp3s."
assert power_algo in ["DL", "RL"], "Learning algorithm must be DL or RL."
assert freqAlgo in ['Uniform', 'RL', 'Random'], "Must one of three"

# Logging
logging.basicConfig(filename='app.log', level=logging.INFO)
logging.info('Exp3s:')
logging.info(str((nrNodes, nrIntNodes, nrBS, initial, radius, distribution, avgSendTime,
                  horTime, packetLength, sfSet, freqSet, powSet, CRSet,
                  captureEffect, interSFInterference, info_mode, algo, power_algo, freqAlgo, logdir, exp_name,
                  learning_rate2, adr, mobility, mobile_nodes, velocity, mobile_cal)))

# running simulation
bsDict, nodeDict = sim(nrNodes, nrIntNodes, nrBS, initial, radius, distribution, avgSendTime,
                       horTime, packetLength, sfSet, freqSet, powSet, CRSet,
                       captureEffect, interSFInterference, info_mode, algo, power_algo, freqAlgo, logdir, exp_name,
                       learning_rate2, adr, mobility, mobile_nodes, velocity, mobile_cal)
