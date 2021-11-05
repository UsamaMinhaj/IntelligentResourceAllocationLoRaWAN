""" LPWAN Simulator: Hepper functions
============================================
Utilities (:mod:`lora.utils`)
============================================
.. autosummary::
   :toctree: generated/
   print_params             -- Print the arguments of the simulation.
   sim                      -- Run the simulation
"""
import os
import numpy as np
from os.path import join, exists
from os import makedirs
import simpy
from .node import myNode
from .bs import myBS
from .bsFunctions import transmitPacket, cuckooClock, saveProb, saveRatio, saveEnergy, saveTraffic
from .loratools import dBmTomW, getMaxTransmitDistance, placeRandomlyInRange, placeRandomly
from .plotting import plotLocations


def print_params(nrNodes, nrIntNodes, nrBS, initial, radius, distribution, avgSendTime, horTime, packetLength, sfSet,
                 freqSet, powSet, captureEffect, interSFInterference, info_mode, algo):
    # print parametters
    print("Simulation Parameters:")
    print("\t Nodes:", nrNodes)
    print("\t Intelligent Nodes:", nrIntNodes)
    print("\t BS:", nrBS)
    print("\t Initial mode:", initial)
    print("\t Radius:", radius)
    print("\t Region Distribution:", distribution)
    print("\t AvgSendTime (exp. distributed):", avgSendTime)
    print("\t Horizon Time:", horTime)
    print("\t Packet lengths:", packetLength)
    print("\t SF set:", sfSet)
    print("\t Channels: ", freqSet)
    print("\t Power set[dB]:", powSet)
    print("\t Capture Effect:", captureEffect)
    print("\t Inter-SF Interference:", interSFInterference)
    print("\t Information mode:", info_mode)
    print("\t Learning algorithm:", algo)


def sim(nrNodes, nrIntNodes, nrBS, initial, radius, distribution, avgSendTime, horTime, packetLength, sfSet, freqSet,
        powSet, CRSet, captureEffect, interSFInterference, info_mode, algo, power_algo, freqAlgo, logdir, exp_name,
        learning_rate2, adr, mobility, mobile_nodes, velocity, mobile_cal):
    simtime = horTime * avgSendTime  # simulation time in ms

    grid = [int(15000), int(15000)]  # maximum simulation area in m

    # Node parameters
    ackLength = 8  # length of the ack
    maxPtx = max(powSet)

    # traffic
    lambda_i = (1 / avgSendTime)  # packet generation rate
    lambda_e = ((nrNodes - nrIntNodes) / nrNodes) * lambda_i * np.random.rand(len(sfSet), len(freqSet))

    # phy parameters (rdd, packetLength, preambleLength, syncLength, headerEnable, crc)
    phyParams = (1, packetLength, 8, 4.25, False, True)

    # Environement parameters (Log-shadowing model)
    # logDistParams = (2.32, 128.95, 1000.0)  # (gamma, Lpld0, d0)
    logDistParams = (3, 107.41, 40.0)
    interferenceThreshold = -150  # dBm

    # BS parameters
    nDemodulator = 8  # number of demodulators on base-station

    # sensitivity
    sf7 = np.array([7, -123.0, -121.5, -118.5])
    sf8 = np.array([8, -126.0, -124.0, -121.0])
    sf9 = np.array([9, -129.5, -126.5, -123.5])
    sf10 = np.array([10, -132.0, -129.0, -126.0])
    sf11 = np.array([11, -134.5, -131.5, -128.5])
    sf12 = np.array([12, -137.0, -134.0, -131.0])
    sensi = np.array([sf7, sf8, sf9, sf10, sf11, sf12])  # array of sensitivity values

    # The spreading factor interaction matrix derived from lab tests and Semtech documentation
    # SFs are not perfectly orthogonal: a signal at SF_m faces interferences from all signals on other SFs
    if captureEffect == True:
        captureThreshold = dBmTomW(6)  # threshold is 6 dB
        if interSFInterference == True:
            interactionMatrix = np.array([
                [dBmTomW(6), dBmTomW(-7.5), dBmTomW(-7.5), dBmTomW(-7.5), dBmTomW(-7.5), dBmTomW(-7.5)],
                [dBmTomW(-9), dBmTomW(6), dBmTomW(-9), dBmTomW(-9), dBmTomW(-9), dBmTomW(-9)],
                [dBmTomW(-13.5), dBmTomW(-13.5), dBmTomW(6), dBmTomW(-13.5), dBmTomW(-13.5), dBmTomW(-13.5)],
                [dBmTomW(-15), dBmTomW(-15), dBmTomW(-15), dBmTomW(6), dBmTomW(-15), dBmTomW(-15)],
                [dBmTomW(-18), dBmTomW(-18), dBmTomW(-18), dBmTomW(-18), dBmTomW(6), dBmTomW(-18)],
                [dBmTomW(-22.5), dBmTomW(-22.5), dBmTomW(-22.5), dBmTomW(-22.5), dBmTomW(-22.5), dBmTomW(6)]])
        else:
            interactionMatrix = np.array([
                [dBmTomW(6), 0, 0, 0, 0, 0],
                [0, dBmTomW(6), 0, 0, 0, 0],
                [0, 0, dBmTomW(6), 0, 0, 0],
                [0, 0, 0, dBmTomW(6), 0, 0],
                [0, 0, 0, 0, dBmTomW(6), 0],
                [0, 0, 0, 0, 0, dBmTomW(6)]])
    else:
        captureThreshold = 0  # threshold is 0 dB
        if interSFInterference == True:
            interactionMatrix = np.array([
                [0, dBmTomW(-7.5), dBmTomW(-7.5), dBmTomW(-7.5), dBmTomW(-7.5), dBmTomW(-7.5)],
                [dBmTomW(-9), 0, dBmTomW(-9), dBmTomW(-9), dBmTomW(-9), dBmTomW(-9)],
                [dBmTomW(-13.5), dBmTomW(-13.5), 0, dBmTomW(-13.5), dBmTomW(-13.5), dBmTomW(-13.5)],
                [dBmTomW(-15), dBmTomW(-15), dBmTomW(-15), 0, dBmTomW(-15), dBmTomW(-15)],
                [dBmTomW(-18), dBmTomW(-18), dBmTomW(-18), dBmTomW(-18), 0, dBmTomW(-18)],
                [dBmTomW(-22.5), dBmTomW(-22.5), dBmTomW(-22.5), dBmTomW(-22.5), dBmTomW(-22.5), 0]])
        else:
            interactionMatrix = np.eye(6, dtype=int)

    # Location generator
    print("======= Generate/Load simulation scenario =======")
    distMatrix, bestDist, bestSF, bestBW = getMaxTransmitDistance(sensi, maxPtx, logDistParams, phyParams)
    print("Max range = {} at SF = {}, BW = {}".format(bestDist, bestSF, bestBW))
    distMatrix[0] = radius / len(distMatrix)
    for i in range(len(distMatrix) - 1):
        distMatrix[i + 1] = distMatrix[i] + radius / len(distMatrix)

    print(f"{distMatrix} :distMatrix ")
    # Generate base station and nodes
    np.random.seed(42)  # seed the random generator

    # Place base-stations randomly
    if mobility:
        algorithm = algo + "_" + power_algo + "_" + freqAlgo + "_" + str(len(freqSet)) + "_CR_" + str(
            len(CRSet)) + "_R_" \
                    + str(radius) + '_' + str(avgSendTime / (60 * 1000)) + 'min' + str(mobile_cal)
    else:
        algorithm = algo + "_" + power_algo + "_" + freqAlgo + "_" + str(len(freqSet)) + "_CR_" + str(
            len(CRSet)) + "_R_" + str(radius) + '_' + str(avgSendTime / (60 * 1000)) + 'min'

    if (algo == 'ADR'):
        simu_dir = join(logdir, exp_name, algorithm, "ADR_" + str(adr))
    else:
        simu_dir = join(logdir, exp_name, algorithm, "Learning Rate_" + str(learning_rate2))

    print(simu_dir)

    # Variables to keep track of every new simulation
    saveProbVar = False
    saveRatioVar = False
    saveTrafficVar = False
    saveEnergyVar = False

    # make folder
    if not exists(simu_dir):
        makedirs(simu_dir)

    filename = join(simu_dir, str('SimulationParameters') + '.txt')

    with open(filename, "a") as myfile:
        myfile.write(
            str(nrNodes) + ',' + str(nrIntNodes) + ',' + str(distribution) + ',' + str(avgSendTime) + ',' + str(
                horTime) + ',' + str(packetLength) + ',' + str(freqSet) + ',' + str(powSet))

    print('here again hello')
    file_1 = "hello"  # join(logdir, "bsList_bs" + str(nrBS) + "_nodes" + str(nrNodes) + ".npy")
    file_2 = "hello2"  # join(logdir, "nodeList_bs" + str(nrBS) + "_nodes" + str(nrNodes) + ".npy")

    if not os.path.exists(file_1):
        if not os.path.exists(file_2):
            print("Generated locations for {} base-stations and {} nodes".format(nrBS, nrNodes))
            BSLoc = np.zeros((nrBS, 3))
            if nrBS == 1:
                BSLoc[0] = [0, grid[0] * 0.5, grid[1] * 0.5]
            else:
                placeRandomly(nrBS, BSLoc, [grid[0] * 0.1, grid[0] * 0.9], [grid[1] * 0.1, grid[1] * 0.9])

            # Place nodes randomly
            nodeLoc = np.zeros((nrNodes, 14))
            print('Place randomly here')
            placeRandomlyInRange(nrNodes, nrIntNodes, nodeLoc, [0, grid[0]], [0, grid[1]], BSLoc,
                                 (bestDist, bestSF, bestBW), radius, phyParams, maxPtx, distribution, distMatrix)

            # save to file
            print('Gone')
            np.save(file_1, BSLoc)
            np.save(file_2, nodeLoc)
    else:
        # Load =location
        print('not randomly bro')
        print("\t Load locations for {} base-stations and {} nodes".format(nrBS, nrNodes))
        BSLoc = np.load(file_1)
        nodeLoc = np.load(file_2)

    # Simulation
    nTransmitted = 0
    nRecvd = 0

    BSList = BSLoc[0:nrBS, :]

    nodeList = nodeLoc[0:nrNodes, :]
    # print(nodeList)
    nodeList[0:nrIntNodes, -1] = 1  # Intelligent nodes

    nodeList[0:nrNodes, -2] = avgSendTime  # avgSendTime

    # Plotting - location
    plotLocations(BSLoc, nodeLoc, grid[0], grid[1], bestDist, distMatrix)

    env = simpy.Environment()
    env.process(cuckooClock(env))

    fname = str(algo) + '_' + str(power_algo) + '_' + str(nrIntNodes) + '_smartNodes_' + 'initial_' + str(
        initial) + '_infoMode_' + str(info_mode) + '_captureEffect_' + str(captureEffect) + '_interSFMode_' + str(
        interSFInterference)

    # Dictionary for packets history
    packet_history = dict()
    bsDict = {}  # setup empty dictionary for base-stations
    # print(BSList)
    # print('-1')
    for elem in BSList:
        bsDict[int(elem[0])] = myBS(int(elem[0]), (elem[1], elem[2]), interactionMatrix, nDemodulator, ackLength,
                                    freqSet, sfSet, captureThreshold)

    nodeDict = {}  # setup empty dictionary for nodes
    for elem in nodeList:
        # print(elem)
        node = myNode(int(elem[0]), (elem[1], elem[2]), elem[3:13], initial, sfSet, freqSet, powSet, CRSet,
                      BSList, interferenceThreshold, logDistParams, sensi, elem[13], info_mode, horTime, algo, freqAlgo,
                      simu_dir, fname, learning_rate2, adr, mobility, mobile_nodes, velocity, mobile_cal)
        nodeDict[node.nodeid] = node
        # print(nodeDict[node.nodeid].prob_x)
        # print("0")
        # if(node.nodeid ==)
        env.process(transmitPacket(env, node, bsDict, logDistParams, algo, power_algo, freqAlgo, packet_history))

    # save results
    print("1")
    env.process(saveProb(env, nodeDict, fname, simu_dir, saveProbVar))
    env.process(saveRatio(env, nodeDict, fname, simu_dir, saveRatioVar))
    env.process(saveEnergy(env, nodeDict, fname, simu_dir, saveEnergyVar))
    env.process(saveTraffic(env, nodeDict, fname, simu_dir, sfSet, freqSet, lambda_i, lambda_e, saveTrafficVar))

    # Variables to keep track of every new simulation
    saveProbVar = True
    saveRatioVar = True
    saveTrafficVar = True
    saveEnergyVar = True

    env.run(until=simtime)

    # reception
    nTransmitted = sum(node.packetsTransmitted for nodeid, node in nodeDict.items())
    nRecvd = sum(node.packetsSuccessful for nodeid, node in nodeDict.items())

    # print("hello3: " + str(node.packetsSuccessful for nodeid, node in nodeDict.items()))
    PacketReceptionRatio = nRecvd / nTransmitted

    # print results        
    print("================== Results ==================")
    print("# Transmitted = {}".format(nTransmitted))
    print("# Received = {}".format(nRecvd))
    print("# Ratio = {}".format(PacketReceptionRatio))
    #  print(nodeDict.items())
    return (bsDict, nodeDict)
