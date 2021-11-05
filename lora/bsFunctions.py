""" LPWAN Simulator: Hepper functions
============================================
Utilities (:mod:`lora.bsFunctions`)
============================================
.. autosummary::
   :toctree: generated/
   transmitPacket           -- Transmission process with discret event simulation.
   cuckooClock              -- Notify the simulation time (for each 1k hours).
   saveProb                 -- Save the probability profile of each node.
"""
import os
import random
import numpy as np
from os.path import join
from collections import deque
from .loratools import airtime, dBmTomW
from datetime import datetime

count = 0
dict_energy = {2: 14.9, 5: 16.3, 8: 18.5, 11: 23, 14: 31.7, 17: 90,
               20: 125}  # , 23:0} # 23 pTX not allowed so no tranmission

from .packet import myPacket


# Transmit
def transmitPacket(env, node, bsDict, logDistParams, algo, power_algo, freqAlgo, packet_history):
    """ Transmit a packet from node to all BSs in the list.
    Parameters
    ----------
    env : simpy environement
        Simulation environment.
    node: my Node
        LoRa node.
    bsDict: dict
        list of BSs.
    logDistParams: list
        channel params
    algo: string
        learning algorithm
    Returns
    -------
    """
    while True:
        # The inter-packet waiting time. Assumed to be exponential here.
        prevTime = env.now
        yield env.timeout(random.expovariate(1 / float(node.period)))
        newTime = env.now

        if node.nodeid not in packet_history:
            packet_history[node.nodeid] = deque(maxlen=20)

        # update settings if any
        node.updateTXSettings()
        node.resetACK()
        node.packetNumber += 1

        # send a virtual packet to each base-station in range and those we may affect
        # print(node.prob)
        # print(node.nodeid)
        loop = False
        for bsid, dist in node.proximateBS.items():
            loop = True
            prob_temp = [node.prob[x] for x in node.prob]
            # print(prob_temp)
            prevSF = node.packets[bsid].sf
            node.packets[bsid].updateTXSettings(bsDict, logDistParams, prob_temp, power_algo, node.algo, freqAlgo,
                                                node.rg, packet_history, node.freqSet, node.packetsSuccessful,
                                                prevTime, newTime)
            newSF = node.packets[bsid].sf
            if prevSF != newSF:
                node.ackLosts = 0
            # print("inside")
            # print(node.nodeid, bsid)
            bsDict[bsid].addPacket(node.nodeid, node.packets[bsid])
            bsDict[bsid].resetACK()

        # wait until critical section starts
        Tcritical = (2 ** node.packets[0].sf / node.packets[0].bw) * (
                node.packets[0].preambleLength - 5)  # time until the start of the critical section
        # print(Tcritical)

        yield env.timeout(Tcritical)

        # make the packet critical on all nearby basestations
        for bsid in node.proximateBS.keys():
            bsDict[bsid].makeCritical(node.nodeid)

        Trest = airtime((node.packets[0].sf, node.packets[0].rdd, node.packets[0].bw, node.packets[0].packetLength,
                         node.packets[0].preambleLength, node.packets[0].syncLength, node.packets[0].headerEnable,
                         node.packets[0].crc)) - Tcritical  # time until the rest of the message completes

        yield env.timeout(Trest)

        successfulRx = False
        ACKrest = 0

        # transmit ACK
        for bsid in node.proximateBS.keys():
            # print("=====> eval bs {}".format(bsid))
            if bsDict[bsid].removePacket(node.nodeid):
                bsDict[bsid].addACK(node.nodeid, node.packets[bsid])
                ACKrest = airtime((node.packets[0].sf, node.packets[0].rdd, node.packets[0].bw,
                                   node.packets[0].packetLength, node.packets[0].preambleLength,
                                   node.packets[0].syncLength, node.packets[0].headerEnable,
                                   node.packets[0].crc))  # time until the ACK completes
                yield env.timeout(ACKrest)
                node.addACK(bsDict[bsid].bsid, node.packets[bsid])
                successfulRx = True

        # update probability        
        node.packetsTransmitted += 1
        if node.packets[0].isLostNoise:
            node.packetsLostNoise += 1
        # print('Hello: ' + str (node.packetsLostNoise))

        # self.packet_num_received_from[from_node.id] = 0
        # self.distinct_bytes_received_from[from_node.id] = 0
        actual = airtime((node.packets[0].sf, node.packets[0].rdd, node.packets[0].bw, node.packets[0].packetLength,
                          node.packets[0].preambleLength, node.packets[0].syncLength, node.packets[0].headerEnable,
                          node.packets[0].crc))

        node.energy += actual * dBmTomW(node.packets[0].pTX) * (3.0) / 1e6  # V = 3.0     # voltage XXX
        node.energyEARN += actual * dict_energy[node.packets[0].pTX] * 3.3 / 1e6

        # if((actual > 1400 or actual < 1000) and count < 100):
        #     count += 1
        #     print(node.packets[0].rectime)
        #     print(node.packets[0].sf)
        #     print(actual)

        # print(actual)
        if successfulRx:

            packet_history[node.nodeid].append(node.packets[0].SNR)
            # print(packet_history[node.nodeid])
            #  print(packet_history)
            # print('Hello\n')
            if (node.packets[0].dist > 6e3):
                x = 1
                # print('NOde history = {}'.format(packet_history[node.nodeid]))

            if node.info_mode in ["NO", "PARTIAL"]:
                node.packetsSuccessful += 1
                node.packetsPRR += node.packets[0].PRR
                node.transmitTime += node.packets[0].rectime
            elif node.info_mode == "FULL":
                if not node.ack[0].isCollision:
                    node.packetsSuccessful += 1
                    node.packetsPRR += node.packets[0].PRR
                    node.transmitTime += node.packets[0].rectime
            node.updateProb(algo)
        # print("Probability of action from node " +str(node.nodeid)+ " at (t+1)= {}".format(int(1+env.now/(6*60*1000))))
        # print(node.prob)
        # print(node.weight)
        # wait to next period
        yield env.timeout(float(node.period) - Tcritical - Trest - ACKrest)
        # input()


def cuckooClock(env):
    """ Notifies the simulation time.
    Parameters
    ----------
    env : simpy environement
        Simulation environment.
    Returns
    -------
    """
    while True:
        yield env.timeout(1000 * 3600000)
        print("Running {} kHrs".format(env.now / (1000 * 3600000)))


def saveProb(env, nodeDict, fname, simu_dir, temp):
    """ Save probabilities every to file
    Parameters
    ----------
    env : simpy environement
        Simulation environment.
    nodeDict:dict
        list of nodes.
    fname: string
        file name structure
    simu_dir: string
        folder
    Returns
    -------
    """
    while True:
        yield env.timeout(100 * 36000000)  # orignal 3600000
        # write prob to file
        for nodeid in nodeDict.keys():
            if nodeDict[nodeid].node_mode != "UNIFORM":
                filename = join(simu_dir, str('prob_' + fname) + '_id_' + str(nodeid) + '.csv')
                # print(filename)
                # print(nodeDict[nodeid].prob.values())

                save = str(list(nodeDict[nodeid].prob.values()))[1:-1]
                if nodeDict[nodeid].algo == 'exp4':
                    # print('hello2')
                    save = save + "\n Prob_x:" + str(nodeDict[nodeid].prob_x) + "\n weight: " + str(
                        nodeDict[nodeid].weight_x)
                    save = save + "\n weight_e: " + str(nodeDict[nodeid].weight_e)
                elif nodeDict[nodeid].algo == 'exp3s' or nodeDict[nodeid].algo == 'exp3' or nodeDict[
                    nodeid].algo == 'exp4o':
                    save = save + "\n weight: " + str(nodeDict[nodeid].weight)

                if not temp:
                    now = datetime.now()
                    current_time = now.strftime("%H:%M:%S")
                    res = "\nStarting: " + current_time + f"\n {nodeDict[nodeid].packets[0].dist}"
                    print(res)
                    break
                    print('hello')
                    # res = "\nStarting" + res
                else:
                    res = ""

                if os.path.isfile(filename):
                    res = res + "\n" + save
                else:
                    res = res + save
                with open(filename, "a") as myfile:
                    # myfile.truncate(0)
                    myfile.write(res)
                myfile.close()

        temp = True  # Just do this one time in the gateway


def saveRatio(env, nodeDict, fname, simu_dir, temp):
    """ Save packet reception ratio to file
    Parameters
    ----------
    env : simpy environement
        Simulation environment.
    nodeDict:dict
        list of nodes.
    fname: string
        file name structure
    simu_dir: string
        folder
    Returns
    -------
    """
    while True:
        yield env.timeout(100 * 36000 * 10)  # one less zero in energy also
        # write packet reception ratio to file
        nTransmitted = 0
        nRecvd = 0
        PacketReceptionRatio = 0
        nTransmitted = sum(nodeDict[nodeid].packetsTransmitted for nodeid in nodeDict.keys())
        nRecvd = sum(nodeDict[nodeid].packetsSuccessful for nodeid in nodeDict.keys())
        nPRR = sum(nodeDict[nodeid].packetsPRR for nodeid in nodeDict.keys())
        PacketReceptionRatio = nRecvd / nTransmitted
        PacketReceptionRatioPRR = float(nPRR / nTransmitted)
        filename = join(simu_dir, str('ratio_') + '.csv')
        filename2 = join(simu_dir, str('PRR_ratio_') + '.csv')
        SFdictFile = join(simu_dir, str('SFdict_') +'.csv')
        PowerdictFile = join(simu_dir, str('Powerdict_') + '.csv')

        if not temp:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            res_temp = "\nStarting: " + current_time + '\n'
            temp = True  # Just do this one time in the gateway
        else:
            res_temp = ""

        if os.path.isfile(filename):
            res = res_temp + "\n" + str(PacketReceptionRatio)
        else:
            res = res_temp + str(PacketReceptionRatio)
        with open(filename, "a") as myfile:
            myfile.write(res)
        myfile.close()

        if os.path.isfile(filename2):
            res = res_temp + "\n" + str(PacketReceptionRatioPRR)
        else:
            res = res_temp + str(PacketReceptionRatioPRR)
        with open(filename2, "a") as myfile:
            myfile.write(res)

        if os.path.isfile(SFdictFile):
            res = res_temp + "\n" + str((myPacket.SFdict.items()))
        else:
            res = res_temp + str((myPacket.SFdict.items()))
        with open(SFdictFile, "a") as myfile:
            myfile.write(res)

        if os.path.isfile(PowerdictFile):
            res = res_temp + "\n" + str(myPacket.powerDict.items())
        else:
            res = res_temp + str((myPacket.powerDict.items()))
        with open(PowerdictFile, "a") as myfile:
            myfile.write(res)
        myfile.close()


def saveEnergy(env, nodeDict, fname, simu_dir, temp):
    """ Save energy to file
    Parameters
    ----------
    env : simpy environement
        Simulation environment.
    nodeDict:dict
        list of nodes.
    fname: string
        file name structure
    simu_dir: string
        folder
    Returns
    -------
    """
    while True:
        yield env.timeout(100 * 36000 * 10)
        # compute and wirte energy consumption to file
        totalEnergy = sum(nodeDict[nodeid].energy for nodeid in nodeDict.keys())
        totalEnergyEARN = sum(nodeDict[nodeid].energyEARN for nodeid in nodeDict.keys())
        nTransmitted = sum(nodeDict[nodeid].packetsTransmitted for nodeid in nodeDict.keys())
        nRecvd = sum(nodeDict[nodeid].packetsSuccessful for nodeid in nodeDict.keys())
        filename = join(simu_dir, str('energy_') + '.csv')

        if not temp:

            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            res = "\nStarting: " + current_time

            temp = True  # Just do this one time in the gateway
        else:
            res = ""

        if os.path.isfile(filename):
            res = res + "\n" + str(totalEnergy) + " " + str(nTransmitted) + " " + str(nRecvd) + " " + str(
                totalEnergyEARN)
        else:
            res = res + '\n' + str(totalEnergy) + " " + str(nTransmitted) + " " + str(nRecvd) + " " + str(totalEnergyEARN)
        with open(filename, "a") as myfile:
            myfile.write(res)
        myfile.close()


def saveTraffic(env, nodeDict, fname, simu_dir, sfSet, freqSet, lambda_i, lambda_e, temp):
    """ Save norm traffic and throughput to file
    Parameters
    ----------
    env : simpy environement
        Simulation environment.
    nodeDict:dict
        list of nodes.
    fname: string
        file name structure
    simu_dir: string
        folder
    sfSet: list
        set of possible sf
    freqSet: list
        set of possible freq
    Returns
    -------
    """
    while True:
        yield env.timeout(100 * 3600000)  # orignal 10
        # compute and wirte traffic and throughtput to file
        # total_Ts = sum(nodeDict[nodeid].transmitTime for nodeid in nodeDict.keys())
        Gsc = np.zeros((len(sfSet), len(freqSet)))
        Tsc = np.zeros((len(sfSet), len(freqSet)))
        Gsc += lambda_e
        print('hello')
        for nodeid in nodeDict.keys():
            if nodeDict[nodeid].packets[0].sf != None:
                if nodeDict[nodeid].packets[0].freq != None:
                    si = sfSet.index(nodeDict[nodeid].packets[0].sf)
                    ci = freqSet.index((nodeDict[nodeid].packets[0].freq))
                    Gsc[si, ci] += lambda_i

        for i in range(len(sfSet)):
            Gsc[i, :] *= airtime((sfSet[i], nodeDict[0].packets[0].rdd, nodeDict[0].packets[0].bw,
                                  nodeDict[0].packets[0].packetLength, nodeDict[0].packets[0].preambleLength,
                                  nodeDict[0].packets[0].syncLength, nodeDict[0].packets[0].headerEnable,
                                  nodeDict[0].packets[0].crc))

        for i in range(len(sfSet)):
            for j in range(len(freqSet)):
                Tsc[i][j] = Gsc[i][j] * np.exp(-2 * Gsc[i][j])

        if not temp:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S\n")
            temp_res = "\nStarting: " + current_time
            temp = True  # Just do this one time in the gateway
        else:
            temp_res = ""

        filename = join(simu_dir, str('traffic_' + fname) + '.csv')
        if os.path.isfile(filename):
            res = temp_res + "\n" + str(sum(sum(Gsc))) + " " + str(sum(sum(Tsc)))
        else:
            res = temp_res + str(sum(sum(Gsc))) + " " + str(sum(sum(Tsc)))
        with open(filename, "a") as myfile:
            myfile.write(res)
        print(Gsc)
        filename2 = join(simu_dir, str('traffic2_' + fname) + '.csv')
        save = str(list(Gsc))[1:-1]
        if os.path.isfile(filename2):
            res = temp_res + "\n" + save
        else:
            res = temp_res + save

        with open(filename2, "a") as myfile:
            myfile.write(res)
        myfile.close()

        myfile.close()

#
# def adr(env, node, bsDict, logDistParams, algo, power_algo, packet_history):
#     history = packet_history[node.nodeid]
#     adr_margin_db = 10
#     SNR_margin = np.array([-7.5, -10, -12.5, -15, -17.5, -20])
#     if len(history) is 20:
#         if node.adr == 0:
#                 snr_history_val = np.amax(np.asanyarray(history))
#         elif node.adr == 1:
#             snr_history_val = np.amin(np.asanyarray(history))
#         elif node.adr == 2:
#             snr_history_val = np.average(np.asanyarray(history))
#         else:
#             # default
#             snr_history_val = np.amax(np.asanyarray(history))
#
#         adr_required_snr = SNR_margin[int(node.sf - 7)]
#
#         snr_margin = snr_history_val - adr_required_snr - adr_margin_db
#         print(f"SNR margin {snr_margin}")
#         num_steps = np.round(snr_margin / 3)
#         dr_changing = 0
#         if num_steps > 0:
#             num_step_possible = node.sf - 7
#             if num_step_possible > num_steps:
#                 dr_changing = num_step_possible
#                 num_steps_remaining = num_steps - num_step_possible
#                 decrease_tx_power = num_steps_remaining * 3  # the remainder is used  to decrease the TXpower by
#                 # 3dBm per step, until TXmin is reached. TXmin = 2 dBm for EU868.
#                 node.power = np.amax([node.power - decrease_tx_power, 5])
#             elif num_steps <= num_step_possible:
#                 dr_changing = num_steps
#                 # use default decrease tx power (0)
#             node.sf = node.sf - dr_changing
#
#
#
#     x = 2
