from numpy import zeros, random, where, zeros
from .loratools import getRXPower, getRXPower2, dBmTomW, airtime, getSNR, PRR_calculator
from tensorflow import function
from keras.models import load_model
# from bsFunctions import adr
import numpy as np
from pandas import DataFrame
from random import randint
import joblib

# from PowerOptimizer import model_im

# import random
countOsama = 0  ########################################


# @function
def model_imp(distance, m, var):
    #    from keras.models import load_model

    d = distance
    power_array = [x for x in range(5, 21, 3)]
    Output = np.zeros((6, 2))
    # model = load_model('PowerOptimizer.h5')

    # joblib.load(r'random_forest2.joblib')
    if var == 'tf':
        model = load_model(r'D:\Iot Research\Code\IoT-MAB2 (windows)\IoT-MAB2\lora\PowerOptimizer.h5')
    else:
        # model = joblib.load(r"D:\Iot Research\Code\IoT-MAB2 (windows)\IoT-MAB2\lora\random_forest2.joblib")
        model = joblib.load(r"lora\random_forest2.joblib")

    if var == 'tf':

        for SF in range(7, 13):
            for CR in range(5, 8, 2):
                inpt = np.array([[SF / 12], [CR / 7], [distance / 4500], [m / 100]])
                inpt = inpt.reshape(1, -1)
                Power = model(inpt)
                index = np.argmax(Power[0], axis=0)

                if index == 6:
                    Output[int(SF - 7)][int((CR - 5) / 2)] = 23  # None For now
                else:
                    # print('he')
                    # print(SF - 7)
                    # print((CR - 5) / 2)
                    Output[int(SF - 7)][int((CR - 5) / 2)] = power_array[index]
    else:
        for SF in range(7, 13):
            for CR in range(5, 8, 2):
                inpt = np.array([[distance], [SF], [CR], [m]])
                inpt = inpt.reshape(1, -1)
                # print(inpt)
                Power = model.predict(inpt)
                if Power == 23:
                    Output[int(SF - 7)][int((CR - 5) / 2)] = 23  # None For now
                else:
                    # print('he')
                    # print(SF - 7)
                    # print((CR - 5) / 2)
                    Output[int(SF - 7)][int((CR - 5) / 2)] = Power  # power_array[index]
    # print(Output)
    return Output


class myPacket():
    """ LPWAN Simulator: packet
    Base station class
   
    |category /LoRa
    |keywords lora
    
    \param [IN] nodeid: id of the node
    \param [IN] bsid: id of the base station
    \param [IN] dist: distance between node and bs
    \param [IN] transmitParams: physical layer parameters
                [sf, rdd, bw, packetLength, preambleLength, syncLength, headerEnable, crc, pTX, period] 
    \param [IN] logDistParams: log shadowing channel parameters
    \param [IN] sensi: sensitivity matrix
    \param [IN] setActions: set of possible actions
    \param [IN] nrActions: number of actions
    \param [IN] sfSet: set of spreading factors
    \param [IN] prob: probability
    """

    SFdict = {7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0}
    powerDict = {2: 0, 5: 0, 8: 0, 11: 0, 14: 0, 17: 0, 20: 0}
    LostNoise = 0  ########### Number of packets lost due to noise

    def __init__(self, nodeid, bsid, dist, transmitParams, logDistParams, sensi, setActions, nrActions, sfSet,
                 prob, adr, algo, mobility, mobile_nodes, velocity, mobile_cal):  # choosenAction):
        self.nodeid = nodeid
        self.bsid = bsid
        self.dist = dist

        self.mu = random.randint(1, 3)

        if (self.mu == 3):
            self.mu = 100
        elif (self.mu == 2):
            self.mu = 5

        # params

        self.sf = int(transmitParams[0])
        self.rdd = int(transmitParams[1])
        self.bw = int(transmitParams[2])
        self.CR = 5  # Replace with a CR parameter from input
        self.packetLength = int(transmitParams[3])
        self.preambleLength = int(transmitParams[4])
        self.syncLength = transmitParams[5]
        self.headerEnable = int(transmitParams[6])
        self.crc = int(transmitParams[7])
        self.pTXmax = int(transmitParams[8])
        self.sensi = sensi
        self.sfSet = sfSet

        # learn strategy
        self.setActions = setActions
        self.nrActions = nrActions
        self.prob = [prob[x] for x in prob]
        # self.choosenAction = choosenAction
        # self.sf, self.freq, self.pTX = self.setActions[self.choosenAction]
        self.sf = None
        self.adr = adr
        self.freq = None
        if algo == 'ADR':
            self.sf = 7
            self.pTX = 14
            self.CR = 5
            self.bw = 125

        # Here we should add the machine learning model to predict the output

        self.pTX = self.pTXmax

        self.mobility = mobility
        self.velocity = velocity
        self.mobile_cal = mobile_cal
        self.mobile_nodes = mobile_nodes

        if mobile_cal:
            self.powerArray = []
            for x in range(1000, 6000, 1000):
                self.powerArray.append(model_imp(x, 100, 'sc'))
            # print('here In append')
            # print(self.powerArray)
        else:
            self.powerArray = model_imp(self.dist, self.mu, 'sc')

        # self.powerArray = model_imp(self.dist, self.mu, 'sc')
        if self.nodeid == 300:
            x = self.powerArray[:, :]
            y = DataFrame(x)
            # print(y)
            x = y.rename(columns={0: 'Power'}, inplace=False)
            print(self.dist)
            print(f'Predicted Power Array is:\n {x}')
            print(self.powerArray)

        # received params
        self.rectime = airtime(transmitParams[0:8])
        self.pRX = getRXPower(self.pTX, self.dist, logDistParams)
        self.SNR = None
        self.signalLevel = None

        # measurement params
        self.packetNumber = 0
        self.isLost = False
        self.isLostNoise = False  #########################
        self.isCritical = False
        self.isCollision = False
        self.PRR = None

    def computePowerDist(self, bsDict, logDistParams):
        """ Get the power distribution .
        Parameters
        ----------
        self : packet
            Packet.
        bsDict: dictionary
            Dictionary of BSs
        Returns
        -------
        signalLevel: dictionary
            The power contribution of a packet in various frequency buckets for each BS
    
        """
        signal = self.getPowerContribution()
        signalLevel = {x: signal[x] for x in signal.keys() & bsDict[self.bsid].signalLevel.keys()}
        return signalLevel

    def adr_algo(self, packet_history, freqSet):
        history = packet_history[self.nodeid]
        adr_margin_db = 10
        SNR_margin = np.array([-7.5, -10, -12.5, -15, -17.5, -20])
        sf = self.sf
        power = self.pTX
        if len(history) == 20:
            if self.adr == 0:
                snr_history_val = np.amax(np.asanyarray(history))
            elif self.adr == 1:
                snr_history_val = np.amin(np.asanyarray(history))
            elif self.adr == 2:
                snr_history_val = np.average(np.asanyarray(history))
            else:
                # default
                snr_history_val = np.amax(np.asanyarray(history))

            adr_required_snr = SNR_margin[int(self.sf - 7)]

            snr_margin = snr_history_val - adr_required_snr - adr_margin_db
            # print("SNR Margin: {}".format(snr_margin))
            num_steps = np.round(snr_margin / 3)
            # print(num_steps)
            dr_changing = 0
            if num_steps > 0:
                num_step_possible = self.sf - 7

                if num_step_possible < num_steps:
                    dr_changing = num_step_possible
                    num_steps_remaining = num_steps - num_step_possible

                    decrease_tx_power = num_steps_remaining * 3  # the remainder is used  to decrease the TXpower by
                    # 3dBm per step, until TXmin is reached. TXmin = 2 dBm for EU868.
                    power = np.amax([self.pTX - decrease_tx_power, 5])
                elif num_steps <= num_step_possible:
                    dr_changing = num_steps
                # elif

                # use default decrease tx power (0)

                sf = np.max([self.sf - dr_changing, 7])

            elif num_steps < 0:
                # TX power is increased by 3dBm per step, until TXmax is reached (=14 dBm for EU868).
                num_steps = - num_steps  # invert so we do not need to work with negative numbers
                power = np.amin([self.pTX + (num_steps * 3), 23])

        rand = randint(0, len(freqSet) - 1)
        # print('rand:')
        # print(rand)
        freq = freqSet[rand]

        return sf, freq, power, 5

    def updateTXSettings(self, bsDict, logDistParams, prob, power_algo, algo, freqAlgo, rg, packet_history, freqSet,
                         packetsSuccessful, prevTime, newTime):
            """ Update the TX settings after frequency hopping.
            Parameters
            ----------
            bsDict: dictionary
                Dictionary of BSs
            logDistParams: list
                Channel parameters, e.x., log-shadowing model: (gamma, Lpld0, d0)]

            Returns
            isLost: bool
                Packet is lost ot not by compare the pRX with RSSI
            -------

            """

            if self.mobility:
                time_passed = (newTime - prevTime) / 1000  # Time in seconds
                distance_travelled = time_passed * self.velocity
                radius = 5e3  # Hardcoded for now
                # if self.nodeid == 150:
                #     print('------')
                #     print(time_passed)
                #     print(self.dist)
                #     print(distance_travelled)
                print('should not be here')
                self.dist = (self.dist + distance_travelled) % radius  # Ensure that node remains inside a fixed radius

            if (self.dist > 6e3):
                x = 1
                # print("Previous SF: {}, Previous pTX: {} and Distacne {} ".format(self.sf, self.pTX, self.dist))
            # self.sf, self.freq, self.pTX, self.CR = self.adr_algo(packet_history)

            self.freq = freqSet[0]

            if (self.dist > 6e3):
                x = 1
                # print("New SF: {}, Previous pTX: {}".format(self.sf, self.pTX))

            self.packetNumber += 1
            if algo == 'Random':
                prob = np.random.rand(self.nrActions)
                prob = prob / sum(prob)
            self.prob = prob

            again = True

            while again:
                self.choosenAction = random.choice(self.nrActions, p=self.prob)
                if algo != 'ADR':
                    self.sf, self.freq, self.pTX, self.CR = self.setActions[self.choosenAction]

                if power_algo == 'DL' and algo != 'ADR':
                    if self.mobile_cal:
                        #     print('In packets')
                        #     print(self.powerArray)
                        #     print(self.dist)
                        #     print(self.mobile_cal)
                        self.pTX = self.powerArray[int(self.dist / 1000)][int(self.sf - 7)][int((self.CR - 5) / 2)]
                    else:
                        self.pTX = self.powerArray[int(self.sf - 7)][int((self.CR - 5) / 2)]
                # print(self.pTX)

                if self.pTX != 23 and algo != 'ADR':
                    again = False

            if freqAlgo == "Random":
                rand = randint(0, len(freqSet) - 1)
                self.freq = freqSet[rand]
            elif freqAlgo == "Uniform":
                self.freq = freqSet[self.nodeid % len(freqSet)]

            if algo == 'ADR' and packetsSuccessful % 20 == 0:  # Run ADR after receiving every 20 packets
                # print('hello')
                self.sf, self.freq, self.pTX, self.CR = self.adr_algo(packet_history, freqSet)
                # print('Whey not 1 {} {} '.format(freqSet, self.freq))

            # Update dictionary
            # if self.sf in myPacket.SFdict:
            #
            # else:
            #     myPacket.SFdict[self.sf] = 1
            myPacket.SFdict[self.sf] += 1
            myPacket.powerDict[self.pTX] += 1

            self.pRX = getRXPower2(self.pTX, self.freq, self.dist, logDistParams)
            var = False
            if self.SNR:
                var = True
                temp = self.SNR
            self.SNR = getSNR(self.pRX, self.mu, 1, 1, rg)

            self.signalLevel = self.computePowerDist(bsDict, logDistParams)
            self.PRR = PRR_calculator(self.SNR, self.sf, self.CR, self.bw,
                                      self.packetLength * 8)  # Give packet length in bits
            if self.pRX >= self.sensi[self.sf - 7, 1 + int(self.bw / 250)]:
                self.isLostNoise = False
                self.isLost = False
            else:
                myPacket.LostNoise += 1

                #   print(myPacket.LostNoise)
                self.isLostNoise = True
                self.isLost = True
                # print(self.pRX)
                # print ("Node " + str(self.nodeid) + ": packet is lost (smaller than RSSI)!")
            # print(str(self.SNR) + " SF: " + str(self.sf) + "CR" + str(self.CR) + "BW" + str(self.bw) + "length" + str(
            # self.packetLength) + " distance " + str(self.dist) + " Power " + str(self.pTX))
            # print(self.PRR)
            if self.PRR < 0.01:
                self.isLost = True
            else:
                self.isLost = False

            self.isCritical = False

    def getAffectedFreqBuckets(self):
        """ Get the list of affected frequency buckets from [fc-bw/2 fc+bw/2].
        Parameters
        ----------
        
        Returns
        fRange: list
            List of frequencies that effected by the using frequency
        -------
        """
        low = self.freq - self.bw / 2  # Note: this is approx due to integer division for 125
        high = self.freq + self.bw / 2  # Note: this is approx due to integer division for 125
        lowBucketStart = int(low - (low % 200) + 100)
        highBucketEnd = int(high + 200 - (high % 200) - 100)

        # the +1 ensures that the last value is included in the set
        return range(lowBucketStart, highBucketEnd + 1, 200)

    def getPowerContribution(self):
        """ Get the power contribution of a packet in various frequency buckets.
        Parameters
        ----------

        Returns
        powDict: dic
            Power distribution by frequency
        -------
    
        """
        freqBuckets = self.getAffectedFreqBuckets()
        powermW = dBmTomW(self.pRX)
        # print(self.pRX, powermW)
        signal = zeros((6, 1))
        full_setSF = [7, 8, 9, 10, 11, 12]
        idx = full_setSF.index(self.sf)
        # print(idx)
        signal[idx] = powermW
        # print(signal)
        return {freqBuckets[0]: signal}
