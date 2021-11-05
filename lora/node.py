from __future__ import division
import numpy as np
from .loratools import getDistanceFromPower
from .packet import myPacket
from numpy.random import Generator, PCG64
import random


class myNode():
    """ LPWAN Simulator: node
    Base station class
   
    |category /LoRa
    |keywords lora
    
    \param [IN] nodeid: id of the node
    \param [IN] position: position of the node in format [x y]
    \param [IN] transmitParams: physical layer's parameters
    \param [IN] bsList: list of BS
    \param [IN] interferenceThreshold: interference threshold
    \param [IN] logDistParams: log shadowing channel parameters
    \param [IN] sensi: sensitivity matrix
    \param [IN] nSF: number of spreading factors
    
    """
    Counter = 0
    def fixedExperts(self):
        # Expert 1 calcuations
        expert1 = np.zeros(self.nrActions)
        freqSetLen = len(self.freqSet)
        if (self.freq_algo == "Random" or self.freq_algo == "Uniform"):
            freqSetLen = 1
        x = 0
        for i in range(len(self.sfSet)):
            for j in range(freqSetLen):
                for k in range(len(self.powerSet)):
                    for l in range(len(self.CRSet)):

                        if l == 0:
                            expert1[x] = ((i + 7) * 4 / 5) / 2 ** (i + 7)  # For CR 5
                            x += 1
                        elif l == 1:
                            expert1[x] = ((i + 7) * 4 / 7) / 2 ** (i + 7)  # For CR 7
                            x += 1

        # for x in range(self.nrActions/len(self.freqSet)):
        #     if (x % 2 == 0):
        #         expert1[x] = ((x + 7) * 4 / 5) / 2 ** (x + 7)  # For CR 5
        #     else:
        #         expert1[x] = ((x + 7) * 4 / 7) / 2 ** (x + 7)  # For CR 7

        expert1 = expert1 / sum(expert1)  # Normalizing

        expertMatrix = expert1  # np.array([expert1]

        return expertMatrix

    def __init__(self, nodeid, position, transmitParams, initial, sfSet, freqSet, powSet, CRSet, bsList,
                 interferenceThreshold, logDistParams, sensi, node_mode, info_mode, horTime, algo, freq_algo, simu_dir,
                 fname, learning_rate2, adr, mobility, mobile_nodes, velocity, mobile_cal):

        # Mobile Nodes

        self.mobility = mobility
        self.mobile_nodes = mobile_nodes
        self.velocity = velocity
        self.mobile_cal = mobile_cal

        self.nodeid = nodeid  # id
        np.set_printoptions(precision=3)
        self.x, self.y = position  # location
        if node_mode == 0:
            self.node_mode = initial
        else:
            self.node_mode = "SMART"

        self.info_mode = info_mode  # 'no', 'partial', 'full'
        self.rg = Generator(PCG64(12345))
        self.bw = int(transmitParams[2])
        self.period = float(transmitParams[9])
        self.pTXmax = max(powSet)  # max pTX
        self.sensi = sensi
        self.adr = adr  # Type of adr algorithm we are using

        #

        # generate proximateBS
        self.proximateBS = self.generateProximateBS(bsList, interferenceThreshold, logDistParams)

        # set of actions
        self.freqSet = freqSet
        self.powerSet = powSet
        self.CRSet = CRSet

        if self.info_mode == "NO":
            self.sfSet = sfSet
        else:
            self.sfSet = self.generateHoppingSfFromDistance(sfSet, logDistParams)

        if freq_algo != "Random" and freq_algo != "Uniform":
            self.setActions = [(self.sfSet[i], self.freqSet[j], self.powerSet[k], self.CRSet[l]) for i in
                               range(len(self.sfSet)) for j in
                               range(len(self.freqSet)) for k in range(len(self.powerSet)) for l in
                               range(len(self.CRSet))]
        else:
            self.setActions = [(self.sfSet[i], self.freqSet[j], self.powerSet[k], self.CRSet[l]) for i in
                               range(len(self.sfSet)) for j in
                               range(1) for k in range(len(self.powerSet)) for l in
                               range(len(self.CRSet))]

        self.nrActions = len(self.setActions)
        self.initial = initial
        self.nrExperts = 2
        self.algo = algo
        self.freq_algo = freq_algo

        # learning algorithm
        if algo == 'exp3' or algo == 'Random':
            self.learning_rate = np.minimum(1, np.sqrt(
                (self.nrActions * np.log(self.nrActions)) / ((horTime) * (np.exp(1.0) - 1))))
            self.alpha = None
        elif algo == 'exp3s' or algo == 'exp4o':
            self.learning_rate = np.minimum(1, np.sqrt((self.nrActions * np.log(self.nrActions * horTime)) / horTime))
            self.alpha = 1 / horTime
            # weight and prob for learning
        elif algo == 'exp4':
            self.learning_rate = np.minimum(1, np.sqrt((self.nrActions * np.log(self.nrActions * horTime)) / horTime))
            self.alpha = 1 / horTime
            self.learning_rate2 = learning_rate2
        self.weight = {x: 1 for x in range(0, self.nrActions)}
        if algo == 'exp4o':
            self.weight = {x: 2 ** (5 - x) for x in range(0, self.nrActions)}
        self.weight_e = np.ones((self.nrExperts))
        self.weight_x = self.weight
        if self.initial == "RANDOM":
            prob = np.random.rand(self.nrActions)
            prob = prob / sum(prob)
        else:
            prob = (1 / self.nrActions) * np.ones(self.nrActions)

        self.prob = {x: prob[x] for x in range(0, self.nrActions)}
        # prob_x is the probability for exp4 algorithm
        # if(algo == 'exp4'):
        self.prob_x = np.ones(self.nrActions)
        self.prob_x = self.prob_x / sum(self.prob_x)

        if algo == 'exp4':
            for j in range(self.nrActions):
                W_t = np.sum(self.weight_e)

                expertMat = self.fixedExperts()
                # if self.nodeid == 300 and j == 5:
                #
                #     print('First:')
                #     print(f"Expert Matrix: {expertMat}")

                expertMat = np.array([expertMat, self.prob_x])
                #if self.nodeid == 300 and j == 5:
                    #print(f"Weight w: {self.weight_e}")
                    #print(f"Expert Matrix: {expertMat}")
                temp = np.zeros((4, 1))

                for i in range(self.nrExperts):
                    temp[i] = self.weight_e[i] * expertMat[i][j]

                temp_sum = np.sum(temp)
                prob[j] = (1 - self.learning_rate2) * temp_sum / W_t + self.learning_rate2 / self.nrActions
                if self.nodeid == 300 and j == 5:
                   # self.Counter = 1
                    x = np.array(list(prob))
                    #print(f"Final Probability vector: {x}")

        # generate packet and ack
        self.packets = self.generatePacketsToBS(transmitParams, logDistParams)
        self.ack = {}
        self.ackLosts = 0  # Variable for number of acks lost

        # measurement params
        self.packetNumber = 0
        self.packetsTransmitted = 0
        self.packetsLostNoise = 0  ######## number of packets lost due to noise
        self.packetsSuccessful = 0
        self.packetsPRR = 0
        self.transmitTime = 0
        self.energy = 0
        self.energyEARN = 0 #Energy for EARN comparision

    def generateProximateBS(self, bsList, interferenceThreshold, logDistParams):
        """ Generate dictionary of base-stations in proximity.
        Parameters
        ----------
        bsList : list
            list of BSs.
        interferenceThreshold: float
            Interference threshold
        logDistParams: list
            Channel parameters
        Returns
        -------
        proximateBS: list
            List of proximated BS
        """

        maxInterferenceDist = getDistanceFromPower(self.pTXmax, interferenceThreshold, logDistParams)
        dist = np.sqrt((bsList[:, 1] - self.x) ** 2 + (bsList[:, 2] - self.y) ** 2)
        # print(self.nodeid)
        index = np.nonzero(dist <= maxInterferenceDist * 10)
        # print(f"Max Dist: {maxInterferenceDist})")
        # print (f" distance: {dist}")

        proximateBS = {}  # create empty dictionary
        for i in index[0]:
            # print('not here')
            proximateBS[int(bsList[i, 0])] = dist[i]

        return proximateBS

    def generatePacketsToBS(self, transmitParams, logDistParams):
        """ Generate dictionary of base-stations in proximity.
        Parameters
        ----------
        transmitParams : list
            Transmission parameters.
        logDistParams: list
            Channel parameters
        Returns
        -------
        packets: packet
            packets at BS
        """
        packets = {}  # empty dictionary to store packets originating at a node

        for bsid, dist in self.proximateBS.items():
            packets[bsid] = myPacket(self.nodeid, bsid, dist, transmitParams, logDistParams, self.sensi,
                                     self.setActions, self.nrActions, self.sfSet, self.prob, self.adr,
                                     self.algo, self.mobility, self.mobile_nodes, self.velocity, self.mobile_cal)  # choosenAction)
        return packets

    # print("probability of node " +str(self.nodeid)+" is: " +str(self.prob))

    def generateHoppingSfFromDistance(self, sfSet, logDistParams):
        """ Generate the sf hopping sequence from distance
        Parameters
        ----------
        logDistParams: list in format [gamma, Lpld0, d0]
            Parameters for log shadowing channel model.
        Returns
        -------
    
        """
        sfBuckets = []
        gamma, Lpld0, d0 = logDistParams
        dist = self.proximateBS[0]

        if self.bw == 125:
            bwInd = 0
        else:
            bwInd = 1
        Lpl = self.pTXmax - self.sensi[:, bwInd + 1]

        LplMatrix = Lpl.reshape((6, 1))
        distMatrix = np.dot(d0, np.power(10, np.divide(LplMatrix - Lpld0, 10 * gamma)))

        for i in range(6):
            if dist <= distMatrix[0, 0]:
                minSF = 7
            elif distMatrix[i, 0] <= dist < distMatrix[i + 1, 0]:
                minSF = (i + 1) + 7
        tempSF = [sf for sf in sfSet if sf >= minSF]
        sfBuckets.extend(tempSF)

        return sfBuckets

    # This function generate a matrix for experts using Time on Air etc

    def updateProb(self, algo):
        """ Update the probability of each action by using EXP3 algorithm.
        Parameters
        ----------
       
        Returns
        -------
    
        """
        prob = [self.prob[x] for x in self.prob]

        weight = [self.weight[x] for x in self.weight]
        reward = np.zeros(self.nrActions)
        # print('prob:' + str(prob) + 'weight' + str(weight))
        # compute reward
        if self.node_mode == "SMART":
            # no and partial information case:
            if self.info_mode in ["NO", "PARTIAL"]:
                # with ACK -> 1, no ACK -> 0
                if self.ack:
                    # print(prob[self.packets[0].choosenAction])
                    # print(self.packets[0].sf)

                    reward[self.packets[0].choosenAction] = self.packets[0].PRR / prob[self.packets[0].choosenAction]
                    # print(reward)
                else:
                    self.ackLosts += 1
                    if self.algo == 'ADR' and (self.ackLosts % 3 == 0):  # If adr algo is
                        self.sf = np.minimum(12, self.sf + 1)
                    #if(self.nodeid == 300):
                        #print("Finally Lost")
                    reward[self.packets[0].choosenAction] = 0
            # full information case:
            else:
                if self.ack:
                    if not self.ack[0].isCollision:
                        reward[self.packets[0].choosenAction] = 1 / prob[self.packets[0].choosenAction]
                    else:
                        reward[self.packets[0].choosenAction] = 0.5 / prob[self.packets[0].choosenAction]
                else:
                    reward[self.packets[0].choosenAction] = 0

        #if self.nodeid == 300:
            #print(f"Reward Generated is: {reward}")
            #if not self.ack:
                #print("FInally ###############################################")
        # update weight
        for j in range(0, self.nrActions):
            if algo == "exp3":
                weight[j] *= np.exp((self.learning_rate * reward[j]) / self.nrActions)
            elif algo == "exp3s" or algo == 'exp4o':
                temp = weight[j]
                weight[j] *= np.exp((self.learning_rate * reward[j]) / self.nrActions)
                weight[j] += ((np.exp(1) * self.alpha) / self.nrActions) * sum(weight)

            elif algo == 'exp4':

                expertMat = self.fixedExperts()
                expertMat = np.array([expertMat, self.prob_x])
                if j == 0:  #Calcualte the weights in only one iteration
                    for i in range(self.nrExperts):
                        temp = np.sum(np.multiply(reward, expertMat[i]))
                        self.weight_e[i] *= np.exp((self.learning_rate2 * temp) / self.nrActions)

                self.weight_x[j] *= np.exp((self.learning_rate * reward[j]) / self.nrActions)
                self.weight_x[j] += ((np.exp(1) * self.alpha) / self.nrActions) * sum(self.weight_x)
                # print(f"probx : {self.prob_x} \n prob: {self.prob} \n weigth: {self.weight_e} \n weight_x {self.weight_x} \n reward {reward}")

        # update prob
        if self.node_mode == "SMART" and algo != 'exp4' and algo != 'ADR':
            for j in range(0, self.nrActions):
                prob[j] = (1 - self.learning_rate) * (weight[j] / sum(weight)) + (self.learning_rate / self.nrActions)
        elif self.node_mode == "RANDOM":
            prob = np.random.rand(self.nrActions)
            prob = prob / sum(prob)
        elif self.node_mode == 'SMART' and algo == 'exp4':
            for j in range(self.nrActions):
                W_t = np.sum(self.weight_e)
                expertMat = self.fixedExperts()

                expertMat = np.array([expertMat, self.prob_x])

                temp = np.zeros((4, 1))

                for i in range(self.nrExperts):
                    temp[i] = self.weight_e[i] * expertMat[i][j]

                temp_sum = np.sum(temp)
                self.prob_x[j] = (1 - self.learning_rate) * (self.weight_x[j] / sum(self.weight_x)) + (
                        self.learning_rate / self.nrActions)
                prob[j] = (1 - self.learning_rate2) * temp_sum / W_t + self.learning_rate2 / self.nrActions
                if self.nodeid == 300 and j == 5:
                    #np.set_printoptions(precision=3)
                    x = np.array(list(prob))
                    #print(f"Final Probability vector: {x}")

        elif algo == "ADR":
            if not self.ack:  # If there is no ack
                self.sf = max(12, self.sf + 1)  # Decrease the data rate as packet has been lost

        else:
            prob = (1 / self.nrActions) * np.ones(self.nrActions)

        # trick: force the small value (<1/5000) to 0 and normalize
        prob = np.array(prob)
        prob[prob < 0.0005] = 0
        prob = prob / sum(prob)
        self.prob_x = self.prob_x / sum(self.prob_x)
        self.weight = {x: weight[x] for x in range(0, self.nrActions)}
        self.prob = {x: prob[x] for x in range(0, self.nrActions)}

    def resetACK(self):
        """Reset ACK"""
        self.ack = {}

    def addACK(self, bsid, packet):
        """Send an ACK to the node"""
        self.ack[bsid] = packet

    def updateTXSettings(self):
        """Update TX setting"""
        pass
