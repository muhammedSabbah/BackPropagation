from ReadFromFile import ReadFile
from FlowerClass import Flower
import numpy as np
import math

class Module:
    def __init__(self):
        self.R = ReadFile()
        self.F1 = Flower()
        self.F2 = Flower()
        self.F3 = Flower()
        self.Dictionary = {"1": self.F1, "2": self.F2, "3": self.F3}
        self.ListWeight = []
        self.ListOfF = []
        self.ListOfError = []

    def FillData(self):
        self.R.Read()
        self.F1.FillData(Lines=self.R.Lines, numberFlower=1)
        self.F2.FillData(Lines=self.R.Lines, numberFlower=2)
        self.F3.FillData(Lines=self.R.Lines, numberFlower=3)

    def bulidListWeights(self, NumHiddenLayer, ListNumNeurons):
        ListNumNeurons.insert(0, 4)
        ListNumNeurons.append(3)
        counter = 0
        while counter <= NumHiddenLayer:
            w1 = np.random.rand(ListNumNeurons[counter + 1], ListNumNeurons[counter] + 1)
            self.ListWeight.append(w1)
            counter = counter + 1

    def Forward(self, NumHiddenLayer, ListNumNeurons, classNumber, index, bais, sigmoid, Hyper):
        x1 = float(self.Dictionary[classNumber].DictionaryFlowerDataX["x1"][index])
        x2 = float(self.Dictionary[classNumber].DictionaryFlowerDataX["x2"][index])
        x3 = float(self.Dictionary[classNumber].DictionaryFlowerDataX["x3"][index])
        x4 = float(self.Dictionary[classNumber].DictionaryFlowerDataX["x4"][index])
        self.ListOfF.append([x1, x2, x3, x4])
        counter = 0
        while counter <= NumHiddenLayer:
            HiddenLayer = []
            innerCounter = 0
            self.ListOfF[counter].insert(0, bais)
            while innerCounter < ListNumNeurons[counter + 1]:
                xi = np.array(self.ListOfF[counter])
                wi = self.ListWeight[counter][innerCounter]
                yi = np.dot(wi, xi)
                if sigmoid == 1:
                    yi = self.sigmoid(1, yi)
                elif Hyper == 1:
                    yi = self.Hyper(1, yi)
                HiddenLayer.append(yi)
                innerCounter = innerCounter + 1
            self.ListOfF.append(HiddenLayer)
            self.ListOfF[counter].pop(0)
            counter = counter + 1

    def LastError(self, ListTarget):
        Error = []
        indx = len(self.ListOfF) - 1
        for i in range(3):
            Fnet = self.ListOfF[indx][i]
            E = (ListTarget[i] - Fnet) * Fnet * (1 - Fnet)
            Error.append(E)
        self.ListOfError.append(np.array(Error))

    def Backward(self):
        indxofF = len(self.ListOfF) - 2
        indxofE = 0
        while indxofF > 0:
            Error = []
            length = len(self.ListOfF[indxofF])
            for i in range(length):
                Fnet = self.ListOfF[indxofF][i]

                segma = np.array(self.ListOfError[indxofE])

                weight = np.array(self.GetWeights(indxofF, i + 1))

                yi = np.dot(segma, weight)
                result = yi * Fnet * (1 - Fnet)
                Error.append(result)
            self.ListOfError.append(Error)
            indxofE = indxofE + 1
            indxofF = indxofF - 1

    def Updateweight(self, learning, bais):
        indxError = len(self.ListOfError) - 1
        indxListWeight = 0
        while indxError >= 0:
            lengthListofErro = len(self.ListOfError[indxError])
            self.ListOfF[indxListWeight].insert(0, bais)
            for i in range(lengthListofErro):
                Error = self.ListOfError[indxError][i]
                w = self.ListWeight[indxListWeight][i]
                xi = np.array(self.ListOfF[indxListWeight])
                self.ListWeight[indxListWeight][i] = w  + ((learning * Error) * xi)
            indxListWeight = indxListWeight + 1
            indxError = indxError - 1

    def GetWeights(self, outerIndx, innerIndx):
        ListWeight = []
        for weights in self.ListWeight[outerIndx]:
            ListWeight.append(weights[innerIndx])
        return np.array(ListWeight)

    def sigmoid(self, a, vk):
        return 1 / (1 + math.exp(-1 * a * vk))

    def Hyper(self, a, vk):
        return (1 - math.exp(-1 * a * vk)) / (1 + math.exp(-1 * a * vk))

    def BackPropagation(self, NumHiddenLayer, ListNumNeurons, LearningRate, epoc, bais, sigmoid, Hyper):
        self.ListWeight = []
        self.bulidListWeights(NumHiddenLayer, ListNumNeurons)
        while epoc > 0:
            target = [1, 0, 0]
            for i in range(30):
                self.ListOfF = []
                self.ListOfError = []
                self.Forward(NumHiddenLayer, ListNumNeurons, "1", i, bais, sigmoid, Hyper)
                self.LastError(target)
                self.Backward()
                self.Updateweight(LearningRate, bais)
            target = [0, 1, 0]
            for i in range(30):
                self.ListOfF = []
                self.ListOfError = []
                self.Forward(NumHiddenLayer, ListNumNeurons, "2", i, bais, sigmoid, Hyper)
                self.LastError(target)
                self.Backward()
                self.Updateweight(LearningRate, bais)
            target = [0, 0, 1]
            for i in range(30):
                self.ListOfF = []
                self.ListOfError = []
                self.Forward(NumHiddenLayer, ListNumNeurons, "3", i, bais, sigmoid, Hyper)
                self.LastError(target)
                self.Backward()
                self.Updateweight(LearningRate, bais)
            epoc = epoc - 1

    def Test(self, NumHiddenLayer, ListNumNeurons, bais, sigmoid, Hyper):
        output = [[0 for j in range(3)] for i in range(3)]
        indx = 30
        while indx < 50:
            self.ListOfF = []
            self.Forward(NumHiddenLayer, ListNumNeurons, "1", indx, bais, sigmoid, Hyper)
            indxofF = len(self.ListOfF) - 1
            positionOfMax = self.ListOfF[indxofF].index(max(self.ListOfF[indxofF]))
            output[0][positionOfMax] = output[0][positionOfMax] + 1
            indx = indx + 1
        indx = 30
        while indx < 50:
            self.ListOfF = []
            self.Forward(NumHiddenLayer, ListNumNeurons, "2", indx, bais, sigmoid, Hyper)
            indxofF = len(self.ListOfF) - 1
            positionOfMax = self.ListOfF[indxofF].index(max(self.ListOfF[indxofF]))
            output[1][positionOfMax] = output[1][positionOfMax] + 1
            indx = indx + 1
        indx = 30
        while indx < 50:
            self.ListOfF = []
            self.Forward(NumHiddenLayer, ListNumNeurons, "3", indx, bais, sigmoid, Hyper)
            indxofF = len(self.ListOfF) - 1
            positionOfMax = self.ListOfF[indxofF].index(max(self.ListOfF[indxofF]))
            output[2][positionOfMax] = output[2][positionOfMax] + 1
            indx = indx + 1
        print(output[0])
        print(output[1])
        print(output[2])
        A = (output[0][0] + output[1][1] + output[2][2]) / 60 * 100
        print(A)


