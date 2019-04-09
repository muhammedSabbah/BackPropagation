import numpy as np

class Flower:
    def __init__(self):
        self.Lines = []
        self.x1 = []
        self.x2 = []
        self.x3 = []
        self.x4 = []
        self.Label = "NULL"
        self.DictionaryFlowerDataX = {"x1": self.x1, "x2": self.x2, "x3": self.x3, "x4": self.x4, "Label": self.Label}

    def FillData(self, Lines, numberFlower):
        self.Lines = Lines
        start = 1
        if numberFlower == 1 :
            start = 1
        elif numberFlower == 2 :
            start = 51
        else:
            start = 101
        self.Label = self.Lines[start].split(",")[4]
        self.DictionaryFlowerDataX["Label"] = self.Label
        end = start + 50
        while(start < end):
            self.x1.append(self.Lines[start].split(",")[0])
            self.x2.append(self.Lines[start].split(",")[1])
            self.x3.append(self.Lines[start].split(",")[2])
            self.x4.append(self.Lines[start].split(",")[3])
            start = start + 1

