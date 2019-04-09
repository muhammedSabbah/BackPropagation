import numpy as np

class ReadFile:
    def __init__(self):
        self.txtFile = "IrisData.txt"
        self.Lines = np.empty([1, 150])

    def Read(self):
        with open(self.txtFile, 'r') as F:
            self.Lines = F.readlines()
