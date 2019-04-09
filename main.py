from tkinter import *
from module import Module
import matplotlib.pyplot as plt

class GUI:
    def __init__(self):
        self.root = Tk()
        self.module1 = Module()
        self.module1.FillData()
        self.NumHiddenLayer = StringVar()
        self.ListNumNeurons = StringVar()
        self.learning = StringVar()
        self.epoc = StringVar()
        self.bais = StringVar()
        self.listN = []
        self.sigmoid = StringVar()
        self.Hyper = StringVar()

    def BulidPage(self):
        self.root.geometry("300x200")
        Label(self.root, text='Number Hidden Layer').grid(row=0)
        Label(self.root, text='List Number Neurons').grid(row=1)
        e1 = Entry(self.root, textvariable=self.NumHiddenLayer)
        e2 = Entry(self.root, textvariable=self.ListNumNeurons)
        e1.grid(row=0, column=2)
        e2.grid(row=1, column=2)

        Label(self.root, text='Learning Rate').grid(row=5)
        Label(self.root, text='epoc').grid(row=6)
        Label(self.root, text='bais').grid(row=7)
        Label(self.root, text='sigmoid').grid(row=8)
        Label(self.root, text='Hyperbolic').grid(row=9)
        e5 = Entry(self.root, textvariable=self.learning)
        e6 = Entry(self.root, textvariable=self.epoc)
        e7 = Entry(self.root, textvariable=self.bais)
        e8 = Entry(self.root, textvariable=self.sigmoid)
        e9 = Entry(self.root, textvariable=self.Hyper)
        e5.grid(row=5, column=2)
        e6.grid(row=6, column=2)
        e7.grid(row=7, column=2)
        e8.grid(row=8, column=2)
        e9.grid(row=9, column=2)

        B = Button(self.root, text="DONE", command=self.ClickButton)
        B.place(x=150, y=160)
        mainloop()

    def ClickButton(self):
        self.train()
        self.test()

    def train(self):
        self.listN = []
        NHL = int(self.NumHiddenLayer.get())
        LNN = str(self.ListNumNeurons.get())
        ListNeurons = LNN.split()
        self.listN = list(map(int, ListNeurons))
        learning = float(str(self.learning.get()))
        epoc = int(self.epoc.get())
        bais = int(self.bais.get())
        sigmoid = int(self.sigmoid.get())
        Hyper = int(self.Hyper.get())
        self.module1.BackPropagation(NHL, self.listN, learning, epoc, bais, sigmoid, Hyper)

    def test(self):
        NHL = int(self.NumHiddenLayer.get())
        bais = int(str(self.bais.get()))
        sigmoid = int(self.sigmoid.get())
        Hyper = int(self.Hyper.get())
        self.module1.Test(NHL, self.listN, bais, sigmoid, Hyper)
        #print(self.listN)

G = GUI()
G.BulidPage()

