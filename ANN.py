import pylab as pl
import numpy as np


class ANN:

    #Initlialize
    def __init__(self, inputs, targets, nhidden1 = 0, nhidden2 = 0, nlayers = 1, momentum = 0, beta = 1):
        #use positive ones as opposed to book
        #only works for input that is more than two dimensional
        #includes bias in count for input only
        self.inputs = np.concatenate((np.ones((np.shape(inputs)[0], 1)), inputs),axis=1)
        self.targets = targets
        self.feature_size = np.shape(self.inputs)[1]
        self.output_size = np.shape(self.targets)[1]
        self.hidden_layer1_size = nhidden1
        self.hidden_layer2_size = nhidden2
        self.hidden_layer_count = nlayers
        self.momentum = momentum
        self.beta = beta

        if self.hidden_layer_count == 0:
            self.weights1 = (np.random.rand(self.feature_size,self.output_size)-0.5)*2/np.sqrt(self.feature_size)
            self.weights2 = []
            self.weights3 = []
        elif self.hidden_layer_count == 1:
            self.weights1 = (np.random.rand(self.feature_size,self.hidden_layer1_size)-0.5)*2/np.sqrt(self.feature_size)
            self.weights2 = (np.random.rand(self.hidden_layer1_size+1, self.output_size) - 0.5)* 2/np.sqrt(self.hidden_layer1_size)
            self.weights3 = []
        elif self.hidden_layer_count == 2:
            self.weights1 = (np.random.rand(self.feature_size, self.hidden_layer1_size)-0.5)*2/np.sqrt(self.feature_size)
            self.weights2 = (np.random.rand(self.hidden_layer1_size+1, self.hidden_layer2_size)-0.5)*2/np.sqrt(self.hidden_layer1_size)
            self.weights3 = (np.random.rand(self.hidden_layer2_size+1, self.output_size)-0.5)*2/np.sqrt(self.hidden_layer2_size)
        self.updatew1 = np.zeros((np.shape(self.weights1)))
        self.updatew2 = np.zeros((np.shape(self.weights2)))
        self.updatew3 = np.zeros((np.shape(self.weights3))) 

        self.train = []
        self.traint = []
        self.valid = []
        self.validt = []
        self.test = []
        self.testt = []

    def split_50_25_25(self):
        self.train = self.inputs[::2, :]
        self.traint = self.targets[::2]
        self.valid = self.inputs[1::4, :]
        self.validt = self.targets[1::4]
        self.test = self.inputs[3::4, :]
        self.testt = self.targets[3::4]

#make more efficient by removing redundant
    def forward_pass(self):
        if self.hidden_layer_count == 0:
            self.outputs = np.dot(self.train, self.weights1)         
        elif self.hidden_layer_count == 1:
            self.hidden1 = np.dot(self.train, self.weights1)
            self.hidden1 = np.concatenate((np.ones((np.shape(self.hidden1)[0],1)),self.hidden1),axis=1)
            self.outputs = np.dot(self.hidden1, self.weights2)
        elif self.hidden_layer_count == 2:
            self.hidden1 = np.dot(self.train, self.weights1)
            self.hidden1 = np.concatenate((np.ones((np.shape(self.hidden1)[0],1)),self.hidden1),axis=1)
            self.hidden2 = np.dot(self.hidden1, self.weights2)
            self.hidden2 = concatenate((np.ones((np.shape(self.hidden2)[0],1)),self.hidden2),axis=1)
            self.outputs = np.dot(self.hidden2,self.weights3)
        return 1.0/(1.0+np.exp(-self.beta*self.outputs))

    def train_n_iterations(self, iterations, learning_rate):

        #add case for no splitting
        if self.train == []:
            self.train = self.inputs

        for i in range(iterataions):
            self.outputs = self.forward_pass()
            deltao = self.beta*(self.outputs-self.targets)*self.outputs*(1.0-self.outputs)
            if self.hidden_layer_count == 1:
                deltah1 = self.hidden1*self.beta*(1.0-self.hidden1)*(np.dot(deltao,np.transpose(self.weights2)))
            elif self.hidden_layer_count == 2:
                deltah2 = self.hidden2*self.beta*(1.0-self.hidden2)*(np.dot(deltao,np.transpose(self.weights3)))
                deltah1 = self.hidden1*self.beta*(1.0-self.hidden1)*(np.dot(deltah2,np.transpose(self.weights2)))

            if self.hidden_layer_count == 0:
                self.updatew1 = learning_rate*(np.dot(np.transpose(self.inputs),deltao)) + self.momentum*self.updatew1
                self.weights1 -= self.updatew1
            if self.hidden_layer_count == 1:
                self.updatew1 = learning_rate*(np.dot(np.transpose(self.inputs),deltah1[:,1:])) + self.momentum*self.updatew1
                self.updatew2 = learning_rate*(np.dot(np.transpose(self.hidden1),deltao)) + self.momentum*self.updatew2
                self.weights1 -= self.updatew1
                self.weights2 -= self.updatew2
            elif self.hidden_layer_count == 2:
                self.updatew1 = learning_rate*(np.dot(np.transpose(self.inputs),deltah1[:,1:])) + self.momentum*self.updatew1
                self.updatew2 = learning_rate*(np.dot(np.transpose(self.hidden1),deltah2[:,1:])) + self.momentum*self.updatew2
                self.updatew3 = learning_rate*(np.dot(np.transpose(self.hidden2),deltao)) + self.momentum*self.updatew3
                self.weights1 -= self.updatew1
                self.weights2 -= self.updatew2
                self.weights3 -= self.updatew3



