import pylab as pl
import numpy as np


class ANN:

    #Initlialize
    def __init__(self, inputs, targets, nhidden1 = 0, nhidden2 = 0, nlayers = 1):
        #use positive ones as opposed to book
        #only works for input that is more than two dimensional
        #includes bias in count for input only
        self.inputs = np.concatenate(np.ones(np.shape(inputs)[0], 1), inputs)
        self.targets = targets
        self.feature_size = np.shape(self.inputs)[1]
        self.output_size = np.shape(self.targets)[1]
        self.hidden_layer1_size = nhidden1
        self.hidden_layer2_size = nhidden2
        self.hidden_layer_count = nlayer

        if self.hidden_layer_count == 0:
            weights1 = (np.random.rand(self.feature_size,self.output_size)-0.5)*2/np.sqrt(self.feature_size)
            weights2 = []
            weights3 = []
        elif self.hidden_layer_count == 1:
            weights1 = (np.random.rand(self.feature_size,self.hidden_layer1_size)-0.5)*2/np.sqrt(self.feature_size)
            weights2 = (np.random.rand(self.hidden_layer1_size+1,self.output_size)-0.5)*2/np.sqrt(self.hidden_layer1_size)
            weights3 = []
        elif self.hidden_layer_count == 2:
            weights1 = (np.random.rand(self.feature_size,self.hidden_layer1_size)-0.5)*2/np.sqrt(self.feature_size)
            weights2 = (np.random.rand(self.hidden_layer1_size+1,self.hidden_layer2_size)-0.5)*2/np.sqrt(self.hidden_layer1_size)
            weights3 = (np.random.rand(self.hidden_layer2_size+1,self.output_size)-0.5)*2/np.sqrt(self.hidden_layer2_size)



    def train_n_iterations(self,iterations):
        print "nothing yet"

