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
    def forward_pass(self, input_data='none'):
        if input_data == 'none':
            input_data = self.inputs
        self.hidden1 = []
        self.hidden2 = []  
        if self.hidden_layer_count == 0:
            self.outputs = np.dot(input_data, self.weights1)
        elif self.hidden_layer_count == 1:
            self.hidden1 = np.dot(input_data, self.weights1)            
            self.hidden1 = 1.0/(1.0+np.exp(-self.beta*self.hidden1))
            self.hidden1 = np.concatenate((np.ones((np.shape(self.hidden1)[0],1)),self.hidden1),axis=1)            
            self.outputs = np.dot(self.hidden1, self.weights2)
        elif self.hidden_layer_count == 2:
            self.hidden1 = np.dot(input_data, self.weights1)
            self.hidden1 = 1.0/(1.0+np.exp(-self.beta*self.hidden1))
            self.hidden1 = np.concatenate((np.ones((np.shape(self.hidden1)[0],1)),self.hidden1),axis=1)
            self.hidden2 = np.dot(self.hidden1, self.weights2)
            self.hidden2 = 1.0/(1.0+np.exp(-self.beta*self.hidden2))
            self.hidden2 = np.concatenate((np.ones((np.shape(self.hidden2)[0],1)),self.hidden2),axis=1)
            self.outputs = np.dot(self.hidden2,self.weights3)        
        return 1.0/(1.0+np.exp(-self.beta*self.outputs))

    def train_n_iterations(self, iterations, learning_rate, plot_errors = False):

        #add case for no splitting
        if self.train == []:
            self.train = self.inputs
            self.traint = self.targets

        if plot_errors == True:
            points = []

        for i in range(iterations):
            self.outputs = self.forward_pass(self.train)

            error = 0.5*np.sum((self.outputs-self.traint)**2)
            if (np.mod(i,100)==0):
                print "Iteration: ",i, " Error: ",error    
            deltao = self.beta*(self.outputs-self.traint)*self.outputs*(1.0-self.outputs)
            if self.hidden_layer_count == 0:
                self.updatew1 = learning_rate*(np.dot(np.transpose(self.train),deltao)) + self.momentum*self.updatew1
                self.weights1 -= self.updatew1
            if self.hidden_layer_count == 1:
                deltah1 = self.hidden1*self.beta*(1.0-self.hidden1)*(np.dot(deltao,np.transpose(self.weights2)))
                self.updatew1 = learning_rate*(np.dot(np.transpose(self.train),deltah1[:,1:])) + self.momentum*self.updatew1
                self.updatew2 = learning_rate*(np.dot(np.transpose(self.hidden1),deltao)) + self.momentum*self.updatew2
                self.weights1 -= self.updatew1
                self.weights2 -= self.updatew2
            elif self.hidden_layer_count == 2:
                deltah2 = self.hidden2*self.beta*(1.0-self.hidden2)*(np.dot(deltao,np.transpose(self.weights3)))
                deltah1 = self.hidden1*self.beta*(1.0-self.hidden1)*(np.dot(deltah2[:,1:],np.transpose(self.weights2)))
                self.updatew1 = learning_rate*(np.dot(np.transpose(self.train),deltah1[:,1:])) + self.momentum*self.updatew1
                self.updatew2 = learning_rate*(np.dot(np.transpose(self.hidden1),deltah2[:,1:])) + self.momentum*self.updatew2
                self.updatew3 = learning_rate*(np.dot(np.transpose(self.hidden2),deltao)) + self.momentum*self.updatew3
                self.weights1 -= self.updatew1
                self.weights2 -= self.updatew2
                self.weights3 -= self.updatew3            
            if plot_errors == True:
                points.append([100-self.confmat(inputs=self.train,targets=self.traint,print_info=False),100-self.confmat(inputs=self.valid,targets=self.validt,print_info=False)])
        if plot_errors == True:
            pl.plot(points)
            pl.show()
    def train_n_iterations_seq(self, iterations, learning_rate, plot_errors = False):

        #add case for no splitting
        if self.train == []:
            self.train = self.inputs

        if plot_errors == True:
            points = []

        for i in range(iterations):
            for j in range(np.shape(self.train)[0]):
                self.outputs = self.forward_pass(self.train[j,:]*np.ones((1,self.feature_size)))

                error = 0.5*np.sum((self.outputs-self.traint[j,:])**2)
                if (np.mod(i,100)==0):
                    print "Iteration: ",i, " Error: ",error    
                #use jth term
                deltao = self.beta*(self.outputs-self.traint[j])*self.outputs*(1.0-self.outputs)
                if self.hidden_layer_count == 0:
                    self.updatew1 = learning_rate*(np.dot(np.transpose(self.train),deltao)) + self.momentum*self.updatew1
                    self.weights1 -= self.updatew1
                if self.hidden_layer_count == 1:
                    #replace train with train[j]
                    deltah1 = self.hidden1*self.beta*(1.0-self.hidden1)*(np.dot(deltao,np.transpose(self.weights2)))
                    self.updatew1 = learning_rate*(np.dot(np.transpose((self.train[j,:]*np.ones((1,self.feature_size)))),deltah1[:,1:])) + self.momentum*self.updatew1
                    self.updatew2 = learning_rate*(np.dot(np.transpose(self.hidden1),deltao)) + self.momentum*self.updatew2
                    self.weights1 -= self.updatew1
                    self.weights2 -= self.updatew2
                elif self.hidden_layer_count == 2:
                    deltah2 = self.hidden2*self.beta*(1.0-self.hidden2)*(np.dot(deltao,np.transpose(self.weights3)))
                    deltah1 = self.hidden1*self.beta*(1.0-self.hidden1)*(np.dot(deltah2,np.transpose(self.weights2)))
                    self.updatew1 = learning_rate*(np.dot(np.transpose(self.train),deltah1[:,1:])) + self.momentum*self.updatew1
                    self.updatew2 = learning_rate*(np.dot(np.transpose(self.hidden1),deltah2[:,1:])) + self.momentum*self.updatew2
                    self.updatew3 = learning_rate*(np.dot(np.transpose(self.hidden2),deltao)) + self.momentum*self.updatew3
                    self.weights1 -= self.updatew1
                    self.weights2 -= self.updatew2
                    self.weights3 -= self.updatew3            
                if plot_errors == True:
                    points.append([100-self.confmat(inputs=self.train,targets=self.traint,print_info=False),100-self.confmat(inputs=self.valid,targets=self.validt,print_info=False)])
        if plot_errors == True:
            pl.plot(points)
            pl.show()
    #this code is almost directly copied from book with a fix of the axes
    def confmat(self,inputs='none',targets='none', print_info = True):
        if inputs == 'none':
            inputs=self.valid
        if targets == 'none':
            targets=self.validt
        nclasses = self.output_size        
        outputs = self.forward_pass(inputs)
        if nclasses==1:
            nclasses = 2
            outputs = np.where(outputs>0.5,1,0)
        else:
            # 1-of-N encoding
            outputs = np.argmax(outputs,1)
            targets = np.argmax(targets,1)
        cm = np.zeros((nclasses,nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i,j] = np.sum(np.where(targets==i,1,0)*np.where(outputs==j,1,0))

        if print_info == True:
            print "Confusion matrix is:"
            print cm
            print "Percentage Correct: ",np.trace(cm)/np.sum(cm)*100
        return np.trace(cm)/np.sum(cm)*100

