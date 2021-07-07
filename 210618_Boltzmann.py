#import the libraries

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

#importing dataset
movies =  pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
users = pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')

#structure of 'ratings'
#first column (index 0) corresponds to user
#second column (index 1) corresponds to movies
#third column (index 2) corresponds to ratings (5: really like)
#fourth column (index 3) corresponds to time stamps

#prepare training and test set
#in ml-100k...
#... base means training set and test means test set

#training_set has 80T ratings (80% of original dataset)
#training_set has same column setup like 'ratings'
training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t')
#convert dataframe to array
#'int' means the array will only consist of integers
training_set = np.array(training_set, dtype='int')

#test_set has 20000 entries
test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t')
test_set = np.array(test_set, dtype='int')

#get total number of users and movies
#later we prepare 2 matrices
#one for training and one for test set
#matrices will contain user-lines and movie-columns
#in the cells, there will be the ratings

#0 is the column of the users
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
#1 is the index of the movie column
nb_movies =  int(max(max(training_set[:,1]), max(test_set[:,1])))

#convert data to array with user-lines and movie-columns
def convert(data):
    #create a list of lists (each user has one list of movie-ratings)
    new_data = []
    for id_users in range(1, nb_users + 1):
        #taking all movie id's of one user
        #1 stands for the movie id's
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        #create a list of zeros (if user rated a movie, 0 should be replaced by rating)
        ratings = np.zeros(nb_movies)
        #in first [], we take the indexes of the movies that where rated 
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

#apply function to training and test set
training_set = convert(training_set)
test_set = convert(test_set)

#convert data into torch tensors
#tensors are array with elements of single datatype
#could be understood as pytorch array
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

#starting with Restricted Boltzmann Machines
#convert ratings into binary ones 1 (like), 0 (dislike)

#replace all original 0's by -1
training_set[training_set == 0] = -1

#convert original ratings into 0 (dislike)
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0

#convert original ratings into 1 (like)
training_set[training_set >= 3] = 1

test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

#create architecture of Neural Network
#firstly build classes (instructions)
#out of these classes, we create objects (they are the networks)

#__init__ function contains the parameters of the object

class RBM():
    #nv is number of visible nodes
    #nh is number of hidden nodes
    def __init__(self, nv, nh):
        #initialize parameters of future objects, created from this class
        #first parameter: weights
        #creation of matrix with W as values, that has size nh, nv
        self.W = torch.randn(nh, nv)
        #second parameter: bias
        #meaning of 1: creation of 2d tensor
        #first dimension corresponds to the batch
        #second dimension corresponds to bias
        #'a' is bias for the hidden nodes
        self.a = torch.randn(1, nh)
        #'b' is bias for the visible nodes
        self.b = torch.randn(1, nv)
        
    #sampling of the hidden (h) nodes
    #we use self to access W, a, b
    #x corresponds to visible neurons v in the probabilities ph given v
    def sample_h(self, x):
        #compute probability of h given v
        #it's probability that hidden neuron = 1, given values of visible neurons
        #'mm' makes to product of 2 tensors
        #we compute product of weights and x
        #t() means transpose
        wx = torch.mm(x, self.W.t())
        #compute sigmoid activation 
        #with expand_as, we ensure that bias are applied to each line of minibatch
        #activation function returns probability, that hidden node will be activated according to value of visible node 
        activation = wx + self.a.expand_as(wx)
        #meaning of 'p_h_given_v': prob that hidden node is activated
        #...given value of visible node
        p_h_given_v = torch.sigmoid(activation)
        #p_h_given_v is vector of n h (hidden nodes) elements
        #n = 100; each element has a probability
        #goal of bernoulli sampling: create vector of 0's and 1's 
        #0: hidden node not activated
        #1: hidden node activated
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    #sampling of the visible (v) nodes
    #y corresponds to hidden nodes
    def sample_v(self, y):
        #compute probability of v given h
        #it's probability that visible neuron = 1, given values of hidden neurons
        #'mm' makes to product of 2 tensors
        #we compute product of weights and y
        #no transpose t() required here
        wy = torch.mm(y, self.W)
        #compute sigmoid activation 
        #with expand_as, we ensure that bias are applied to each line of minibatch
        #activation function returns probability, that visible node will be activated according to value of hidden node
        #'b' is bias of visible nodes
        activation = wy + self.b.expand_as(wy)
        #meaning of 'p_v_given_h': prob that visible node is activated, depending on hidden nodes
        p_v_given_h = torch.sigmoid(activation)
        #p_v_given_h is vector of n v (visible nodes) elements
        #we have 1682 movies (elements in final vector)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    #function for contrastive divergence
    #goal is to maximize log-likelihood of training set
    #and to minimize energy state of the model 
    #in order to reach the goal, we have to compute the gradient
    #but directly compute gradients is too heavy
    #easier way: approximate gradient
    #we use contrastive divergence to approximate the gradient
    #here we also apply Gibbs sampling
    
    #v0: input vector (all user ratings)
    #vk: visible nodes obtained after k samplings
    #ph0: vector of probs, that at first iteration, hidden nodes = 1, given values of v0  
    #phk: probs of hidden nodes after k sampling given values of visible nodes vk
    def train(self, v0, vk, ph0, phk):
        #update of weights
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        #update of bias;'0' is there to keep the tensor with 2 dimension
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)
        
#create RBM object with params nv and nh
#nv refers to number of movies; nh: number of hidden nodes
nv = len(training_set[0])
nh = 100
#batch size (weights updated after several observations)
#batch refers to same number of observations, after that the weights are updated
batch_size = 100

#rbm is the object; RBM is the class
rbm = RBM(nv, nh)

#train the RBM
#choose number of epochs
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    #loss function to meausure the error
    train_loss = 0
    #'s' is a counter (the . means it should be a float)
    s = 0.
    #get the user batches
    #last batch_size means we are always taking 100 users in one batch
    for id_user in range(0, nb_users - batch_size, batch_size):
        #get the input 
        #at the start, vk is the input batch of the observations
        vk = training_set[id_user:id_user+batch_size]
        #get the target (we don't change it, just want to compare it with predicted values)
        #v0 are the movies that are already rated in batch
        v0 = training_set[id_user:id_user+batch_size]
        #take initial probs (ph0: probs at the start that hidden nodes = 1 given real ratings)
        #,_ means that we only want to take first result (element) of the function, that should be p_h_given_v
        ph0,_ = rbm.sample_h(v0)
        
        #do a for loop for the k steps of contrastive divergence
        for k in range(10):
            #call sample_h on the visible nodes
            #hk means hidden nodes at k'th step of contrastive divergence
            #_,hk means we only want to take second result (element) of the function
            _,hk = rbm.sample_h(vk)
            #update vk --> call sample_v on first sample of hidden nodes
            _,vk = rbm.sample_v(hk)
            #don't include cells with -1 rating (no original ratings)
            vk[v0<0] = v0[v0<0]
        
        #compute phk (required for train function)
        #by taking first element of sample_h function
        phk,_ = rbm.sample_h(vk)
        
        #call train funtcion
        rbm.train(v0, vk, ph0, phk)
        
        #measure the loss
        #abs means absolute difference
        #we are interested in difference between target and predictions
        #[v0>0]: use the ratings that are existent 
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        
        #update the counter
        #counter has to normalize train loss
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

#make predictions on the test set (new observations)

#loss function to meausure the error
test_loss = 0
#'s' is a counter (the . means it should be a float)
s = 0.

#looping over all users
for id_user in range(nb_users):
    #get the input, on which we will make the prediction
    #by using input of the training set, we will activate neurons of RBM to predict ratings that are still missing
    v = training_set[id_user:id_user+1]
    #vt is our target (it contains the original ratings)
    vt = test_set[id_user:id_user+1]
    
    #filter out non existing ratings
    if len(vt[vt>=0]) > 0:
        #make one prediction step    
        #call sample_h on the visible nodes
        #_,h means we only want to take second result (element) of the function
        _,h = rbm.sample_h(v)
        #update v --> call sample_v on first sample of hidden nodes
        _,v = rbm.sample_v(h)
   
        #measure the loss
        #abs means absolute difference
        #we are interested in difference between target and predictions
        #[v0>0]: use the ratings that are existent 
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
    
        #update the counter
        #counter has to normalize train loss
        s += 1.
print('test loss: '+str(test_loss/s))

        
        
#final goal:
#get the ratings (0, 1) for those movies, that have not been rated by user

#adjust the model and see if loss can mitigated