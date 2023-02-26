import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import copy

class Probe(torch.nn.Module):
    # model structure
    def __init__(self, input_size):
        super(Probe,self).__init__()
        self.fc1 = torch.nn.Linear(input_size,1)
        self.sigmoid = torch.nn.Sigmoid()
        self.input = input_size

    # forward input through model
    def forward(self,x):
        # x is input to network
        return(self.sigmoid(self.fc1(x)))

class CCS():

  def __init__(self, neg, pos, neg_test, pos_test, test_labels, num_tries = 10, lr = 0.01, num_epochs = 1000, batch_size = None, device="cuda"):
    
    # initialize model
    self.device = device
    self.num_examples = neg.shape[0]
    self.input_size = neg.shape[1]
    self.probe = Probe(self.input_size)
    self.probe.to(self.device)
    self.best_probe = copy.deepcopy(self.probe)
    self.best_loss = 9999
    
    # initialize model training parameters
    self.num_tries = num_tries
    self.lr = lr
    self.num_epochs = num_epochs
    if batch_size is not None:
      self.batch_size = batch_size
    else:
      self.batch_size = self.num_examples
    
    # normalize and set hidden states
    self.neg = self.normalize_data(neg)
    self.pos = self.normalize_data(pos)
    self.neg_test = self.normalize_data(neg_test)
    self.pos_test = self.normalize_data(pos_test)
    self.labels = test_labels

  def normalize_data(self, data):
    '''
    standardize data by mean and standard deviation.
    '''
    normalized_data = (data - data.mean(axis = 0, keepdims = True))
    final_data = normalized_data/normalized_data.std(axis=0, keepdims=True)
    return(final_data)
  
  def reset_probe(self):
    self.probe = Probe(self.input_size)
    self.probe.to(self.device)
  
  def train_probe(self):
    '''
    trains an individual probe given CCS parameters.
    '''
    self.neg = self.neg.to(self.device)
    self.pos = self.pos.to(self.device)

    # define optimizer and data loading
    optimizer = torch.optim.AdamW(self.probe.parameters(), lr = self.lr)
    shuffler = torch.randperm(self.num_examples)
    neg_shuffled = self.neg[shuffler]
    pos_shuffled = self.pos[shuffler]
    self.num_batches = self.num_examples // self.batch_size

    for epoch in range(self.num_epochs):
      for batch in range(self.num_batches):
        # pass neg, pos through probe, compute loss
        neg_batch = neg_shuffled[batch*self.batch_size: (batch+1)*self.batch_size]
        pos_batch = pos_shuffled[batch*self.batch_size: (batch+1)*self.batch_size]
        neg_out = self.probe(neg_batch)
        pos_out = self.probe(pos_batch)
        loss = self.compute_loss(neg_out, pos_out)

        # backpropagate through probe
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return(loss.detach().cpu().item())
      
  def train(self):
    '''
    trains a number of probes and saves the one with the best train loss.
    '''
    for i in tqdm(range(self.num_tries)):
      # train and evaluate a new probe
      loss = self.train_probe()

      # save probe if it improves on unsupervised loss
      if loss < self.best_loss:
        self.best_loss = loss
        self.best_probe = copy.deepcopy(self.probe)
      self.reset_probe()
    
    self.probe = self.best_probe
  def compute_loss(self, neg_prob, pos_prob):
    '''
    compute probe loss given predictions for positive, negative labels
    '''
    consistency_loss = torch.mean(((pos_prob - (1-neg_prob))**2), dim=0)
    confidence_loss = torch.mean((torch.min(pos_prob, neg_prob)**2), dim=0)
    return(consistency_loss + confidence_loss)

  def compute_accuracy(self, neg_prob, pos_prob, labels):
    '''
    compute accuracy given positive and negative labels.
    due to loss constraints, we must use their average to do this.
    '''
    conf = 0.5*(pos_prob + (1 - neg_prob))
    pred = conf > 0.5
    comp = (pred == labels).float()
    acc = torch.mean(comp).detach().cpu().item()
    return(max(acc, 1-acc)) # labels could be flipped
  
  def evaluate(self, neg_test, pos_test, labels, probe=None):
    '''
    evaluate model (using accuracy) over a test set
    '''
    if probe == None:
      probe = self.best_probe
    neg_test = self.normalize_data(neg_test)
    neg_test = neg_test.to(self.device)
    pos_test = self.normalize_data(pos_test)
    pos_test = pos_test.to(self.device)
    labels = labels.to(self.device)

    neg_prob = probe(neg_test)
    pos_prob = probe(pos_test)

    return(self.compute_accuracy(neg_prob, pos_prob, labels))

if __name__ == "__main__":
    neg_train = torch.Tensor(np.load('data/neg_hidden_states_train.npy'))
    pos_train = torch.Tensor(np.load('data/pos_hidden_states_train.npy'))
    neg_test = torch.Tensor(np.load('data/neg_hidden_states_test.npy'))
    pos_test = torch.Tensor(np.load('data/pos_hidden_states_test.npy'))
    labels = torch.Tensor(np.load('data/labels.npy'))

    data_size = neg_train.shape[0]
    train_split = 0.6
    test_labels = labels[int(data_size * train_split):] 
    
    if (torch.cuda.is_available()):
        device = "cuda:0"
    else:
        device = "cpu"

    CCS_test = CCS(neg_train, pos_train, neg_test, pos_test, test_labels, num_tries = 10, num_epochs = 1000, lr = 0.01, device=device)
    CCS_test.train()
    print('Eval Acc: ' + str(CCS_test.evaluate(neg_test, pos_test, test_labels)))