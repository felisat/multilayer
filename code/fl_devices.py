import os
import random, re
import torch
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_op(model, loader, optimizer, epochs=1):
    model.train()  
    for ep in range(epochs):
        running_loss, samples = 0.0, 0
        for x, y in loader:   
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            loss = torch.nn.CrossEntropyLoss()(model(x), y)
            running_loss += loss.item()*y.shape[0]
            samples += y.shape[0]

            loss.backward()
            optimizer.step()  
    return {"loss" : running_loss / samples}
      

def eval_op(model, loader):
    model.train()
    samples, correct = 0, 0

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            y_ = model(x)
            _, predicted = torch.max(y_.data, 1)
            
            samples += y.shape[0]
            correct += (predicted == y).sum().item()

    return {"accuracy" : correct/samples}

def flatten(source):
    return torch.cat([value.flatten() for value in source.values()])

def copy(target, source):
    for name in target:
        target[name].data = source[name].data.clone()
      
def subtract(target, minuend, subtrahend):
    for name in target:
        target[name].data = minuend[name].data.clone()-subtrahend[name].data.clone()
    
def compress(target, compression_fn):
    for name in target:
        target[name].data = compression_fn(target[name].data.clone())
    
def compress_and_accumulate(target, residual, compression_fn):
    for name in target:
        residual[name].data += target[name].data.clone()
        target[name].data = compression_fn(residual[name].data.clone())
        residual[name].data -= target[name].data.clone()
    
def reduce_add_average(target, sources):
    for name in target:
        tmp = torch.mean(torch.stack([source[name].data for source in sources]), dim=0).clone()
        target[name].data += tmp


def pairwise_angles(sources):
    angles = torch.zeros([len(sources), len(sources)])
    for i, source1 in enumerate(sources):
        for j, source2 in enumerate(sources):
            s1 = flatten(source1)
            s2 = flatten(source2)
            angles[i,j] = torch.sum(s1*s2)/(torch.norm(s1)*torch.norm(s2)+1e-8)

    return angles.numpy()

class FederatedTrainingDevice(object):
    def __init__(self, model_fn, data, batch_size, layers, train_frac=0.8):
        self.model = model_fn().to(device)
        self.data = data
        n_train = int(len(data)*train_frac)
        n_eval = len(data) - n_train 
        data_train, data_eval = torch.utils.data.random_split(self.data, [n_train, n_eval])

        self.loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
        self.eval_loader = DataLoader(data_eval, batch_size=batch_size, shuffle=True)

        self.W = {key : value for key, value in self.model.named_parameters() if re.match(layers, key)}


    def evaluate(self, loader=None):
        return eval_op(self.model, self.eval_loader if not loader else loader)
  
  
class Client(FederatedTrainingDevice):
    def __init__(self, model_fn, optimizer_fn, data, batch_size, layers, idnum):
        super().__init__(model_fn, data, batch_size, layers)  
        self.optimizer = optimizer_fn(self.model.parameters())
        self.id = idnum
        self.W_old = {key : torch.zeros_like(value) for key, value in self.W.items()}
        self.dW = {key : torch.zeros_like(value) for key, value in self.W.items()}
        self.R = {key : torch.zeros_like(value) for key, value in self.W.items()}
    
    def synchronize_with_server(self, server):
        copy(target=self.W, source=server.W)

    def compute_weight_update(self, epochs=1, loader=None):
        copy(target=self.W_old, source=self.W)
        train_stats = train_op(self.model, self.loader if not loader else loader, self.optimizer, epochs)
        subtract(target=self.dW, minuend=self.W, subtrahend=self.W_old)
        return train_stats   

    def compress_weight_update(self, compression_fn=None, accumulate=False):
        if compression_fn:
            if accumulate:
                compress_and_accumulate(self.dW, self.R, compression_fn)
            else:
                compress(self.dW, compression_fn)



class Server(FederatedTrainingDevice):
    def __init__(self, model_fn, data, layers):
        super().__init__(model_fn, data, layers=layers, batch_size=100)
    
    def select_clients(self, clients, frac=1.0):
        return random.sample(clients, int(len(clients)*frac)) 

    def aggregate_weight_updates(self, clients):
        reduce_add_average(target=self.W, sources=[client.dW for client in clients])


    def save_model(self, path=None, name=None, verbose=True):
        if name:
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(self.model.state_dict(), os.path.join(path,name))
            if verbose: print("Saved model to", os.path.join(path,name))

    def load_model(self, path=None, name=None, verbose=True):
        if name:
            self.model.load_state_dict(torch.load(os.path.join(path,name)))
            if verbose: print("Loaded model from", os.path.join(path,name))


    def compute_pairwise_angles_layerwise(self, clients):
        return {"sim_"+key : pairwise_angles([{key : client.dW[key]} for client in clients]) for key in clients[0].dW}