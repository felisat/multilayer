
import torch
import torch.nn.functional as F

class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 4 * 4, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 62)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class RoutingNet(torch.nn.Module):
    def __init__(self):
        super(RoutingNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 4 * 4, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 62)

        self.conv1_loc = torch.nn.Conv2d(1, 6, 5)
        self.conv2_loc = torch.nn.Conv2d(6, 16, 5)
        self.fc1_loc = torch.nn.Linear(16 * 4 * 4, 120)
        self.fc2_loc = torch.nn.Linear(120, 84)
        self.fc3_loc = torch.nn.Linear(84, 62)


        self.rounting_loc = torch.nn.Linear(28*28, 1)

    def forward(self, x):
        x_share = self.pool(F.relu(self.conv1(x)))
        x_share = self.pool(F.relu(self.conv2(x_share)))
        x_share = x_share.view(-1, 16 * 4 * 4)
        x_share = F.relu(self.fc1(x_share))
        x_share = F.relu(self.fc2(x_share))
        x_share = self.fc3(x_share)


        x_loc = self.pool(F.relu(self.conv1_loc(x)))
        x_loc = self.pool(F.relu(self.conv2_loc(x_loc)))
        x_loc = x_loc.view(-1, 16 * 4 * 4)
        x_loc = F.relu(self.fc1_loc(x_loc))
        x_loc = F.relu(self.fc2_loc(x_loc))
        x_loc = self.fc3_loc(x_loc)

        alpha = F.sigmoid(self.rounting_loc(x.view(-1, 28*28)))

        return alpha*x_share+(1-alpha)*x_loc



def get_model(model):

  return  { "ConvNet" : (ConvNet, lambda x : torch.optim.Adam(x, lr=0.001, weight_decay=0.0)),
            "RoutingNet" : (RoutingNet, lambda x : torch.optim.Adam(x, lr=0.001, weight_decay=0.0))
          }[model]


def print_model(model):
    n = 0
    print("Model:")
    for key, value in model.named_parameters():
        print(' -', '{:30}'.format(key), list(value.shape))
        n += value.numel()
    print("Total number of Parameters: ", n) 
    print()