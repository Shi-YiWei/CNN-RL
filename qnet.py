import torch
import torch.nn as nn
import torch.nn.functional as F


class Qnet(torch.nn.Module):
    # 只有一层隐藏层的Q网络
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)



class ConvolutionalQnet(torch.nn.Module):
   #  加入卷积层的Q网络 
    def __init__(self, action_dim, in_channels=3):
        super(ConvolutionalQnet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, 32, kernel_size=20, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=15, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 32, kernel_size=10, stride=1)
        self.fc4 = torch.nn.Linear(64672, 512)
        self.head = torch.nn.Linear(512, action_dim)
        # print("action_dim: in file ConvolutionalQnet:",action_dim)

    def forward(self, x):
        x = x / 255
        x = F.relu(self.conv1(x))
        #print("x = F.relu(self.conv1(x))",x.shape)


        x = F.relu(self.conv2(x))
        #print("x = F.relu(self.conv2(x)):",x.shape)


        x = F.relu(self.conv3(x))
        #print("x = F.relu(self.conv3(x)):",x.shape)
        

        x = x.view(-1, 64672)

        #print("x = x.view(-1, 392):",x.shape)

        x = F.relu(self.fc4(x))
        #print("x = F.relu(self.fc4(x)):",x.shape)


        x = self.head(x)
        #print("x = self.head(x):",x.shape)
        
        return x


'''





# the convolution layer of deepmind
class deepmind(nn.Module):
    def __init__(self):
        super(deepmind, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)
        
        # start to do the init...
        nn.init.orthogonal_(self.conv1.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv2.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv3.weight.data, gain=nn.init.calculate_gain('relu'))
        # init the bias...
        nn.init.constant_(self.conv1.bias.data, 0)
        nn.init.constant_(self.conv2.bias.data, 0)
        nn.init.constant_(self.conv3.bias.data, 0)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 392)

        return x

# in the initial, just the nature CNN
class ConvolutionalQnet(nn.Module):
    def __init__(self, num_actions, use_dueling=False):
        super(ConvolutionalQnet, self).__init__()
        # if use the dueling network
        self.use_dueling = use_dueling
        # define the network
        self.cnn_layer = deepmind()
        # if not use dueling
        if not self.use_dueling:
            self.fc1 = nn.Linear(392, 512)
            self.action_value = nn.Linear(512, num_actions)
        else:
            # the layer for dueling network architecture
            self.action_fc = nn.Linear(392, 512)
            self.state_value_fc = nn.Linear(392, 256)
            self.action_value = nn.Linear(512, num_actions)
            self.state_value = nn.Linear(512, 1)

    def forward(self, inputs):
        x = self.cnn_layer(inputs / 255.0)
        if not self.use_dueling:
            x = F.relu(self.fc1(x))
            action_value_out = self.action_value(x)
        else:
            # get the action value
            action_fc = F.relu(self.action_fc(x))
            action_value = self.action_value(action_fc)
            # get the state value
            state_value_fc = F.relu(self.state_value_fc(x))
            state_value = self.state_value(state_value_fc)
            # action value mean
            action_value_mean = torch.mean(action_value, dim=1, keepdim=True)
            action_value_center = action_value - action_value_mean
            # Q = V + A
            action_value_out = state_value + action_value_center
        return action_value_out
'''