import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
import numpy as np
from settings import params
from GraphNet import topoGraph, agentGraph
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DQN_net(nn.Module):
    def __init__(self,num_edge,num_pos,num_evader,num_action=3):
        super(DQN_net, self).__init__()

        self.num_edge=num_edge
        self.num_pos=num_pos
        self.num_evader=num_evader
        self.num_action=num_action

        self.fc_n1 = nn.Linear((self.num_evader+1)**2+1+self.num_pos+self.num_pos, 32)

        if "4x5Traffic" in params["env_name"]:
            self.conv1_link = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=6, stride=3, padding=1)
            self.conv2_link = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=3, padding=1)
            self.conv3_link = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=2, stride=1, padding=0)
            self.fc_link = nn.Linear(81, 24)
        elif "3x3Traffic" in params["env_name"]:
            self.conv1_link = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=2, padding=1)
            self.conv2_link = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=4, stride=3, padding=1)
            self.conv3_link = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=2, stride=1, padding=0)
            self.fc_link = nn.Linear(49, 24)
        elif "RealMap" in params["env_name"] :
            self.conv1_link = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=7, stride=3, padding=1)
            self.conv2_link = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=3, padding=1)
            self.conv3_link = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=0)
            self.fc_link = nn.Linear(81, 24)

        # self.multihead_attn = nn.MultiheadAttention(num_pos,1)

        self.fc_hid1 = nn.Linear(32+24, 48)
        self.fc_action = nn.Linear(48, num_action)
        self.topoGraph = topoGraph(params)
        self.agentGraph = agentGraph(params)



    # 不要attention和target pos
    def forward(self, steps, target_pos,ego_pos, topo_link_array, traffic_state, node_feature):
        topo_output = self.topoGraph(traffic_state, topo_link_array) # （1,1,48,48）
        graph_output = self.agentGraph(node_feature, topo_output)

        # print("*",steps.size())
        # print("**",torch.flatten(graph_output,1).size())
        # print("***",ego_pos.size())
        # print("****",target_pos.size())

        all_input = torch.cat((steps, torch.flatten(graph_output,1), ego_pos,target_pos),1)
        feature = F.elu(self.fc_n1(all_input))

        # print("****", topo_link_array.size())

        topo=F.relu(self.conv1_link(topo_link_array))
        topo = F.relu(self.conv2_link(topo))
        topo = F.relu(self.conv3_link(topo))

        if "3x3Traffic"in params["env_name"] :
            topo = F.elu(self.fc_link(topo.view(-1, 49)))
        else:
            topo = F.elu(self.fc_link(topo.view(-1, 81)))


        # atten_output, atten_weights = self.multihead_attn(torch.transpose(ego_pos.view(-1,1,self.num_pos),0,1), torch.transpose(all_evaders_pos,0,1), torch.transpose(all_evaders_pos,0,1))
        # atten_weights=torch.transpose(atten_weights,0,1).view(-1,self.num_evader)

        all_features = torch.cat((feature,topo),1)
        all_features = F.elu(self.fc_hid1(all_features))
        Q_values = F.softmax(self.fc_action(all_features), dim=1)
        return Q_values

