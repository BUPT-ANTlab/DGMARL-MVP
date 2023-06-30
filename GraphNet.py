import torch
 # 之前是1.7.1
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from settings import params
from torch_geometric.utils import dense_to_sparse
#
# # 输入的是不同道路上三类车的数量[]
class topoGraph(nn.Module):
    def __init__(self, params):
        super(topoGraph, self).__init__()
        self.num_edge = params["num_edge"]
        self.dim_feature = params["topo_feature"]
        # self.fc1 = nn.Linear(self.dim_feature, 16)
        # self.fc2 = nn.Linear(16, 32)
        # self.fc3 = nn.Linear(32, self.num_edge)
        self.fc1 = nn.Linear(self.dim_feature, self.num_edge)
        self.GraphConv = GCNConv(self.num_edge, self.num_edge)


    def forward(self, x, adj):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        adj_sparse, _ = dense_to_sparse(adj[0][0])
        x_graph = self.GraphConv(x, adj_sparse)
        return x_graph  # (1,1,48,48)

# 一个智能体和所有的逃避车辆
class agentGraph(nn.Module):
    def __init__(self, params):
        super(agentGraph, self).__init__()
        self.num_pos = params["num_pos"]
        self.fc1 = nn.Linear(self.num_pos, 16)
        self.fc2 = nn.Linear(16, self.num_pos)
        self.num_evader = params["num_evader"]
        self.GraphConv = GCNConv(self.num_pos, self.num_evader+1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    def forward(self, node_feature, topo_output):
        # node_feature （1,6,7）
        # topo_output （1,1,48,48）
        # node_feature要包含什么，只有自己的位置？
        adj = self.getAdj(node_feature, topo_output)
        # node_feature = F.relu(self.fc1(node_feature))
        # node_feature = F.relu(self.fc2(node_feature))
        adj_sparse, _ = dense_to_sparse(adj)
        agent_graph = self.GraphConv(node_feature, adj_sparse)
        return agent_graph  # （1,6,5）


    def getAdj(self, node_feature, topo_output):
        adj = [[0]*(params["num_evader"]+1) for _ in range(params["num_evader"]+1)]
        topo_output = topo_output[0][0]
        laneIndexList = self.bin2Index(node_feature)
        for i in range(len(laneIndexList)):
            for j in range(len(laneIndexList)):
                if i == j:
                    adj[i][j] = 1
                else:
                    if laneIndexList[i] != -1 and laneIndexList[j] != -1 and topo_output[laneIndexList[i]][laneIndexList[j]] >= 0:
                        adj[i][j] = 1
        return torch.tensor(adj, dtype=torch.float, device=self.device)

    def bin2Index(self, node_feature):
        index = [-1]*len(node_feature[0])
        for i in range(len(node_feature[0])):
            bin_code = list(map(str, map(int, node_feature[0][i].detach().cpu().numpy().tolist())))
            bin_code = ''.join(bin_code)
            if bin_code[0] != '-':
                index[i] = int(bin_code[:-1], 2)
        return index


class test(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.num_edge = params["num_edge"]
        # self.dim_feature = params["topo_feature"]
        self.fc1 = nn.Linear(3, 16)
        # self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(16, 3)
        self.GraphConv = GCNConv(3, 3)


    def forward(self, x, adj):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        print(x)
        adj_sparse, _ = dense_to_sparse(adj)
        print(adj_sparse)
        x_graph = self.GraphConv(x, adj_sparse)
        return x_graph


if __name__ == '__main__':
    GraphNet = test(params)
    x = [[0.0,1.0,2.0],
         [2.0,4.0,3.0],
         [1.0,3.0,7.0]]
    x = torch.Tensor(x)
    # adj存储的每条边的起始点和终止点 2*E
    adj = [[0,1,0],
           [1,0,1],
           [1,1,1]]
    adj = torch.Tensor(adj)
    y = GraphNet(x, adj)







