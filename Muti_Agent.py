import numpy as np

from Agent_GDQN import GDQN
import torch
import os
import torch.optim as optim
from copy import deepcopy as dc
from env.utils import calculate_dis
from buffer import buffer ######buffer
from torch.distributions import Normal, Categorical
import torch.nn.functional as F

import collections
class Muti_agent():
    def __init__(self, params):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.test_steps_num=collections.deque(maxlen=10)

        self.params=params
        self.train_model = self.params["train_model"]
        
        ##创建buffer_dic
        self.replay_buffer_dict = {}
        for pursuer_id in self.params["pursuer_ids"]:
            self.replay_buffer_dict[pursuer_id] = buffer(params)

        # self.replay_buffer=buffer(params)
        self.agent = GDQN(self.params)



    def process_state(self,state):
        #return all_states dic for agent networks:{pursuer_id: {ego_pos: [ego_pos], target_pos:[target_pos],traffic_state: [traffic_state],topo_link_array: [topo_link_array],all_evaders_pos:[all_evaders_pos] } }
        #pro_state add"target"
        pro_state = dc(state)
        all_states={}

        if not ("target" in state):
            pursuer_target = {}
        else:
            pursuer_target=state["target"]

        all_evaders_pos = []
        for evader_id in self.params["evader_ids"]:
            all_evaders_pos.append(dc(state["evader_pos"][evader_id]))

        all_evaders_pos=[all_evaders_pos]
        traffic_state=[dc(state["traffic_state"])]
        topo_link_array=[dc(state["topology_array"])]

        for pursuer_id in self.params["pursuer_ids"]:
            ego_pos = [dc(state["pursuer_pos"][pursuer_id])]
            if "target" in state:
                target_pos=[dc(state["evader_pos"][state["target"][pursuer_id]])]
            else:
                # ego_pos_tensor = torch.tensor(ego_pos, dtype=torch.float,device=self.device)
                # all_evaders_pos_tensor = torch.tensor(all_evaders_pos, dtype=torch.float,device=self.device)
                # target_code = self.agent.DQN_Net.select_target(ego_pos_tensor,
                #                                                                   all_evaders_pos_tensor)[0]
                # if state["evader_pos"][self.params["evader_ids"][target_code]][0] == -1:
                target_id = self.min_dis_evader(state, pursuer_id)  #
                # else:
                #     target_id=self.params["evader_ids"][target_code]
                pursuer_target[pursuer_id]=target_id
                target_pos = [dc(state["evader_pos"][target_id])]
            ego_state={
                "ego_pos":ego_pos,
                "target_pos":target_pos,
                "traffic_state":traffic_state,
                "topo_link_array":topo_link_array,
                "all_evaders_pos":all_evaders_pos,
                "steps":[[dc(state["steps"])]]
            }
            all_states[pursuer_id]= ego_state
        pro_state["target"]=dc(pursuer_target)
        return pro_state,all_states


    def select_action(self,pro_state, pro_all_states):
        # pro_state, pro_all_states=self.process_state(state)
        actions={}
        actions_prob={}
        for pursuer_id in self.params["pursuer_ids"]:
            ego_state=pro_all_states[pursuer_id]
            action,action_prob=self.agent.select_action(ego_state)
            actions[pursuer_id]=action
            actions_prob[pursuer_id]=action_prob
        return actions,actions_prob,pro_state["target"]


    def min_dis_evader(self,state,pursuer_id):
        min_dis = float('inf')
        min_evader_id = None
        for eva_index, evader_id in enumerate(self.params["evader_ids"]):
            if state["evader_pos"][evader_id][0] !=-1:
                eva_x, eva_y = state["evader_xy"][evader_id]["x"], state["evader_xy"][evader_id]["y"]
                dis = calculate_dis(eva_x, eva_y,
                                          state["pursuer_xy"][pursuer_id]["x"], state["pursuer_xy"][pursuer_id]["y"])
                if dis <= min_dis:
                    min_dis = dis
                    min_evader_id = evader_id
        return min_evader_id
    
    def Cumul_R(self):
        r_dic={}
        R_dic={}
        for pursuer_id in self.params["pursuer_ids"]:
            r_list = []
            R_list = []
            R = 0 
            mp_id = dc(self.replay_buffer_dict[pursuer_id].memory_pool)
            for i in range(len(mp_id)):
                r_list.append(mp_id[i]["reward"])
            r_dic[pursuer_id] = r_list

            for r in r_list:
                R = R + r[0]
                R_list.insert(0,R)
                
            R_dic[pursuer_id] = R_list
        
        return R_dic

    def train_agents(self):
        print("prepare for training......")
        Cumul_Reward = self.Cumul_R()
        Qloss, Gloss, Dloss, Wd = [], [], [], []
        for pursuer_id in self.params["pursuer_ids"]:
            train_set = dc(self.replay_buffer_dict[pursuer_id].memory_pool)       
            Q_loss, G_loss, D_loss, W_d =self.agent.update(train_set,Cumul_Reward[pursuer_id])
            del self.replay_buffer_dict[pursuer_id].memory_pool[:]  #clear the current buffer
            Qloss.append(Q_loss)
            Gloss.append(G_loss)
            Dloss.append(D_loss)
            Wd.append(W_d)
            # if critic_loss<self.agent_list[pursuer_id].best_critic_loss:
            #     self.agent_list[pursuer_id].save_critic_param()
            #     self.agent_list[pursuer_id].best_critic_loss=critic_loss
        print("#####loss list: Q, G, D, W######", Qloss, Gloss, Dloss, Wd)
        return np.array(Qloss).mean(), np.array(Gloss).mean(), np.array(Dloss).mean(), np.array(Wd).mean()







    def load_params(self):
        # for pursuer_id in self.params["pursuer_ids"]:
        self.agent.load_param()

    # def load_critics_param(self):
    #     for pursuer_id in self.params["pursuer_ids"]:
    #         self.agent_list[pursuer_id].load_critic_param()
    #
    # def load_all_networks(self):
    #     self.load_actors_param()
    #     self.load_critics_param()









