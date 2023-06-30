import numpy as np
import random
from copy import deepcopy as dc
class buffer():
    def __init__(self,params):
        self.memory_pool=[]
        #[�غ�(��ͬEpisodeÿ����������Ϊ��ͬ�غ�)*{"state":{"ego_pos":[steps*],"target_pos":[steps*],"traffic_state":[steps*],"topo_link_array":[steps*],"all_evaders_pos":[steps*]},
        # "action":[steps*action],"action_prob":[steps*action_prob],"reward":[steps*reward],"next_state":[steps*next_state],"done":[steps*done]}]

        # self.warm_epoch=params["warmup_epoch"] #��������
        self.params=params
        self.max_capacity = params["memory_capacity"]
        # self.temp_epoch_memory = {}
        #{pursuer_id: {"state":{"ego_pos":[steps*],"target_pos":[steps*],"traffic_state":[steps*],"topo_link_array":[steps*],"all_evaders_pos":[steps*]},
        # "action":[steps*action],"action_prob":[steps*action_prob],"reward":[steps*reward],"next_state":[steps*next_state],"done":[steps*done]}}
        # self.reset_temp_epoch_memory()

    def storage(self,all_state,all_action,all_action_prob,all_reward,all_next_state,done):

        each_pursuer_memory = self.reset_each_pursuer_memory()
        each_pursuer_memory["state"]["ego_pos"].append(np.array(all_state["ego_pos"]).squeeze().tolist())
        each_pursuer_memory["state"]["target_pos"].append(np.array(all_state["target_pos"]).squeeze().tolist())
        each_pursuer_memory["state"]["traffic_state"].append(np.array(all_state["traffic_state"]).squeeze().tolist())
        each_pursuer_memory["state"]["topo_link_array"].append([np.array(all_state["topo_link_array"]).squeeze().tolist()])
        each_pursuer_memory["state"]["all_evaders_pos"].append(np.array(all_state["all_evaders_pos"]).squeeze().tolist())
        each_pursuer_memory["state"]["steps"].append(np.array(all_state["steps"]).squeeze().tolist())


        each_pursuer_memory["action"].append(all_action)
        each_pursuer_memory["action_prob"].append(all_action_prob)
        each_pursuer_memory["reward"].append(all_reward)

        each_pursuer_memory["next_state"]["ego_pos"].append(
            np.array(all_next_state["ego_pos"]).squeeze().tolist())
        each_pursuer_memory["next_state"]["target_pos"].append(
            np.array(all_next_state["target_pos"]).squeeze().tolist())
        each_pursuer_memory["next_state"]["traffic_state"].append(
            np.array(all_next_state["traffic_state"]).squeeze().tolist())
        each_pursuer_memory["next_state"]["topo_link_array"].append(
            [np.array(all_next_state["topo_link_array"]).squeeze().tolist()])
        each_pursuer_memory["next_state"]["all_evaders_pos"].append(
            np.array(all_next_state["all_evaders_pos"]).squeeze().tolist())
        each_pursuer_memory["next_state"]["steps"].append(
            np.array(all_next_state["steps"]).squeeze().tolist())
        if done:
            d=0
        else:
            d=1
        each_pursuer_memory["done"].append(d)
        self.memory_pool.append(each_pursuer_memory)
        self.check_length()
            

            # self.temp_epoch_memory[pursuer_id]["next_state"].append(all_next_state)
            #end:done=T,d=0
            # for pursuer_id in self.params["pursuer_ids"]:
            #     self.memory_pool.append(dc(self.temp_epoch_memory[pursuer_id]))
            # self.reset_temp_epoch_memory()


    def random_sample(self,num_selects):
        if len(self.memory_pool)>num_selects:
            return random.sample(dc(self.memory_pool), num_selects)
        else:
            return dc(self.memory_pool)

    def given_index_sample(self,index_list):
        sample_set=[]
        for index in index_list:
            sample_set.append(self.memory_pool[index])
        return sample_set


    def reset_each_pursuer_memory(self):
        null_state = {
            "ego_pos": [],               #[steps*ego_pos]
            "target_pos": [],
            "traffic_state": [],
            "topo_link_array": [],
            "all_evaders_pos": [],
            "steps":[]
        }
        null_epoch={
            "state": dc(null_state),
            "action":[],                 #[steps*action]
            "action_prob":[],            #[steps*action_prob]
            "reward":[],                 #[steps*reward]
            "next_state":dc(null_state),
            "done":[]                    #[steps*done]

        }
        return dc(null_epoch)
        # for pursuer_id in self.params["pursuer_ids"]:
        #     self.temp_epoch_memory[pursuer_id]=dc(null_epoch)



    def get_length(self):
        return len(self.memory_pool)

    def check_length(self):
        while(len(self.memory_pool)>self.max_capacity):
            del self.memory_pool[0]