import numpy as np
import os
from DQN_Networks import DQN_net
from GAN_networks import Generator, Discriminator
import random
import torch.optim as optim
import torch
from copy import deepcopy as dc
from torch.distributions import Normal, Categorical
from env.utils import calculate_dis
import torch.nn as nn
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn.functional as F
import collections
import copy
import math

class GDQN():
    def __init__(self,params):
        self.train_times=0
        self.update_QNet_times = 0
        self.update_G_times = 0
        self.update_D_times = 0
        self.I_D=3
        # self.critic_update_times = 0
        # self.pursuer_id=pursuer_id

        self.params=params

        self.epsilon = self.params["epsilon"]
        self.epsilon_decay = self.params["epsilon_decay"]
        self.epsilon_min = self.params["epsilon_min"]
        self.num_actions=self.params["num_action"]
        self.tau=self.params["tau"]
        self.b1 = self.params["b1"]
        self.b2 = self.params["b2"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ##########state_shape
        self.num_pos = self.params["lane_code_length"] + 1
        self.num_edge = self.params["num_edge"]
        self.num_evader = self.params["num_evader"]
        # steps, ego_pos, target_pos, traffic_state, topo_link_array, all_evaders_pos
        self.state_input_shape = 1+self.num_pos+self.num_pos+self.num_edge+self.num_evader*self.num_pos
        
        self.DQN_Net = DQN_net(self.params["num_edge"], self.params["lane_code_length"] + 1, self.params["num_evader"])
        self.G = Generator(self.state_input_shape,self.params)
        self.D_list=[]
        self.D_optimizer_list=[]
        for i in range(self.I_D):
            self.D_list.append(Discriminator(self.params))
            self.D_list[i].to(self.device)
            self.D_optimizer_list.append(optim.Adam(self.D_list[i].parameters(), lr=self.params["d_learning_rate"],
                                          betas=(self.b1, self.b2)))


        self.DQN_Net.to(self.device)
        self.G.to(self.device)




        self.QNet_load_param()
        self.G_load_param()
        self.D_load_param()

        self.batch_size=self.params["GAN_batch_size"]


        self.g_times = self.params["g_times"]
        self.c_times = self.params["critic_times"]
        self.gp_lamda = self.params["gp_lamda"]

        self.DQN_optimizer = optim.Adam(self.DQN_Net.parameters(), self.params["DQN_learning_rate"])

        self.G_optimizer = optim.Adam(self.G.parameters(), lr=self.params["g_learning_rate"], betas=(self.b1, self.b2))

        self.test_reward=collections.deque(maxlen=10)
        self.Q_loss_list = collections.deque(maxlen=10)
        self.G_loss_list = collections.deque(maxlen=10)
        self.D_loss_list = collections.deque(maxlen=10)
        self.W_D_list = collections.deque(maxlen=10)

    def select_action(self, ego_state):
        if np.random.rand() < self.epsilon:
            action_prob = F.softmax(
                torch.tensor(np.random.rand(1, self.num_actions), device=self.device, dtype=torch.float32), dim=1)
        else:
            ego_pos_tensor = torch.tensor(ego_state["ego_pos"], dtype=torch.float,device=self.device)
            target_pos_tensor=torch.tensor(ego_state["target_pos"], dtype=torch.float,device=self.device)
            traffic_state_tensor= torch.tensor(ego_state["traffic_state"], dtype=torch.float,device=self.device)
            topo_link_array_tensor=torch.tensor([ego_state["topo_link_array"]], dtype=torch.float,device=self.device)
            # all_evaders_pos_tensor=torch.tensor(ego_state["all_evaders_pos"], dtype=torch.float,device=self.device)
            steps_tensor=torch.tensor(ego_state["steps"], dtype=torch.float,device=self.device)

            node_feature = [dc(ego_state["ego_pos"])]
            for i in range(len(ego_state["all_evaders_pos"][0])):
                node_feature[0].append(ego_state["all_evaders_pos"][0][i])
            node_feature_tensor = torch.tensor(node_feature, dtype=torch.float, device=self.device)


            with torch.no_grad():
                self.DQN_Net.eval()
                # print("****", traffic_state_tensor.size())
                # print("*****", topo_link_array_tensor.size())
                action_prob = self.DQN_Net(steps_tensor, target_pos_tensor,ego_pos_tensor, topo_link_array_tensor, traffic_state_tensor,
                                           node_feature_tensor)
        # target_pos,pursuers_pos,topo_link_array,background_veh,all_evader_pos
        #print("#########action_prob:", action_prob)
        action = torch.argmax(action_prob,dim=1).item()
        return action, action_prob[:,action].item()

    def transitions(self, train_set):
        ego_pos_list, target_pos_list, traffic_state_list, topo_link_array_list, all_evaders_pos_list, steps_list= [],[],[],[],[],[]
        action_list, reward_list, done_list = [], [], []
        back_state_list=[]
        for i in range(len(train_set)):
            state = train_set[i]["state"]
            ego_pos_list.append(state["ego_pos"])
            target_pos_list.append(state["ego_pos"])
            traffic_state_list.append(state["traffic_state"])
            back_state_list.append(np.array(state["traffic_state"]).sum(axis=2))
            topo_link_array_list.append(state["topo_link_array"])
            all_evaders_pos_list.append(state["all_evaders_pos"])
            steps_list.append(state["steps"])

            action = train_set[i]["action"]
            action_list.append(action)
            reward = train_set[i]["reward"]
            reward_list.append(reward)
            done = train_set[i]["done"]
            done_list.append(done)

        
        ego_pos_tensor=torch.tensor(ego_pos_list, dtype=torch.float, device=self.device) #torch.Size([64, 1, 7])
        target_pos_tensor=torch.tensor(target_pos_list, dtype=torch.float, device=self.device) #torch.Size([64, 1, 7])
        traffic_state_tensor=torch.tensor(traffic_state_list, dtype=torch.float, device=self.device) #torch.Size([64, 1, 48])
        topo_link_array_tensor=torch.tensor(topo_link_array_list, dtype=torch.float, device=self.device)  #torch.Size([64, 1, 1, 48, 48])
        all_evaders_pos_tensor=torch.tensor(all_evaders_pos_list, dtype=torch.float, device=self.device) #torch.Size([64, 1, 3, 7])
        steps_tensor=torch.tensor(steps_list, dtype=torch.float, device=self.device).view(-1,1) #torch.Size([64, 1])
        back_state_tensor=torch.tensor(back_state_list, dtype=torch.float, device=self.device)


        action_tensor = torch.tensor(action_list).view(-1, 1).to(self.device)   # torch.Size([64, 1])    
        reward_tensor = torch.tensor(reward_list, dtype=torch.float32).view(-1, 1).to(self.device)  #torch.Size([64, 1])
        #done_tensor=torch.tensor(done_list,dtype=torch.float).view(-1,1).to(self.device)
        # print("********",traffic_state_tensor.size())
        # print("********", back_state_tensor.size())

        return ego_pos_tensor, target_pos_tensor,traffic_state_tensor, back_state_tensor,topo_link_array_tensor,all_evaders_pos_tensor, steps_tensor,action_tensor,reward_tensor

        
    def update(self,training_set,Cumul_Reward_list):
        self.DQN_Net.eval()
        Q_loss=[]
        D_loss = []
        G_loss = []
        W_D = []
        ego_pos_tensor, target_pos_tensor,traffic_state_tensor, back_state_tensor,topo_link_array_tensor,all_evaders_pos_tensor, steps_tensor,action_tensor,reward_tensor = self.transitions(training_set)

        cumul_reward_tensor = torch.tensor(Cumul_Reward_list,dtype=torch.float).view(-1, 1).to(self.device)


        for i in range(self.g_times):
            #print("**************")
            for index in BatchSampler(SubsetRandomSampler(range(action_tensor.shape[0])), self.batch_size, True):
                # print("############", len(index))
                # #print("########action_tensor.shape#######",action_tensor.size()) #torch.size([env_steps,1])
                # print("############", back_state_tensor[index].size())
                lg,ld,wd = self.WGAN_gp(steps_tensor[index],ego_pos_tensor[index],target_pos_tensor[index],back_state_tensor[index],topo_link_array_tensor[index],
                                all_evaders_pos_tensor[index],action_tensor[index],cumul_reward_tensor[index])
                G_loss.append(lg)
                D_loss.append(ld)
                W_D.append(wd)

        
        for i in range(self.params["gdqn_update_times"]):
            for index in BatchSampler(SubsetRandomSampler(range(action_tensor.shape[0])), self.batch_size, True):
                ego_pos = ego_pos_tensor[index]
                # print("*******",ego_pos.size())
                # print("*********",all_evaders_pos_tensor.size())
                # print("**************",torch.squeeze(dc(all_evaders_pos_tensor[index])).size())
                node_feature = torch.cat([dc(ego_pos), torch.squeeze(dc(all_evaders_pos_tensor[index]))],dim=1)
                # for i in range(self.params["num_evader"]):
                #     node_feature[0].append(all_evaders_pos_tensor[index][0][i].detach().cpu().numpy().tolist())
                # print("**********", np.array(node_feature).shape)
                node_feature_tensor = torch.tensor(node_feature, dtype=torch.float, device=self.device)  # （1,ego+all_evader,7）
                # print("####", traffic_state_tensor[index].size())
                # print("####", topo_link_array_tensor[index].size())

                Q_values = self.DQN_Net(steps_tensor[index], target_pos_tensor[index].view(-1,self.num_pos),ego_pos_tensor[index].view(-1,self.num_pos), topo_link_array_tensor[index].view(-1, 1, 48, 48),
                                        traffic_state_tensor[index].view(-1, 1, 48, 3), node_feature_tensor)
                Q_values=Q_values.gather(1,action_tensor[index])  # new policy
                # Q = torch.gather(Q_values,1,action_tensor[index])
                
                # next_Q_values=self.target_DQN_Net(next_steps_tensor[index],next_ego_pos_tensor[index], next_target_pos_tensor[index], next_traffic_state_tensor[index],
                #                                  next_topo_link_array_tensor[index], next_all_evaders_pos_tensor[index])
                # next_Q_values = next_Q_values.max(1)[0].view(-1, 1)
                #
                # target_Q_values=reward_tensor[index]+self.params["gamma"]*(next_Q_values*done_tensor[index])
                z = torch.randn(self.batch_size, 1, 1).to(self.device)
                estimate_Q = reward_tensor[index]+0.7*self.G(steps_tensor[index],ego_pos_tensor[index], target_pos_tensor[index], back_state_tensor[index],
                                                    topo_link_array_tensor[index], all_evaders_pos_tensor[index],action_tensor[index],z)
                
                # print("#########R: ", cumul_reward_tensor[index])
                # print("#######Q:", Q)
                # print("#######estimate_Q:", estimate_Q)
                #print("*********the gap between Q and estimate_Q is ", Q-estimate_Q)
                
                #target_Q, estimate_Q取最小值，或均值
                ####两个Q：归一化，限制上下限

                ##G的
                                                                                                        
                self.DQN_optimizer.zero_grad()
                loss=F.mse_loss(Q_values, estimate_Q)
                loss.backward()
                self.DQN_optimizer.step()
                Q_loss.append(loss.item())
        

        #self.soft_update_target_network()
        self.train_times=self.train_times+1
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)
        
        return np.array(Q_loss).mean(), np.array(G_loss).mean(),np.array(D_loss).mean(), np.array(W_D).mean()

    def WGAN_gp(self, steps,ego_pos, target_pos, traffic_state,topo_link_array,all_evaders_pos, action, cumul_reward): #train_set:[ {state},[],[] ] #state, action, reward


        real_set = copy.deepcopy(cumul_reward).to(self.device)
        Wasserstein_D = 0
        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1
        one.to(self.device)
        mone.to(self.device)
        
        LD, LG =0,0
        LW =0

        # Train Dicriminator forward-loss-backward-update self.c_times while 1 Generator forward-loss-backward-update
        for d_iter in range(self.c_times):
            for i in range(self.I_D):
                self.D_list[i].zero_grad()
                z = torch.rand((self.batch_size, 1, 1)).to(self.device) #noise

                # Train discriminator
                # WGAN - Training discriminator more iterations than generator

                # Train with real images

                d_loss_real = self.D_list[i](real_set)
                d_loss_real = d_loss_real.mean()
                # d_loss_real.backward(mone)

                # Train with fake images
                ###
                fake_set = self.G(steps,ego_pos, target_pos, traffic_state,topo_link_array,all_evaders_pos, action,z).detach() #########噪声+(s,a)
                d_loss_fake = self.D_list[i](fake_set)
                d_loss_fake = d_loss_fake.mean()
                # d_loss_fake.backward(one)

                # Train with gradient penalty
                gradient_penalty = self.calculate_gradient_penalty(real_set,fake_set,i)
                # gradient_penalty.backward()

                d_loss = d_loss_fake - d_loss_real + gradient_penalty
                Wasserstein_D = d_loss_real - d_loss_fake
                d_loss.backward()
                self.D_optimizer_list[i].step()
                LD = LD + d_loss.item()
                LW = LW +Wasserstein_D.item()


        # Generator update
        self.G.zero_grad()
        # train generator
        # compute loss with fake images
        z = torch.randn(self.batch_size, 1, 1).to(self.device)
        fake_data = self.G(steps,ego_pos, target_pos, traffic_state,topo_link_array,all_evaders_pos, action,z)
        g_loss_list=[]
        for i in range(self.I_D):
            g_loss_ego=-self.D_list[i](fake_data).mean()
            g_loss_list.append(g_loss_ego)
        g_loss = self.Pythagorean_Mean(g_loss_list)
        #g_loss.backward(mone)
        g_loss.backward()
        self.G_optimizer.step()
        LG = LG + g_loss.item()


        return LG, LD/(self.c_times*self.I_D), LW/(self.c_times*self.I_D)


    def Pythagorean_Mean(self,a,lam=0.5):
        exp_sum_tensor=torch.tensor(0)
        for i in range(self.I_D):
            exp_sum_tensor=exp_sum_tensor+torch.exp(lam*a[i])
        sum_tensor = torch.tensor(0)
        for i in range(self.I_D):
            sum_tensor=sum_tensor+(torch.exp(lam*a[i])/exp_sum_tensor)*(torch.log(torch.abs(a[i])))
        return -torch.exp(sum_tensor)


    def calculate_gradient_penalty(self, real_set, fake_set,i):
        eta = torch.rand(size=(self.batch_size, 1, 1)).to(self.device)  # 真假样本的采样比例r，batch size个随机数，服从区间[0,1)的均匀分布
        # eta = eta.expand(self.batch_size, real_images.size(1), real_images.size(2), real_images.size(3))

        x_add = eta * real_set + ((1 - eta) * fake_set).requires_grad_(True)## # define it to calculate gradient
        x_add.to(self.device)


        # calculate probability of interpolated examples
        d_add = self.D_list[i](x_add)

        # calculate gradients of probabilities with respect to examples
        grad = torch.autograd.grad(  # 进行梯度计算
            outputs=d_add,  # 计算梯度的函数d，即D(x)
            inputs=x_add,  # 计算梯度的变量x
            grad_outputs=torch.ones_like(d_add).to(self.device),  # 梯度计算权重
            create_graph=True,  # 创建计算图
            retain_graph=True  # 保留计算图
        )[0]  # 返回元组的第一个元素为梯度计算结果
        #####retain_graph = True，-->False
        grad_penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean() * self.gp_lamda
        return grad_penalty



    def save_QNet_param(self):
        dir_path = 'agent_param/' + self.params["env_name"] + '/' + self.params["exp_name"]
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self.update_QNet_times=self.update_QNet_times+1
        torch.save(self.DQN_Net.state_dict(), 'agent_param/' + self.params["env_name"] +'/'+self.params["exp_name"]+'/'+'_QNet.pth')

    def save_G_param(self):
        dir_path = 'agent_param/' + self.params["env_name"] + '/' + self.params["exp_name"]
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self.update_G_times=self.update_G_times+1
        torch.save(self.G.state_dict(), 'agent_param/' + self.params["env_name"] +'/'+self.params["exp_name"]+'/'+'_G.pth')

    def save_D_param(self):
        dir_path = 'agent_param/' + self.params["env_name"] + '/' + self.params["exp_name"]
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self.update_D_times = self.update_D_times + 1
        for i in range (self.I_D):
            torch.save(self.D_list[i].state_dict(),
                   'agent_param/' + self.params["env_name"] + '/' + self.params["exp_name"] + '/' + '_D_'+str(i)+'.pth')

    def QNet_load_param(self):
        file_path='agent_param/' + self.params["env_name"] +'/'+self.params["exp_name"]+'/'+'_QNet.pth'
        if os.path.exists(file_path):
            print("loading from param file QNet....")
            self.DQN_Net.load_state_dict(torch.load(file_path))
            self.DQN_Net.to(self.device)
        else:
            print("creating new param for QNet....")

    def G_load_param(self):
        file_path = 'agent_param/' + self.params["env_name"]+'/'+self.params["exp_name"] + '/' + '_G.pth'
        if os.path.exists(file_path):
            print("loading from param file G....")
            self.G.load_state_dict(torch.load(file_path))
            self.G.to(self.device)
        else:
            print("creating new param for G....")

    def D_load_param(self):
        for i in range (self.I_D):
            file_path = 'agent_param/' + self.params["env_name"]+'/'+self.params["exp_name"] + '/' + '_D_'+str(i)+'.pth'
            if os.path.exists(file_path):
                print("loading from param file D....")
                self.D_list[i].load_state_dict(torch.load(file_path))
                self.D_list[i].to(self.device)
            else:
                print("creating new param for D "+str(i)+"....")



    def load_param(self):
        self.QNet_load_param()
        self.G_load_param()
        self.D_load_param()








