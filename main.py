
from Muti_Agent import Muti_agent
# from Multi_agent_DDPG import Muti_agent
from env.environment import Environment
from settings import params
from copy import deepcopy as dc
import numpy as np
import csv
import os
from pathlib import Path


def run_test(episode,agents,test_times=params["test_episodes"]):
    test_mean_reward_dic = {} #�洢ƽ�����Իغϵ�����������ƽ��ÿ��reward
    test_reward_dic={} #�洢���в��Իغϵ�����������ƽ��ÿ��reward
    for pursuer_id in params["pursuer_ids"]:
        test_reward_dic[pursuer_id] = []
    test_steps=[]
    for test_time in range(test_times):
        env = Environment(params)
        state = env.reset()
        pro_state, pro_all_states = agents.process_state(state)  # state
        episode_mean_reward_dic={} #�洢��ǰ���Իغϵ�����������ƽ��ÿ��reward
        episode_reward_dic = {} #�洢��ǰ���Իغϵ�����������ÿ��reward

        for pursuer_id in params["pursuer_ids"]:
            episode_reward_dic[pursuer_id] = []
        for step in range(params["max_steps"]):
            all_action, all_action_prob, pursuit_target = agents.select_action(pro_state,pro_all_states)
            next_state, done, all_reward = env.step(all_action, pursuit_target)
            if not done:
                next_pro_state, next_pro_all_states = agents.process_state(next_state)
            else:
                next_pro_state, next_pro_all_states=dc(pro_state),dc(pro_all_states)
            for pursuer_id in params["pursuer_ids"]:
                episode_reward_dic[pursuer_id].append(all_reward[pursuer_id])
            
            for pursuer_id in params["pursuer_ids"]:
                agents.replay_buffer_dict[pursuer_id].storage(pro_all_states[pursuer_id], all_action[pursuer_id], all_action_prob[pursuer_id], all_reward[pursuer_id], next_pro_all_states[pursuer_id], done)
                # agents.replay_buffer.storage(pro_all_states, all_action, all_action_prob, all_reward, next_pro_all_states, done)
            
            state = dc(next_state)
            pro_state=dc(next_pro_state)
            pro_all_states=dc(next_pro_all_states)
            if done:
                break

        for pursuer_id in params["pursuer_ids"]:
            test_reward_dic[pursuer_id].append(np.array(episode_reward_dic[pursuer_id]).mean())
        test_steps.append(env.steps)
        for pursuer_id in params["pursuer_ids"]:
            episode_mean_reward_dic[pursuer_id] = np.array(episode_reward_dic[pursuer_id]).mean()
        print("=====episode:", episode," test_num:", test_time," steps:", env.steps, " rewards:", episode_mean_reward_dic, "=====")

    for pursuer_id in params["pursuer_ids"]:
        test_mean_reward_dic[pursuer_id]=np.array(test_reward_dic[pursuer_id]).mean()
    print("======test_episode:", episode, " steps:", np.array(test_steps).mean(), " rewards:", test_mean_reward_dic,"======")
    return test_mean_reward_dic,np.array(test_steps).mean() # 按照pursuer id为字典的平均reward,





def train_episode(agents):
    test_reward={}
    test_steps=0
    train_times=0

    for episode in range(params["Episode"]):
        agents.load_params()
        if episode==0:
            test_reward,test_steps=run_test(episode,agents,test_times=params["warmup_episodes"])
        else:
            # agents.load_all_networks()
            agents_Qloss, agents_Gloss, agents_Dloss, agents_WD = agents.train_agents()
            test_reward, test_steps = run_test(episode,agents)
            # train_times=agents.agent_list[params["pursuer_ids"][0]].train_times
            train_times = agents.agent.train_times

            save_loss_log(train_times, agents_Qloss, agents_Gloss, agents_Dloss, agents_WD)
            save_test_log(train_times, test_reward, test_steps)
        
        agents.test_steps_num.append(test_steps)
        

        # ******************************************保存模型的条件*************************************
        # TODO: 保存的条件改成什么？暂时改的是reward>平均的reward；还有什么条件

        mean_reward = np.mean([test_reward[key] for key in test_reward.keys()])
        agents.agent.test_reward.append(mean_reward)

        ###########三个网络的保存条件？？？？？？？？？？？
        if episode==0:
            if np.mean(agents.agent.test_reward) <= mean_reward:
                agents.agent.save_QNet_param()
                agents.agent.save_G_param()
                agents.agent.save_D_param()
            elif test_steps <= np.mean(agents.test_steps_num):
                agents.agent.save_QNet_param()
                agents.agent.save_G_param()
                agents.agent.save_D_param()

        else:
            agents.agent.Q_loss_list.append(agents_Qloss)
            agents.agent.G_loss_list.append(agents_Gloss)
            agents.agent.D_loss_list.append(agents_Dloss)
            agents.agent.W_D_list.append(agents_WD)
            
            if agents_Gloss <= np.mean(agents.agent.G_loss_list):
                agents.agent.save_G_param()
            if agents_Dloss <= np.mean(agents.agent.D_loss_list):
                agents.agent.save_D_param()



            if agents_Qloss <= np.mean(agents.agent.Q_loss_list):
                agents.agent.save_QNet_param()
            elif np.mean(agents.agent.test_reward) <= mean_reward:
                agents.agent.save_QNet_param()
                agents.agent.save_G_param()
                agents.agent.save_D_param()
            elif test_steps <= np.mean(agents.test_steps_num):
                agents.agent.save_QNet_param()
                agents.agent.save_G_param()
                agents.agent.save_D_param()



        if episode>=0:
            save_stable_log(train_times,agents)

    return 0


def save_test_log(episode,test_reward,test_steps):
    path='log/'+params["env_name"] +'/'+params["exp_name"]+'/test_log.csv'
    dir_path='log/'+params["env_name"] +'/'+params["exp_name"]
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    data_row=[]
    data_row.append(episode)
    data_row.append(test_steps)
    for id in params["pursuer_ids"]:
        data_row.append(test_reward[id])
    if os.path.exists(path):
        with open(path, 'a+',newline="") as f:
            csv_write = csv.writer(f)
            csv_write.writerow(data_row)
    else:
        with open(path, 'a+',newline="") as f:
            csv_write = csv.writer(f)
            title=dc(params["pursuer_ids"])
            title.insert(0,"steps")
            title.insert(0,"train_times")
            csv_write.writerow(title)
            csv_write.writerow(data_row)
    return 0

def save_loss_log(train_times,agents_Qloss, agents_Gloss, agents_Dloss, agents_WD):
    path = 'log/' + params["env_name"] + '/' + params["exp_name"] + '/loss_log.csv'
    dir_path = 'log/' + params["env_name"] + '/' + params["exp_name"]
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    data_row = []
    data_row.append(train_times)
    data_row.append(agents_Qloss)
    data_row.append(agents_Gloss)
    data_row.append(agents_Dloss)
    data_row.append(agents_WD)
   
    if os.path.exists(path):
        with open(path, 'a+', newline="") as f:
            csv_write = csv.writer(f)
            csv_write.writerow(data_row)
    else:
        title=[]
        title.append("train_times")
        title.append("QNet loss")
        title.append("G loss")
        title.append("D loss")
        title.append("Wasserstein D")

        with open(path, 'a+', newline="") as f:
            csv_write = csv.writer(f)
            csv_write.writerow(title)
            csv_write.writerow(data_row)



def save_stable_log(train_times,agents):
    path = 'log/' + params["env_name"] + '/' + params["exp_name"] + '/stable_log.csv'
    dir_path = 'log/' + params["env_name"] + '/' + params["exp_name"]
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    data_row = []
    data_row.append(train_times)
    data_row.append(np.mean(agents.test_steps_num))
   
    data_row.append(np.mean(agents.agent.test_reward))
    data_row.append(np.mean(agents.agent.Q_loss_list))
    data_row.append(agents.agent.update_QNet_times)
    data_row.append(np.mean(agents.agent.G_loss_list))
    data_row.append(agents.agent.update_G_times)
    data_row.append(np.mean(agents.agent.D_loss_list))
    data_row.append(agents.agent.update_D_times)
    data_row.append(np.mean(agents.agent.W_D_list))
 


    ########G和D？？？？？？

    if os.path.exists(path):
        with open(path, 'a+', newline="") as f:
            csv_write = csv.writer(f)
            csv_write.writerow(data_row)
    else:
        title = []
        title.append("train_times")
        title.append("steps")
        title.append("reward")
        title.append("Q loss")
        title.append("update_QNet_times")
        title.append("G loss")
        title.append("update_G_times")
        title.append("D loss")
        title.append("update_D_times")
        title.append("W_D")


        with open(path, 'a+', newline="") as f:
            csv_write = csv.writer(f)
            csv_write.writerow(title)
            csv_write.writerow(data_row)


def calculate_delta_rewards(agents,rewards):
    delta_rewards={}
    for id in params["pursuer_ids"]:
        delta_rewards[id]=(dc(rewards[id])-dc(np.mean(agents.agent_list[id].test_reward)))*10
    return delta_rewards



agents=Muti_agent(params)
train_episode(agents)


