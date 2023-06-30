import argparse

params = {
          "exp_name":"gdqn",
          "Episode": 100000,             # 总的训练轮数
          "max_steps": 800,                # 单次仿真最大步数****
          "test_episodes": 1,          # 测试时执行的轮数****  8
          # "lane_length": 100,           # sumo中每条车道的长度****
          # "use_global_reward": True,    # 是否使用全局的奖励
          "memory_capacity": 800,       # 经验池保存回合数/agent_num
          "train_set_epoch": 1,         # 训练集选取的回合数
          "warmup_episodes": 1,         # 热启动轮数***  8
          "gamma": 0.99,                # 奖励的折扣因子:DQN
          "alpha": 0.001,               #
          "assign_method": "task_allocation",    # 目标分配方式
          "epsilon": 0.1,            # 贪婪策略epsilon初始值
          "epsilon_decay": 0.0001,      # 贪婪策略epsilon衰减
          "epsilon_min": 0.01,          # 贪婪策略epsilon最小值
          # ==========================DQN=============================
          "target_update_period": 5,  # 目标网络更新周期
          "DQN_learning_rate":1e-12,
          "dqn_update_times": 5,
          "dqn_batch_size":64,
          "tau":0.005, #软更新系数
          # ==========================GAN=============================
          "hidden_shape":64,
          "GAN_batch_size":64,
          "d_learning_rate": 4e-12,
          "g_learning_rate": 1e-12,
          "b1":0.5,
          "b2":0.999,
          "critic_times":5,
          "g_times":3,
          "gp_lamda": 10,
          "gdqn_update_times":5,
          "data_len":100,



          # ==========================MADDPG==========================
          "minimax": True,
          #===========================PPO=========================
          # "actor_learning_rate":1e-2,
          # "critic_learning_rate":5e-2,
          # "ppo_update_times":1,
          # "clip_param":0.2,
          # "max_grad_norm":0.5,
          #===========================env=======================
          "port":1,
          "gui": False,
          "env_name": "3x3Traffic_4_2",
          #=====================evaluate_net====================
          "train_model": "random", #random or evaluate
          "topo_feature": 3,
          "num_pos": 7
          }

if params["env_name"] == "3x3Traffic_4_2":#200辆背景车辆
    params["rou_path"] = "./env/3x3Traffic/3x3Grid42.rou.xml"
    params["cfg_path"] = "./env/3x3Traffic/3x3Grid42.sumocfg"
    params["net_path"] = "./env/3x3Traffic/3x3Grid.net.xml"
    params["num_pursuit"] = 4
    params["pursuer_ids"] = ["p0", "p1", "p2", "p3"]
    params["num_evader"] = 2
    params["evader_ids"] = ["e0", "e1"]
    params["num_action"] = 3
    params["lane_code_length"]=6
    # params["lane_length"]=500
    params["num_background_veh"]=200
    params["congested_lane"]=["E711"]
    params["congested_prob"]=0.2
    params["strat_warm_step"]=30
    params["num_edge"] = 48
    params["port"] = 501

if params["env_name"] == "3x3Traffic_7_4":#200辆背景车辆
    params["rou_path"] = "./env/3x3Traffic/3x3Grid74.rou.xml"
    params["cfg_path"] = "./env/3x3Traffic/3x3Grid74.sumocfg"
    params["net_path"] = "./env/3x3Traffic/3x3Grid.net.xml"
    params["num_pursuit"] = 7
    params["pursuer_ids"] = ["p0", "p1", "p2", "p3","p4","p5","p6"]
    params["num_evader"] = 4
    params["evader_ids"] = ["e0", "e1","e2","e3"]
    params["num_action"] = 3
    params["lane_code_length"]=6
    # params["lane_length"]=500
    params["num_background_veh"]=200
    params["congested_lane"]=["E711"]
    params["congested_prob"]=0.2
    params["strat_warm_step"]=30
    params["num_edge"] = 48
    params["port"] = 502

if params["env_name"] == "3x3Traffic_5_3":#200辆背景车辆
    params["rou_path"] = "./env/3x3Traffic/3x3Grid.r53ou.xml"
    params["cfg_path"] = "./env/3x3Traffic/3x3Grid53.sumocfg"
    params["net_path"] = "./env/3x3Traffic/3x3Grid.net.xml"
    params["num_pursuit"] = 5
    params["pursuer_ids"] = ["p0", "p1", "p2", "p3","p4"]
    params["num_evader"] = 3
    params["evader_ids"] = ["e0", "e1","e2"]
    params["num_action"] = 3
    params["lane_code_length"]=6
    # params["lane_length"]=500
    params["num_background_veh"]=200
    params["congested_lane"]=["E711"]
    params["congested_prob"]=0.2
    params["strat_warm_step"]=30
    params["num_edge"] = 48
    params["port"] = 503



GLOBAL_SEED = 520
import torch
import numpy
import random
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(GLOBAL_SEED)
numpy.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

