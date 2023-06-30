import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, state_input_shape,params):
        super(Generator,self).__init__()
        self.input_shape = state_input_shape
        self.params = params
        self.hidd_shape = self.params["hidden_shape"]
        self.batch_size = self.params["GAN_batch_size"]
        self.num_edge = self.params["num_edge"]
        ##state,action
       ##state:batch_size*
        #action: batch_size*1*num_actions

        #self.extract_features
        self.topo_link_model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=4, stride=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=2, stride=1, padding=0),
            nn.ReLU()
        )

        self.fc_link = nn.Linear(49, 24)

        self.fc1 = nn.Linear(self.input_shape,self.hidd_shape)
        self.fc2 = nn.Linear(self.hidd_shape+24, 32)#s_f+link
        self.fc3 = nn.Linear(1, self.hidd_shape)#action
        self.fc4 = nn.Linear(self.hidd_shape+32, 32)#s_f+a_f

        self.g_m = nn.Sequential(
            nn.Linear(32+1, 128),
            nn.LeakyReLU(0.02),
            nn.Linear(128,64),
            nn.LeakyReLU(0.02),
            nn.Linear(64,1),
            #nn.LeakyReLU(0.2)
        )

    def forward(self, steps,ego_pos,target_pos,traffic_state,topo_link_array,all_evaders_pos, actions,z):
        #####特征提取

        
        topo = self.topo_link_model(topo_link_array.view(self.batch_size, -1, self.num_edge,self.num_edge))#[64,1,7,7]
        topo = self.fc_link(topo.view(-1,49))

        s_o = torch.cat((steps,ego_pos.view(self.batch_size,-1),target_pos.view(self.batch_size,-1),torch.flatten(all_evaders_pos,1),torch.flatten(traffic_state,1)),1)
        #s_o size: torch.Size([64, 84])
        # print("************", s_o.size())
        s_o_f = F.elu(self.fc1(s_o))
        s_ = torch.cat((s_o_f,topo),1)
        s_f = F.elu(self.fc2(s_))
        actions_float = actions.float()
        a_f = F.elu(self.fc3(actions_float))
        x_f = torch.cat((s_f,a_f),1) #torch.Size([64, 96])
        features = F.elu(self.fc4(x_f))
        #######G
        ##########features size：[64,32]
        ########## z size: [64,1,1]
        f_z = torch.cat((features, z.view(self.batch_size, -1)), 1)
        g_z = self.g_m(f_z)
        #print("########G output:", g_z.size(),g_z) ##[64,1]
        return g_z


class Discriminator(nn.Module):
    def __init__(self, params):
        super(Discriminator,self).__init__()
        # Output_dim = 1
        self.params = params
        self.batch_size = self.params["GAN_batch_size"]
        self.d_m = nn.Sequential(
            nn.Linear(1, 128),
            nn.LeakyReLU(0.02),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.02),
            nn.Linear(64, 1),
            #nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.d_m(x) #torch.Size([64, 1])
        #print("########D output:", x.size(),x)
        return x
