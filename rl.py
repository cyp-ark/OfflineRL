import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(QNetwork,self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc = nn.Sequential(
            nn.Linear(self.state_dim,64)
            nn.Relu()
            nn.Linear(64,action_dim)
        )

    def forward(self,x):
        x = self.fc(x)
        return x

class RL(object):
    def __init__(self,state_dim,action_dim,gamma,
                learning_rate,update_freq,device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.update_freq = update_freq
        self.device = device
        self.network = QNetwork(state_dim=self.state_dim,action_dim=self.action_dim)

    def train_on_batch(self,s,a,r,s2,t):
        s = s.to(device)
        a = a.to(device)
        r = r2.to(device)
        s2 = s2.to(device)
        t = t.to(device)

        q = network(s)
        q2 = target_network(s2).detach()
        q_pred = q.gather(1, a).squeeze()

        q2_net,_,_ = network(s2)
        q2_net = q2_net.detach()
        q2_max = q2.gather(1, torch.max(q2_net,dim=1)[1].unsqueeze(1)).squeeze()

        bellman_target = torch.clamp(r, max=1.0, min=0.0) + gamma * torch.clamp(q2_max.detach(), max=1.0, min=0.0)*(1-t)
        bellman_loss = F.mse_loss(q_pred, bellman_target)
        cql1_loss = cql_loss(q,a)
        
        loss = cql1_loss + bellman_loss
        loss_val += loss.item()