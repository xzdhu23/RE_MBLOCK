import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from config import gamma


class QNet(nn.Module):

    def __init__(self, num_input, num_output):
        super(QNet, self).__init__()
        self.num_input = num_input
        self.num_output = num_output

        self.fc1 = nn.Linear(num_input, 1024)
        self.fc11 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, num_output)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x1 = F.relu(self.fc11(x))
        qvalue = self.fc2(x1)
        return qvalue

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch):
        states = torch.stack(batch.state)
        next_states = torch.stack(batch.next_state)
        actions = torch.Tensor(np.array(batch.action)).float()
        rewards = torch.Tensor(batch.reward)

        pred = online_net(states).squeeze(1)
        next_pred = target_net(next_states).squeeze(1)
        pred = torch.sum(pred.mul(actions), dim=1)

        target = rewards + gamma * next_pred.max(1)[0]

        loss = F.mse_loss(pred, target.detach())
        optimizer.step()

        return loss

    def get_action(self, input):
        qvalue = self.forward(input)
        action = torch.max(qvalue, 0)[1].data.numpy().item()
        return action