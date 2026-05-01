import torch
import torch.nn as nn
import torch.nn.functional as F

class NGCULoss(nn.Module):
    def __init__(self,
                 train_fn, 
                 c_max_threshold=0.9, 
                 eta1=0.01,   # L_bound 系数
                 eta2=0.001,  # L_stable 系数
                 eta3=0.05,   # L_metabolic 系数
                 target_rate=0.1): # 目标极低放电率 (10%)
        super().__init__()
        self.fn_loss = train_fn
        
        self.c_max_threshold = c_max_threshold
        self.eta1 = eta1
        self.eta2 = eta2
        self.eta3 = eta3
        self.target_rate = target_rate

    def forward(self, logits, targets, c_states_seq_list=None, firing_rates_list=None):
        loss_task = self.fn_loss(logits, targets)
        
        loss_bound = 0.0
        loss_stable = 0.0
        loss_metabolic = 0.0

        if len(c_states_seq_list) > 0:
            for c_seq in c_states_seq_list:
                loss_bound += torch.mean(F.relu(c_seq.abs() - 0.9))
                

                if c_seq.shape[0] > 1:
                    loss_stable += torch.mean((c_seq[1:] - c_seq[:-1]) ** 2)
            
            loss_bound /= len(c_states_seq_list)
            loss_stable /= len(c_states_seq_list)

        if len(firing_rates_list) > 0:
            for rate in firing_rates_list:
                # 3. 脉冲稀疏度惩罚 (0.1 即每 10 步发 1 个脉冲)
                loss_metabolic += torch.mean((rate - 0.1) ** 2)
            loss_metabolic /= len(firing_rates_list)

        loss_total = loss_task + 0.01 * loss_bound + 0.01 * loss_stable + 0.01 * loss_metabolic
        return loss_total