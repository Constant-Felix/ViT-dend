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

        self.last_terms = {}

    @staticmethod
    def _as_list(states):
        if states is None:
            return []
        if isinstance(states, (list, tuple)):
            return [state for state in states if torch.is_tensor(state)]
        if torch.is_tensor(states):
            return [states]
        return []

    def task_loss(self, logits, targets):
        return self.fn_loss(logits, targets)

    def regularization_loss(
        self,
        c_states_seq_list=None,
        firing_rates_list=None,
        astro_traces_list=None,
        ref_tensor=None,
    ):
        c_states_seq_list = self._as_list(c_states_seq_list)
        firing_rates_list = self._as_list(firing_rates_list)
        astro_traces_list = self._as_list(astro_traces_list)

        if ref_tensor is None:
            ref_candidates = c_states_seq_list + firing_rates_list + astro_traces_list
            ref_tensor = ref_candidates[0] if ref_candidates else next(self.parameters(), None)
        if ref_tensor is None:
            device = torch.device("cpu")
            dtype = torch.float32
        else:
            device = ref_tensor.device
            dtype = ref_tensor.dtype

        loss_bound = torch.zeros((), device=device, dtype=dtype)
        loss_stable = torch.zeros((), device=device, dtype=dtype)
        loss_metabolic = torch.zeros((), device=device, dtype=dtype)

        if c_states_seq_list:
            for c_seq in c_states_seq_list:
                c_seq = c_seq.float()
                overflow = F.relu(c_seq.abs() - self.c_max_threshold)
                loss_bound = loss_bound + torch.mean(overflow ** 2)

                if c_seq.shape[0] > 1:
                    loss_stable = loss_stable + torch.mean((c_seq[1:] - c_seq[:-1]) ** 2)

            loss_bound = loss_bound / len(c_states_seq_list)
            loss_stable = loss_stable / len(c_states_seq_list)

        if firing_rates_list:
            for rate in firing_rates_list:
                loss_metabolic = loss_metabolic + torch.mean((rate.float() - self.target_rate) ** 2)
            loss_metabolic = loss_metabolic / len(firing_rates_list)

        if astro_traces_list:
            trace_energy = torch.zeros((), device=device, dtype=dtype)
            for trace in astro_traces_list:
                trace_energy = trace_energy + torch.mean(trace.float() ** 2)
            loss_metabolic = loss_metabolic + 0.1 * trace_energy / len(astro_traces_list)

        total_reg = (
            self.eta1 * loss_bound
            + self.eta2 * loss_stable
            + self.eta3 * loss_metabolic
        )
        self.last_terms = {
            "bound": loss_bound.detach(),
            "stable": loss_stable.detach(),
            "metabolic": loss_metabolic.detach(),
            "regularization": total_reg.detach(),
        }
        return total_reg

    def forward(
        self,
        logits,
        targets,
        c_states_seq_list=None,
        firing_rates_list=None,
        astro_traces_list=None,
    ):
        loss_task = self.task_loss(logits, targets)
        loss_total = loss_task + self.regularization_loss(
            c_states_seq_list=c_states_seq_list,
            firing_rates_list=firing_rates_list,
            astro_traces_list=astro_traces_list,
            ref_tensor=logits,
        )
        self.last_terms["task"] = loss_task.detach()
        return loss_total