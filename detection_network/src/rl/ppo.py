

import torch
import torch.optim as optim

class RolloutStorage:

    def __init__(self,num_steps,num_processes,action_size):
        pass


class PPO(object):
    def __init__(
        self,
        controller,
        clip_param,
        lr,
        baseline_decay,
        action_size = 18,
        ppo_epoch=1,
        num_mini_batch=100,
        max_grad_norm=2.0,
        entropy_coef=0,
        num_steps=100,
        num_processes=1
        ):
        self.ppo_epoch = ppo_epoch
        self.controller = controller
        self.optimizer = optim.Adam(controller.parameters(),lr=lr)
        self.num_mini_batch = num_mini_batch
        self.clip_param = clip_param
        self.max_grad_norm = max_grad_norm
        self.entropy_coef = entropy_coef
        self.rollouts = RolloutStorage(num_steps,num_processes,action_size)

        self.baseline = None
        self.decay = baseline_decay

    def state_dict(self):
        return {
            "baseline":self.baseline,
            "rollouts":self.controller.state_dict(),
            "optimizer:":self.optimizer.state_dict()
        }

    def load_state_dict(self,states):
        pass

    
    def update(self, sample, is_train=True):
        reward, action, log_prob = sample

        if self.baseline is None:
            self.baseline = reward
        else:
            self.baseline = self.decay * self.baseline + (1 - self.decay) * reward

        self.rollouts.insert(action, log_prob, reward)

        if not is_train:
            return -1,-1

        advantages = self.rollouts.rewards - self.baseline

        loss_epoch = 0
        entropy_epoch = 0

        for _ in range(self.ppo_epoch):
            data_generator = self.rollouts.generator(advantages, self.num_mini_batch)
            for sample in data_generator:
                (
                    actions_batch,
                    reward_batch,
                    old_actions_log_probs_batch,
                    adv_targ,
                ) = sample

                action_log_probs, entropy = self.controller.evaluate_actions(
                    actions_batch
                )

                ratio = torch.exp(
                    action_log_probs - torch.from_numpy(adv_targ)
                )

                

