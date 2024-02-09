#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, Any

import numpy as np
import torch
from habitat import logger
from habitat.utils import profiling_wrapper
from torch import Tensor
from torch import nn as nn
from torch import optim as optim

from offnav.utils.utils import soft_update_from_to


class ILAgent(nn.Module):
    def __init__(
            self,
            actor_critic: nn.Module,
            num_envs: int,
            num_mini_batch: int,
            lr: Optional[float] = None,
            encoder_lr: Optional[float] = None,
            eps: Optional[float] = None,
            max_grad_norm: Optional[float] = None,
            wd: Optional[float] = None,
            optimizer: Optional[str] = "AdamW",
            entropy_coef: Optional[float] = 0.0,
    ) -> None:

        super().__init__()

        self.actor_critic = actor_critic

        self.num_mini_batch = num_mini_batch

        self.max_grad_norm = max_grad_norm
        self.num_envs = num_envs
        self.entropy_coef = entropy_coef

        # use different lr for visual encoder and other networks
        visual_encoder_params, other_params = [], []
        for name, param in actor_critic.named_parameters():
            if param.requires_grad:
                if "net.visual_encoder.backbone" in name:
                    visual_encoder_params.append(param)
                else:
                    other_params.append(param)
        logger.info(
            "Visual Encoder params: {}".format(len(visual_encoder_params))
        )
        logger.info("Other params: {}".format(len(other_params)))

        if optimizer == "AdamW":
            self.optimizer = optim.AdamW(
                [
                    {"params": visual_encoder_params, "lr": encoder_lr},
                    {"params": other_params, "lr": lr},
                ],
                lr=lr,
                eps=eps,
                weight_decay=wd,
            )
        else:
            self.optimizer = optim.Adam(
                list(
                    filter(
                        lambda p: p.requires_grad, actor_critic.parameters()
                    )
                ),
                lr=lr,
                eps=eps,
            )
        self.device = next(actor_critic.parameters()).device

    def forward(self, *x):
        raise NotImplementedError

    def update(self, rollouts) -> Tuple[float, float, float]:
        total_loss_epoch = 0.0
        total_entropy = 0.0
        total_action_loss = 0.0

        profiling_wrapper.range_push("BC.update epoch")
        data_generator = rollouts.recurrent_generator(self.num_mini_batch)
        cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")
        hidden_states = []

        for batch in data_generator:
            # Reshape to do in a single forward pass for all steps
            (logits, rnn_hidden_states, dist_entropy) = self.actor_critic(
                batch["observations"],
                batch["recurrent_hidden_states"],
                batch["prev_actions"],
                batch["masks"],
            )

            N = batch["recurrent_hidden_states"].shape[0]
            T = batch["actions"].shape[0] // N
            actions_batch = batch["actions"].view(T, N, -1)
            logits = logits.view(T, N, -1)

            action_loss = cross_entropy_loss(
                logits.permute(0, 2, 1), actions_batch.squeeze(-1)
            )
            entropy_term = dist_entropy * self.entropy_coef

            self.optimizer.zero_grad()
            inflections_batch = batch["observations"][
                "inflection_weight"
            ].view(T, N, -1)

            action_loss_term = (
                    (inflections_batch * action_loss.unsqueeze(-1)).sum(0)
                    / inflections_batch.sum(0)
            ).mean()
            total_loss = action_loss_term - entropy_term

            self.before_backward(total_loss)
            total_loss.backward()
            self.after_backward(total_loss)

            self.before_step()
            self.optimizer.step()
            self.after_step()

            total_loss_epoch += total_loss.item()
            total_action_loss += action_loss_term.item()
            total_entropy += dist_entropy.item()
            hidden_states.append(rnn_hidden_states)

        profiling_wrapper.range_pop()

        hidden_states = torch.cat(hidden_states, dim=0).detach()

        total_loss_epoch /= self.num_mini_batch
        total_entropy /= self.num_mini_batch
        total_action_loss /= self.num_mini_batch

        return (
            total_loss_epoch,
            hidden_states,
            total_entropy,
            total_action_loss,
        )

    def before_backward(self, loss: Tensor) -> None:
        pass

    def after_backward(self, loss: Tensor) -> None:
        pass

    def before_step(self) -> None:
        nn.utils.clip_grad_norm_(
            self.actor_critic.parameters(), self.max_grad_norm
        )

    def after_step(self) -> None:
        pass


EPS_PPO = 1e-5


class IQLAgent(nn.Module):
    def __init__(
            self,
            actor_critic: nn.Module,
            num_envs: int,
            num_mini_batch: int,
            policy_update_period: int,
            q_update_period: int,
            target_update_period: int,
            eps: Optional[float] = None,
            clip_score: Optional[float] = 100,
            entropy_coef: Optional[float] = 0.0,
            discount: Optional[float] = 0.99,
            quantile: Optional[float] = 0.7,
            beta: Optional[float] = 1.0 / 3,
            policy_lr: Optional[float] = 3E-4,
            qf_lr: Optional[float] = 3E-4,
            policy_weight_decay: Optional[float] = 0,
            q_weight_decay: Optional[float] = 0,
            soft_target_tau: Optional[float] = 0.005
    ) -> None:

        super().__init__()

        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.q_update_period = q_update_period
        self.policy_update_period = policy_update_period
        self.beta = beta
        self.quantile = quantile
        self.actor_critic = actor_critic

        self.num_mini_batch = num_mini_batch

        self.clip_score = clip_score
        self.num_envs = num_envs
        self.entropy_coef = entropy_coef
        self.discount = discount

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        # Optimizers
        self.policy_optimizer = optim.Adam(
            list(
                filter(
                    lambda p: p.requires_grad, actor_critic.parameters()
                )
            ),
            lr=policy_lr,
            weight_decay=policy_weight_decay,
            eps=eps,
        )
        self.qf1_optimizer = optim.Adam(
            list(
                filter(
                    lambda p: p.requires_grad, actor_critic.qf1.parameters()
                )
            ),
            lr=qf_lr,
            weight_decay=q_weight_decay,
            eps=eps,
        )
        self.qf2_optimizer = optim.Adam(
            list(
                filter(
                    lambda p: p.requires_grad, actor_critic.qf2.parameters()
                )
            ),
            lr=qf_lr,
            weight_decay=q_weight_decay,
            eps=eps,
        )
        self.vf_optimizer = optim.Adam(
            list(
                filter(
                    lambda p: p.requires_grad, actor_critic.vf.parameters()
                )
            ),
            lr=qf_lr,
            weight_decay=q_weight_decay,
            eps=eps,
        )

        self.device = next(actor_critic.parameters()).device

    def forward(self, *x):
        raise NotImplementedError

    def update(self, rollouts, num_steps_done) -> Tuple[float, Any, float, float]:
        profiling_wrapper.range_push("OFF.update epoch")
        data_generator = rollouts.recurrent_generator(self.num_mini_batch)

        for batch in data_generator:
            obs = batch["observations"]
            actions = batch["actions"]
            rewards = batch["rewards"]
            next_obs = batch["next_observations"]
            terminals = torch.logical_not(batch["masks"]).float()

            # Shuffle the batch data to meet DQN random criteria
            indexes = torch.randperm(actions.shape[0])
            actions = actions[indexes]
            rewards = rewards[indexes]
            terminals = terminals[indexes]

            for k in obs:
                obs[k] = obs[k][indexes]
                next_obs[k] = next_obs[k][indexes]

            """
            QF Loss
            """
            q1_pred = self.actor_critic.qf1(obs, actions)
            q2_pred = self.actor_critic.qf2(obs, actions)
            target_vf_pred = self.actor_critic.vf(next_obs, actions).detach()

            q_target = rewards + (1. - terminals) * self.discount * target_vf_pred
            q_target = q_target.detach()
            qf1_loss = self.qf_criterion(q1_pred, q_target)
            qf2_loss = self.qf_criterion(q2_pred, q_target)

            """
            VF Loss
            """
            q_pred = torch.min(
                self.actor_critic.target_qf1(obs, actions),
                self.actor_critic.target_qf2(obs, actions),
            ).detach()
            vf_pred = self.actor_critic.vf(obs, actions)
            vf_err = vf_pred - q_pred
            vf_sign = (vf_err > 0).float()
            vf_weight = (1 - vf_sign) * self.quantile + vf_sign * (1 - self.quantile)
            vf_loss = (vf_weight * (vf_err ** 2)).mean()

            """
            Policy Loss
            """
            dist = self.actor_critic(obs, actions)
            policy_logpp = dist.log_prob(actions)

            adv = q_pred - vf_pred
            exp_adv = torch.exp(adv / self.beta)
            if self.clip_score is not None:
                exp_adv = torch.clamp(exp_adv, max=self.clip_score)

            weights = exp_adv[:, 0].detach()
            policy_loss = (-policy_logpp * weights).mean()

            """
            Update networks
            """
            if num_steps_done % self.q_update_period == 0:
                self.qf1_optimizer.zero_grad()
                qf1_loss.backward()
                self.qf1_optimizer.step()

                self.qf2_optimizer.zero_grad()
                qf2_loss.backward()
                self.qf2_optimizer.step()

                self.vf_optimizer.zero_grad()
                vf_loss.backward()
                self.vf_optimizer.step()

            if num_steps_done % self.policy_update_period == 0:
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

            """
            Soft Updates
            """
            if num_steps_done % self.target_update_period == 0:
                soft_update_from_to(
                    self.actor_critic.qf1, self.actor_critic.target_qf1, self.soft_target_tau
                )
                soft_update_from_to(
                    self.actor_critic.qf2, self.actor_critic.target_qf2, self.soft_target_tau
                )

        profiling_wrapper.range_pop()

        # hidden_states = torch.cat(hidden_states, dim=0).detach()

        # Save for statistics
        stats = dict(
            qf1_loss=np.mean(self.get_numpy(qf1_loss)),
            qf2_loss=np.mean(self.get_numpy(qf2_loss)),
            policy_loss=np.mean(self.get_numpy(policy_loss)),
            q1_pred=np.mean(self.get_numpy(q1_pred)),
            q2_pred=np.mean(self.get_numpy(q2_pred)),
            q_target=np.mean(self.get_numpy(q_target)),
            weights=np.mean(self.get_numpy(weights)),
            adv=np.mean(self.get_numpy(adv)),
            vf_pred=np.mean(self.get_numpy(vf_pred)),
            vf_loss=np.mean(self.get_numpy(vf_loss)),
        )
        return stats

    def before_backward(self, loss: Tensor) -> None:
        pass

    def after_backward(self, loss: Tensor) -> None:
        pass

    def before_step(self) -> None:
        nn.utils.clip_grad_norm_(
            self.actor_critic.parameters(), self.max_grad_norm
        )

    def after_step(self) -> None:
        pass

    @staticmethod
    def get_numpy(tensor):
        return tensor.to('cpu').detach().numpy()


class IQLRNNAgent(nn.Module):
    def __init__(
            self,
            actor_critic: nn.Module,
            num_envs: int,
            num_mini_batch: int,
            policy_update_period: int,
            q_update_period: int,
            target_update_period: int,
            eps: Optional[float] = None,
            clip_score: Optional[float] = 100,
            entropy_coef: Optional[float] = 0.0,
            discount: Optional[float] = 0.99,
            quantile: Optional[float] = 0.7,
            beta: Optional[float] = 1.0 / 3,
            policy_lr: Optional[float] = 3E-4,
            qf_lr: Optional[float] = 3E-4,
            policy_weight_decay: Optional[float] = 0,
            q_weight_decay: Optional[float] = 0,
            soft_target_tau: Optional[float] = 0.005
    ) -> None:

        super().__init__()

        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.q_update_period = q_update_period
        self.policy_update_period = policy_update_period
        self.beta = beta
        self.quantile = quantile
        self.actor_critic = actor_critic

        self.num_mini_batch = num_mini_batch

        self.clip_score = clip_score
        self.num_envs = num_envs
        self.entropy_coef = entropy_coef
        self.discount = discount

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        # Optimizers
        self.policy_optimizer = optim.AdamW(
            list(
                filter(
                    lambda p: p.requires_grad, actor_critic.parameters()
                )
            ),
            lr=policy_lr,
            weight_decay=policy_weight_decay,
            eps=eps,
        )
        self.qf1_optimizer = optim.AdamW(
            list(
                filter(
                    lambda p: p.requires_grad, actor_critic.qf1.parameters()
                )
            ),
            lr=qf_lr,
            weight_decay=q_weight_decay,
            eps=eps,
        )
        # self.qf2_optimizer = optim.Adam(
        #     list(
        #         filter(
        #             lambda p: p.requires_grad, actor_critic.qf2.parameters()
        #         )
        #     ),
        #     lr=qf_lr,
        #     weight_decay=q_weight_decay,
        #     eps=eps,
        # )
        self.vf_optimizer = optim.AdamW(
            list(
                filter(
                    lambda p: p.requires_grad, actor_critic.vf.parameters()
                )
            ),
            lr=qf_lr,
            weight_decay=q_weight_decay,
            eps=eps,
        )

        self.device = next(actor_critic.parameters()).device

    def forward(self, *x):
        raise NotImplementedError

    def update(self, rollouts, num_steps_done) -> Tuple[float, Any, float, float]:
        profiling_wrapper.range_push("OFF.update epoch")
        data_generator = rollouts.recurrent_generator(self.num_mini_batch)
        hidden_states_qf1 = []
        hidden_states_tqf1 = []
        hidden_states_policy = []
        total_sampled_actions = []
        total_deterministic_actions = []
        total_dataset_actions = []
        total_qf1_loss = 0.0
        total_policy_loss = 0.0
        total_q1_pred = 0.0
        total_q_target = 0.0
        total_weights = 0.0
        total_adv = 0.0
        total_vf_pred = 0.0
        total_vf_loss = 0.0

        for batch in data_generator:
            obs = batch["observations"]
            actions = batch["actions"]
            rewards = batch["rewards"]
            next_obs = batch["next_observations"]
            masks = batch["masks"]
            terminals = torch.logical_not(batch["masks"]).float()
            rnn_hidden_states = batch["recurrent_hidden_states"]
            prev_actions = batch["prev_actions"]
            inflections_batch = batch["observations"]["inflection_weight"]

            # Put all predictions together
            q1_pred, rnn_hidden_q1 = self.actor_critic.qf1(obs, rnn_hidden_states['qf1'], actions, prev_actions, masks)
            # q2_pred, rnn_hidden_q2 = self.actor_critic.qf2(obs, rnn_hidden_states, actions, masks)
            target_vf_pred = self.actor_critic.vf(next_obs, actions).detach()
            tq1_pred, rnn_hidden_tq1 = self.actor_critic.target_qf1(obs, rnn_hidden_states['tqf1'], actions, prev_actions, masks)
            # tq2_pred, rnn_hidden_tq2 = self.actor_critic.target_qf2(obs, rnn_hidden_states, actions, masks)
            q_pred = tq1_pred.detach()
            vf_pred = self.actor_critic.vf(obs, actions)
            dist, rnn_hidden_policy, entropy = self.actor_critic(obs, rnn_hidden_states['policy'], actions, prev_actions, masks)

            """
            QF Loss
            """
            q_target = rewards + (1. - terminals) * self.discount * target_vf_pred
            q_target = q_target.detach()
            qf1_loss = self.qf_criterion(q1_pred, q_target)
            # qf2_loss = self.qf_criterion(q2_pred, q_target)

            """
            VF Loss
            """
            vf_err = vf_pred - q_pred
            vf_sign = (vf_err > 0).float()
            vf_weight = (1 - vf_sign) * self.quantile + vf_sign * (1 - self.quantile)
            vf_loss = (vf_weight * (vf_err ** 2)).mean()

            """
            Policy Loss
            """
            policy_logpp = dist.log_prob(actions.squeeze())
            policy_loss_term = (inflections_batch * policy_logpp).sum(0) / inflections_batch.sum(0)
            sampled_actions = dist.sample().detach().cpu().numpy()
            deterministic_actions = dist.mode().detach().cpu().numpy()
            dataset_actions = actions.detach().cpu().numpy()
            adv = q_pred - vf_pred
            exp_adv = torch.exp(adv / self.beta)
            if self.clip_score is not None:
                exp_adv = torch.clamp(exp_adv, max=self.clip_score)
            weights = exp_adv[:, 0].detach()
            policy_loss = (-policy_loss_term * weights).mean()

            """
            Update networks
            """
            if num_steps_done % self.q_update_period == 0:
                self.qf1_optimizer.zero_grad()
                qf1_loss.backward()
                self.qf1_optimizer.step()

                # self.qf2_optimizer.zero_grad()
                # qf2_loss.backward()
                # self.qf2_optimizer.step()

                self.vf_optimizer.zero_grad()
                vf_loss.backward()
                self.vf_optimizer.step()

            if num_steps_done % self.policy_update_period == 0:
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

            """
            Soft Updates
            """
            if num_steps_done % self.target_update_period == 0:
                soft_update_from_to(
                    self.actor_critic.qf1, self.actor_critic.target_qf1, self.soft_target_tau
                )
                # soft_update_from_to(
                #     self.actor_critic.qf2, self.actor_critic.target_qf2, self.soft_target_tau
                # )

            hidden_states_qf1.append(rnn_hidden_q1)
            hidden_states_tqf1.append(rnn_hidden_tq1)
            hidden_states_policy.append(rnn_hidden_policy)
            total_sampled_actions.append(sampled_actions.squeeze(1))
            total_deterministic_actions.append(deterministic_actions.squeeze(1))
            total_dataset_actions.append(dataset_actions.squeeze(1))
            total_qf1_loss += qf1_loss.item()
            total_policy_loss += policy_loss.item()
            total_q1_pred += q1_pred.mean().item()
            total_q_target += q_target.mean().item()
            total_weights += weights.mean().item()
            total_adv += adv.mean().item()
            total_vf_pred += vf_pred.mean().item()
            total_vf_loss += vf_loss.mean().item()

        profiling_wrapper.range_pop()

        hidden_states_qf1 = torch.cat(hidden_states_qf1, dim=0).detach()
        hidden_states_tqf1 = torch.cat(hidden_states_tqf1, dim=0).detach()
        hidden_states_policy = torch.cat(hidden_states_policy, dim=0).detach()
        total_qf1_loss /= self.num_mini_batch
        total_policy_loss /= self.num_mini_batch
        total_q1_pred /= self.num_mini_batch
        total_q_target /= self.num_mini_batch
        total_weights /= self.num_mini_batch
        total_adv /= self.num_mini_batch
        total_vf_pred /= self.num_mini_batch
        total_vf_loss /= self.num_mini_batch

        # Save hidden state dict
        hidden_states = dict(
            qf1=hidden_states_qf1,
            tqf1=hidden_states_tqf1,
            policy=hidden_states_policy,
        )

        # Save for statistics
        stats = dict(
            qf1_loss=total_qf1_loss,
            # qf2_loss=np.mean(self.get_numpy(qf2_loss)),
            policy_loss=total_policy_loss,
            q1_pred=total_q1_pred,
            # q2_pred=np.mean(self.get_numpy(q2_pred)),
            q_target=total_q_target,
            weights=total_weights,
            adv=total_adv,
            vf_pred=total_vf_pred,
            vf_loss=total_vf_loss,
        )

        action_distributions = dict(
            sampled_actions=np.array(total_sampled_actions),
            deterministic_actions=np.array(total_deterministic_actions),
            dataset_actions=np.array(total_dataset_actions),
        )

        return stats, hidden_states, action_distributions

    def before_backward(self, loss: Tensor) -> None:
        pass

    def after_backward(self, loss: Tensor) -> None:
        pass

    def before_step(self) -> None:
        # nn.utils.clip_grad_norm_(
        #     self.actor_critic.parameters(), self.max_grad_norm
        # )
        pass

    def after_step(self) -> None:
        pass

    @staticmethod
    def get_numpy(tensor):
        return tensor.to('cpu').detach().numpy()


class DecentralizedDistributedMixin:
    def init_distributed(self, find_unused_params: bool = True) -> None:
        r"""Initializes distributed training for the model

        1. Broadcasts the model weights from world_rank 0 to all other workers
        2. Adds gradient hooks to the model

        :param find_unused_params: Whether or not to filter out unused parameters
                                   before gradient reduction.  This *must* be True if
                                   there are any parameters in the model that where unused in the
                                   forward pass, otherwise the gradient reduction
                                   will not work correctly.
        """

        # NB: Used to hide the hooks from the nn.Module,
        # so they don't show up in the state_dict
        class Guard:
            def __init__(self, actor_critic, device):
                if torch.cuda.is_available():
                    self.ddp = torch.nn.parallel.DistributedDataParallel(
                        actor_critic, device_ids=[device], output_device=device
                    )
                else:
                    self.ddp = torch.nn.parallel.DistributedDataParallel(
                        actor_critic
                    )

        self._ddp_hooks = Guard(self.actor_critic, self.device)  # type: ignore
        # self.get_advantages = self._get_advantages_distributed

        self.reducer = self._ddp_hooks.ddp.reducer
        self.find_unused_params = find_unused_params

    def before_backward(self, loss: Tensor) -> None:
        super().before_backward(loss)  # type: ignore

        if self.find_unused_params:
            self.reducer.prepare_for_backward([loss])  # type: ignore
        else:
            self.reducer.prepare_for_backward([])  # type: ignore


class DDPILAgent(DecentralizedDistributedMixin, ILAgent):
    pass


class OffIQLAgent(DecentralizedDistributedMixin, IQLAgent):
    pass


class OffIQLRNNAgent(DecentralizedDistributedMixin, IQLRNNAgent):
    pass
