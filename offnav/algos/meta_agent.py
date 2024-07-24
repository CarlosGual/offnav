#!/usr/bin/env python3
import copy
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, Any
import learn2learn as l2l
import higher
import numpy as np
import torch
from habitat import logger
from habitat.utils import profiling_wrapper
from torch import Tensor
from torch import nn as nn
from torch import optim as optim


class MILAgent(nn.Module):
    def __init__(
            self,
            actor_critic: nn.Module,
            num_envs: int,
            num_mini_batch: int,
            outer_lr: Optional[float] = None,
            outer_encoder_lr: Optional[float] = None,
            inner_lr: Optional[float] = None,
            eps: Optional[float] = None,
            max_grad_norm: Optional[float] = None,
            wd: Optional[float] = None,
            outer_optimizer: Optional[str] = "AdamW",
            entropy_coef: Optional[float] = 0.0,
    ) -> None:

        super().__init__()

        self.actor_critic = l2l.algorithms.MAML(actor_critic, lr=inner_lr, allow_nograd=True, allow_unused=True)

        self.num_mini_batch = num_mini_batch

        self.max_grad_norm = max_grad_norm
        self.num_envs = num_envs
        self.entropy_coef = entropy_coef

        # use different lr for visual encoder and other networks
        visual_encoder_params, other_params = [], []
        for name, param in self.actor_critic.named_parameters():
            if param.requires_grad:
                if "net.visual_encoder.backbone" in name:
                    visual_encoder_params.append(param)
                else:
                    other_params.append(param)
        logger.info(
            "Visual Encoder params: {}".format(len(visual_encoder_params))
        )
        logger.info("Other params: {}".format(len(other_params)))

        if outer_optimizer == "AdamW":
            self.outer_optimizer = optim.AdamW(
                [
                    {"params": visual_encoder_params, "lr": outer_encoder_lr},
                    {"params": other_params, "lr": outer_lr},
                ],
                lr=outer_lr,
                eps=eps,
                weight_decay=wd,
            )
        else:
            self.outer_optimizer = optim.Adam(
                list(
                    filter(
                        lambda p: p.requires_grad, self.actor_critic.parameters()
                    )
                ),
                lr=outer_lr,
                eps=eps,
            )
        self.device = next(self.actor_critic.parameters()).device

    def forward(self, *x):
        raise NotImplementedError

    def calculate_loss(self, task, learner):
        cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")
        (logits, rnn_hidden_states, dist_entropy) = learner(
            task["observations"],
            task["recurrent_hidden_states"],
            task["prev_actions"],
            task["masks"],
        )

        N = task["recurrent_hidden_states"].shape[0]
        T = task["actions"].shape[0] // N
        actions_batch = task["actions"].view(T, N, -1)
        logits = logits.view(T, N, -1)

        action_loss = cross_entropy_loss(
            logits.permute(0, 2, 1), actions_batch.squeeze(-1)
        )
        entropy_term = dist_entropy * self.entropy_coef

        inflections_batch = task["observations"][
            "inflection_weight"
        ].view(T, N, -1)

        action_loss_term = (
                (inflections_batch * action_loss.unsqueeze(-1)).sum(0)
                / inflections_batch.sum(0)
        ).mean()

        total_loss = action_loss_term - entropy_term

        return (
            total_loss,
            rnn_hidden_states,
            dist_entropy,
            action_loss_term)

    def calculate_valid_loss(self, task, learner, rnn_hidden_states):
        cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")
        (logits, rnn_hidden_states, dist_entropy) = learner(
            task["observations"],
            rnn_hidden_states,
            task["prev_actions"],
            task["masks"],
        )

        N = task["recurrent_hidden_states"].shape[0]
        T = task["actions"].shape[0] // N
        actions_batch = task["actions"].view(T, N, -1)
        logits = logits.view(T, N, -1)

        action_loss = cross_entropy_loss(
            logits.permute(0, 2, 1), actions_batch.squeeze(-1)
        )
        entropy_term = dist_entropy * self.entropy_coef

        inflections_batch = task["observations"][
            "inflection_weight"
        ].view(T, N, -1)

        action_loss_term = (
                (inflections_batch * action_loss.unsqueeze(-1)).sum(0)
                / inflections_batch.sum(0)
        ).mean()

        total_loss = action_loss_term - entropy_term

        return (
            total_loss,
            rnn_hidden_states,
            dist_entropy,
            action_loss_term)

    def inner_update(self, rollouts) -> Tuple[float, torch.Tensor, float, float, list]:
        total_loss_inner = 0.0
        total_entropy_inner = 0.0
        total_action_loss_inner = 0.0

        profiling_wrapper.range_push("BC.update epoch")
        train_task_generator = rollouts.recurrent_generator(self.num_mini_batch)
        cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")
        hidden_states = []

        for i, task in enumerate(train_task_generator):
            # Reshape to do in a single forward pass for all steps
            learner = self.actor_critic.clone()
            (logits, rnn_hidden_states, dist_entropy) = learner(
                task["observations"],
                task["recurrent_hidden_states"],
                task["prev_actions"],
                task["masks"],
            )

            N = task["recurrent_hidden_states"].shape[0]
            T = task["actions"].shape[0] // N
            actions_batch = task["actions"].view(T, N, -1)
            logits = logits.view(T, N, -1)

            action_loss = cross_entropy_loss(
                logits.permute(0, 2, 1), actions_batch.squeeze(-1)
            )
            entropy_term = dist_entropy * self.entropy_coef

            inflections_batch = task["observations"][
                "inflection_weight"
            ].view(T, N, -1)

            action_loss_term = (
                    (inflections_batch * action_loss.unsqueeze(-1)).sum(0)
                    / inflections_batch.sum(0)
            ).mean()
            total_loss = action_loss_term - entropy_term

            self.before_backward(total_loss)
            self.after_backward(total_loss)

            self.before_step()
            learner.adapt(total_loss)
            self.after_step()

            total_loss_inner += total_loss
            total_action_loss_inner += action_loss_term
            total_entropy_inner += dist_entropy
            hidden_states.append(rnn_hidden_states)

        profiling_wrapper.range_pop()

        hidden_states = torch.cat(hidden_states, dim=0).detach()

        total_loss_inner /= self.num_mini_batch
        total_entropy_inner /= self.num_mini_batch
        total_action_loss_inner /= self.num_mini_batch

        return (
            total_loss_inner,
            hidden_states,
            total_entropy_inner,
            total_action_loss_inner,
        )

    def outer_update(self, rollouts) -> Tuple[float, torch.Tensor, float, float]:
        total_loss_outer = 0.0
        total_entropy_outer = 0.0
        total_action_loss_outer = 0.0
        torch.autograd.set_detect_anomaly(True)
        profiling_wrapper.range_push("BC.update epoch")
        valid_task_generator = rollouts.recurrent_generator(self.num_mini_batch)
        cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")
        hidden_states = []
        total_loss_outer = []
        # used to store adapted parameters

        for i, task in enumerate(valid_task_generator):
            # Reshape to do in a single forward pass for all steps
            (logits, rnn_hidden_states, dist_entropy) = fmodel(
                task["observations"],
                task["recurrent_hidden_states"],
                task["prev_actions"],
                task["masks"],
            )

            N = task["recurrent_hidden_states"].shape[0]
            T = task["actions"].shape[0] // N
            actions_batch = task["actions"].view(T, N, -1)
            logits = logits.view(T, N, -1)

            action_loss = cross_entropy_loss(
                logits.permute(0, 2, 1), actions_batch.squeeze(-1)
            )
            entropy_term = dist_entropy * self.entropy_coef

            inflections_batch = task["observations"][
                "inflection_weight"
            ].view(T, N, -1)

            action_loss_term = (
                    (inflections_batch * action_loss.unsqueeze(-1)).sum(0)
                    / inflections_batch.sum(0)
            ).mean()
            total_loss = action_loss_term - entropy_term

            total_loss_outer.append(total_loss)
            total_action_loss_outer += action_loss_term.item()
            total_entropy_outer += dist_entropy.item()
            hidden_states.append(rnn_hidden_states)

        # Optimize model
        total_loss_outer = torch.stack(total_loss_outer).mean()

        self.before_backward(total_loss_outer)
        total_loss_outer.backward()
        self.after_backward(total_loss_outer)

        self.outer_optimizer.zero_grad()
        self.before_step()
        self.outer_optimizer.step()
        self.after_step()

        profiling_wrapper.range_pop()

        hidden_states = torch.cat(hidden_states, dim=0).detach()
        total_loss_outer = total_loss_outer.item()
        total_loss_outer /= self.num_mini_batch
        total_entropy_outer /= self.num_mini_batch
        total_action_loss_outer /= self.num_mini_batch

        return (
            total_loss_outer,
            hidden_states,
            total_entropy_outer,
            total_action_loss_outer,
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


class DDPMILAgent(DecentralizedDistributedMixin, MILAgent):
    pass
