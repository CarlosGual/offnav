#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import random
import time
from collections import defaultdict, deque
from distutils.command.config import config
from typing import Any, Dict, List
import copy
import numpy as np
import torch
import tqdm
import wandb
from gym import spaces

from habitat import Config, logger
from habitat.utils import profiling_wrapper
from habitat.utils.env_utils import construct_envs
from habitat.utils.render_wrapper import overlay_frame
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter, get_writer
from offnav.common.ddp_utils import (
    EXIT,
    add_signal_handlers,
    init_distrib_tsubame,
    is_slurm_batch_job,
    load_resume_state,
    rank0_only,
    requeue_job,
    save_resume_state, is_tsubame_batch_job,
)
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from habitat_baselines.utils.common import (
    ObservationBatchingCache,
    action_array_to_dict,
    batch_obs,
    generate_video,
    get_num_actions,
    is_continuous_action_space,
    linear_decay,
)
from torch import nn as nn
from torch.optim.lr_scheduler import LambdaLR

from offnav.algos.agent import DDPILAgent
from offnav.algos.meta_agent import DDPMILAgent
from offnav.common.rollout_storage import ILRolloutStorage, MILRolloutStorage
from offnav.envs.env_utils import construct_meta_envs
from habitat.core.environments import get_env_class
from offnav.envs.meta_vector_env import MetaVectorEnv


@baseline_registry.register_trainer(name="pirlnav-mil")
class MILEnvDDPTrainer(PPOTrainer):
    envs: MetaVectorEnv

    def __init__(self, config=None):
        super().__init__(config)

    def _init_envs(self, config=None):
        if config is None:
            config = self.config

        self.envs = construct_meta_envs(
            config,
            get_env_class(config.ENV_NAME),
            workers_ignore_signals=is_tsubame_batch_job(),
        )

    def _init_eval_envs(self, config=None):
        if config is None:
            config = self.config

        self.envs = construct_envs(
            config,
            get_env_class("SimpleRLEnv"),
            workers_ignore_signals=is_tsubame_batch_job(),
        )

    def _setup_actor_critic_agent(self, il_cfg: Config) -> None:
        r"""Sets up actor critic and agent for IL.

        Args:
            il_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        observation_space = self.envs.observation_spaces[0]
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )
        self.obs_space = observation_space

        policy = baseline_registry.get_policy(self.config.IL.POLICY.name)
        self.actor_critic = policy.from_config(
            self.config, observation_space, self.envs.action_spaces[0]
        )
        self.actor_critic.to(self.device)

        self.agent = DDPMILAgent(
            actor_critic=self.actor_critic,
            num_envs=self.envs.num_envs,
            num_mini_batch=il_cfg.num_mini_batch,
            inner_lr=il_cfg.lr,
            outer_lr=il_cfg.lr,
            outer_encoder_lr=il_cfg.encoder_lr,
            eps=il_cfg.eps,
            max_grad_norm=il_cfg.max_grad_norm,
            wd=il_cfg.wd,
            entropy_coef=il_cfg.entropy_coef,
        )

    def sample_and_set_tasks(self, num_tasks: int):
        tasks = self.envs.sample_tasks(num_tasks)
        self.envs.set_tasks(tasks)

        observations = self.envs.reset()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        if self._static_encoder:
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        self.rollouts.buffers["observations"][0] = batch  # type: ignore

    def _init_train(self):
        resume_state = load_resume_state(self.config)
        if resume_state is not None:
            if self.config.OVERWRITE_NUM_UPDATES:
                num_updates = self.config.NUM_UPDATES
                self.config: Config = resume_state["config"]
                self.config.defrost()
                self.config.NUM_UPDATES = num_updates
                self.config.freeze()
            else:
                self.config: Config = resume_state["config"]

        if self.config.RL.DDPPO.force_distributed:
            self._is_distributed = True

        if is_slurm_batch_job():
            add_signal_handlers()

        # Add replay sensors
        self.config.defrost()
        self.config.TASK_CONFIG.TASK.SENSORS.extend(
            ["DEMONSTRATION_SENSOR", "INFLECTION_WEIGHT_SENSOR"]
        )
        self.config.freeze()

        if self._is_distributed:
            local_rank, tcp_store = init_distrib_tsubame(
                self.config.RL.DDPPO.distrib_backend
            )
            if rank0_only():
                logger.info(
                    "Initialized DD-PPO with {} workers".format(
                        torch.distributed.get_world_size()
                    )
                )

            self.config.defrost()
            self.config.TORCH_GPU_ID = local_rank
            self.config.SIMULATOR_GPU_ID = local_rank
            # Multiply by the number of simulators to make sure they also get unique seeds
            self.config.TASK_CONFIG.SEED += (
                    torch.distributed.get_rank() * self.config.NUM_ENVIRONMENTS
            )
            self.config.freeze()

            random.seed(self.config.TASK_CONFIG.SEED)
            np.random.seed(self.config.TASK_CONFIG.SEED)
            torch.manual_seed(self.config.TASK_CONFIG.SEED)
            self.num_rollouts_done_store = torch.distributed.PrefixStore(
                "rollout_tracker", tcp_store
            )
            self.num_rollouts_done_store.set("num_done", "0")

        if rank0_only() and self.config.VERBOSE:
            logger.info(f"config: {self.config}")

        profiling_wrapper.configure(
            capture_start_step=self.config.PROFILING.CAPTURE_START_STEP,
            num_steps_to_capture=self.config.PROFILING.NUM_STEPS_TO_CAPTURE,
        )

        self._init_envs()

        action_space = self.envs.action_spaces[0]
        self.policy_action_space = action_space

        if is_continuous_action_space(action_space):
            # Assume ALL actions are NOT discrete
            action_shape = (get_num_actions(action_space),)
            discrete_actions = False
        else:
            # For discrete pointnav
            action_shape = None
            discrete_actions = True

        il_cfg = self.config.IL.BehaviorCloning
        meta_cfg = self.config.META.MIL

        policy_cfg = self.config.POLICY
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.config.TORCH_GPU_ID)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        if rank0_only() and not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        self._setup_actor_critic_agent(il_cfg)
        if self._is_distributed:
            self.agent.init_distributed(find_unused_params=True)  # type: ignore

        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        obs_space = self.obs_space
        if self._static_encoder:
            self._encoder = self.actor_critic.net.visual_encoder
            obs_space = spaces.Dict(
                {
                    "visual_features": spaces.Box(
                        low=np.finfo(np.float32).min,
                        high=np.finfo(np.float32).max,
                        shape=self._encoder.output_shape,
                        dtype=np.float32,
                    ),
                    **obs_space.spaces,
                }
            )

        self._nbuffers = 2 if il_cfg.use_double_buffered_sampler else 1

        self.rollouts = MILRolloutStorage(
            meta_cfg.num_gradient_updates,
            il_cfg.num_steps,
            self.envs.num_envs,
            obs_space,
            self.policy_action_space,
            policy_cfg.STATE_ENCODER.hidden_size,
            num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
            is_double_buffered=il_cfg.use_double_buffered_sampler,
            action_shape=action_shape,
            discrete_actions=discrete_actions,
        )
        self.rollouts.to(self.device)
        assert il_cfg.num_mini_batch == self.config.META.MIL.num_tasks, "Number of tasks must be equal to the number of environments divided by the number of mini-batches"
        num_tasks = self.config.META.MIL.num_tasks
        self.sample_and_set_tasks(num_tasks)

        self.current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        self.running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        self.window_episode_stats = defaultdict(
            lambda: deque(maxlen=il_cfg.reward_window_size)
        )

        self.env_time = 0.0
        self.pth_time = 0.0
        self.t_start = time.time()
        return resume_state

    def _compute_actions_and_step_envs(self, buffer_index: int = 0):
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._nbuffers),
            int((buffer_index + 1) * num_envs / self._nbuffers),
        )

        t_sample_action = time.time()

        # fetch actions from replay buffer
        step_batch = self.rollouts.buffers[
            self.rollouts.current_rollout_step_idxs[buffer_index],
            env_slice,
        ]
        next_actions = step_batch["observations"]["next_actions"]
        actions = next_actions.long().unsqueeze(-1)

        # NB: Move actions to CPU.  If CUDA tensors are
        # sent in to env.step(), that will create CUDA contexts
        # in the subprocesses.
        # For backwards compatibility, we also call .item() to convert to
        # an int
        actions = actions.to(device="cpu")
        self.pth_time += time.time() - t_sample_action

        profiling_wrapper.range_pop()  # compute actions

        t_step_env = time.time()

        for index_env, act in zip(
                range(env_slice.start, env_slice.stop), actions.unbind(0)
        ):
            if act.shape[0] > 1:
                step_action = action_array_to_dict(
                    self.policy_action_space, act
                )
            else:
                step_action = act.item()
            self.envs.async_step_at(index_env, step_action)

        self.env_time += time.time() - t_step_env

        self.rollouts.insert(
            actions=actions,
            buffer_index=buffer_index,
        )

    def _make_rollouts(self, il_cfg: Config, meta_cfg: Config, count_steps_delta: int = 0, eval: bool = False):
        profiling_wrapper.range_push("inner rollouts loop")
        profiling_wrapper.range_push("_collect_rollout_step")
        if eval:
            extra_loop = 0
        else:
            extra_loop = 1
        self.agent.eval()
        for buffer_index in range(self._nbuffers):
            self._compute_actions_and_step_envs(buffer_index)

        for step in range(il_cfg.num_steps * (meta_cfg.num_gradient_updates + extra_loop)):
            is_last_step = (
                    self.should_end_early(step + 1)
                    or (step + 1) == il_cfg.num_steps * (meta_cfg.num_gradient_updates + extra_loop)
            )

            for buffer_index in range(self._nbuffers):
                count_steps_delta += self._collect_environment_result(
                    buffer_index
                )

                if (buffer_index + 1) == self._nbuffers:
                    profiling_wrapper.range_pop()  # _collect_rollout_step

                if not is_last_step:
                    if (buffer_index + 1) == self._nbuffers:
                        profiling_wrapper.range_push(
                            "_collect_rollout_step"
                        )

                    self._compute_actions_and_step_envs(buffer_index)

            if is_last_step:
                break

            profiling_wrapper.range_pop()  # rollouts loop

        return count_steps_delta

    @profiling_wrapper.RangeContext("train")
    def train(self) -> None:
        r"""Main method for training DD/PPO.

        Returns:
            None
        """

        resume_state = self._init_train()

        count_checkpoints = 0
        prev_time = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.outer_optimizer,
            lr_lambda=lambda x: 1 - self.percent_done(),
        )

        # resume_state = load_resume_state(self.config)
        if resume_state is not None:
            self.agent.load_state_dict(resume_state["state_dict"])
            self.agent.outer_optimizer.load_state_dict(resume_state["optim_state"])
            lr_scheduler.load_state_dict(resume_state["lr_sched_state"])

            requeue_stats = resume_state["requeue_stats"]
            self.env_time = requeue_stats["env_time"]
            self.pth_time = requeue_stats["pth_time"]
            self.num_steps_done = requeue_stats["num_steps_done"]
            self.num_updates_done = requeue_stats["num_updates_done"]
            self._last_checkpoint_percent = requeue_stats[
                "_last_checkpoint_percent"
            ]
            count_checkpoints = requeue_stats["count_checkpoints"]
            prev_time = requeue_stats["prev_time"]

            self.running_episode_stats = requeue_stats["running_episode_stats"]
            self.window_episode_stats.update(
                requeue_stats["window_episode_stats"]
            )

        il_cfg = self.config.IL.BehaviorCloning
        meta_cfg = self.config.META.MIL
        iter_sampled_tasks = 0

        with (
                get_writer(self.config, flush_secs=self.flush_secs)
                if rank0_only()
                else contextlib.suppress()
        ) as writer:
            while not self.is_done():
                profiling_wrapper.on_start_step()
                profiling_wrapper.range_push("train update")

                if il_cfg.use_linear_clip_decay:
                    self.agent.clip_param = il_cfg.clip_param * (
                            1 - self.percent_done()
                    )

                if rank0_only() and self._should_save_resume_state():
                    requeue_stats = dict(
                        env_time=self.env_time,
                        pth_time=self.pth_time,
                        count_checkpoints=count_checkpoints,
                        num_steps_done=self.num_steps_done,
                        num_updates_done=self.num_updates_done,
                        _last_checkpoint_percent=self._last_checkpoint_percent,
                        prev_time=(time.time() - self.t_start) + prev_time,
                        running_episode_stats=self.running_episode_stats,
                        window_episode_stats=dict(self.window_episode_stats),
                    )

                    save_resume_state(
                        dict(
                            state_dict=self.agent.state_dict(),
                            optim_state=self.agent.outer_optimizer.state_dict(),
                            lr_sched_state=lr_scheduler.state_dict(),
                            config=self.config,
                            requeue_stats=requeue_stats,
                        ),
                        self.config,
                    )

                if EXIT.is_set():
                    profiling_wrapper.range_pop()  # train update

                    self.envs.close()

                    requeue_job()

                    return

                count_steps_delta = 0

                # Make several gradient updates so the agent can adapt to trajectories that make it to the goal
                # INNER UPDATE LOOP
                total_inner_action_loss = 0.0
                total_inner_dist_entropy = 0.0
                total_inner_total_loss = 0.0
                total_outer_total_loss = 0.0
                total_outer_dist_entropy = 0.0
                total_outer_action_loss = 0.0
                count_steps_delta = self._make_rollouts(il_cfg, meta_cfg, count_steps_delta)
                adapt_tasks_generator = self.rollouts.adapt_recurrent_generator(il_cfg.num_mini_batch)
                valid_tasks_generator = self.rollouts.valid_recurrent_generator(il_cfg.num_mini_batch)
                hidden_states = []

                self.agent.actor_critic.train()

                learner = self.agent.actor_critic.clone()

                for adapt_task, valid_task in zip(adapt_tasks_generator, valid_tasks_generator):

                    for num_gradient_update, task in enumerate(adapt_task):

                        if num_gradient_update == 0: # If it is the first iteration, we don't have hidden states
                            (
                                inner_total_loss,
                                rnn_hidden_states,
                                inner_dist_entropy,
                                inner_action_loss
                            ) = self.agent.calculate_loss(task, learner, hidden_states=None)

                        elif num_gradient_update > 0:
                            (
                                inner_total_loss,
                                rnn_hidden_states,
                                inner_dist_entropy,
                                inner_action_loss
                            ) = self.agent.calculate_loss(task, learner, hidden_states=rnn_hidden_states)

                        learner.adapt(inner_total_loss)

                        total_inner_action_loss += inner_action_loss
                        total_inner_dist_entropy += inner_dist_entropy
                        total_inner_total_loss += inner_total_loss

                    (
                        outer_total_loss,
                        rnn_hidden_states,
                        outer_dist_entropy,
                        outer_action_loss
                    ) = self.agent.calculate_valid_loss(valid_task, learner, rnn_hidden_states.detach())
                    hidden_states.append(rnn_hidden_states)

                total_outer_total_loss += outer_total_loss
                total_outer_dist_entropy += outer_dist_entropy
                total_outer_action_loss += outer_action_loss
                hidden_states = torch.cat(hidden_states, dim=0).detach()
                self.rollouts.after_update(hidden_states)
                total_outer_total_loss /= self.config.NUM_ENVIRONMENTS

                # OUTER UPDATE
                self.agent.outer_optimizer.zero_grad()
                total_outer_total_loss.backward()
                self.agent.outer_optimizer.step()

                # Sample new tasks for the next update.
                # Don't do it all the time, so we can explore a little bit
                # more the tasks, since these are challenging tasks.
                iter_sampled_tasks += 1
                if iter_sampled_tasks % self.config.META.MIL.num_updates_per_sampled_tasks == 0:
                    logger.info('----------------- NEW SAMPLE ----------------------')
                    self.sample_and_set_tasks(self.config.META.MIL.num_tasks)
                    iter_sampled_tasks = 0

                if self._is_distributed:
                    self.num_rollouts_done_store.add("num_done", self.config.META.MIL.num_gradient_updates + 1)

                logger.info('----------------- New batch ----------------------')
                for i, episode in enumerate(self.envs.current_episodes()):
                    logger.info(
                        "Environment: {}, Current scene: {}, Current goal {}, Current task_id {}, episode: {}, "
                        "length: {}".format(
                            i,
                            episode.scene_id.split(".")[-3].split("/")[-1],
                            episode.object_category,
                            episode.goals_key,
                            episode.episode_id,
                            len(episode.reference_replay)
                        )
                    )

                if il_cfg.use_linear_lr_decay:
                    lr_scheduler.step()  # type: ignore

                self.num_updates_done += 1
                losses = self._coalesce_post_step(
                    dict(
                        inner_action_loss=inner_action_loss.detach(),
                        inner_entropy=inner_dist_entropy.detach(),
                        outer_action_loss=outer_action_loss.detach(),
                        outer_entropy=outer_dist_entropy.detach(),
                    ),
                    count_steps_delta,
                )

                self._training_log(writer, losses, prev_time)

                # checkpoint model
                if rank0_only() and self.should_checkpoint():
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth",
                        dict(
                            step=self.num_steps_done,
                            wall_time=(time.time() - self.t_start) + prev_time,
                        ),
                    )
                    count_checkpoints += 1

                profiling_wrapper.range_pop()  # train update

            self.envs.close()

    @rank0_only
    def _training_log(
            self, writer, losses: Dict[str, float], prev_time: int = 0
    ):
        deltas = {
            k: (
                (v[-1] - v[0]).sum().item()
                if len(v) > 1
                else v[0].sum().item()
            )
            for k, v in self.window_episode_stats.items()
        }
        deltas["count"] = max(deltas["count"], 1.0)

        writer.add_scalar(
            "reward",
            deltas["reward"] / deltas["count"],
            self.num_steps_done,
        )

        # Check to see if there are any metrics
        # that haven't been logged yet
        metrics = {
            k: v / deltas["count"]
            for k, v in deltas.items()
            if k not in {"reward", "count"}
        }

        for k, v in metrics.items():
            writer.add_scalar(f"metrics/{k}", v, self.num_steps_done)
        for k, v in losses.items():
            writer.add_scalar(f"losses/{k}", v, self.num_steps_done)

        fps = self.num_steps_done / ((time.time() - self.t_start) + prev_time)
        writer.add_scalar("metrics/fps", fps, self.num_steps_done)

        # log stats
        if self.num_updates_done % self.config.LOG_INTERVAL == 0:
            logger.info(
                "update: {}\tfps: {:.3f}\t".format(
                    self.num_updates_done,
                    fps,
                )
            )

            logger.info(
                "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                "frames: {}".format(
                    self.num_updates_done,
                    self.env_time,
                    self.pth_time,
                    self.num_steps_done,
                )
            )

            logger.info(
                "Average window size: {}  {}  {}".format(
                    len(self.window_episode_stats["count"]),
                    "  ".join(
                        "{}: {:.3f}".format(k, v / deltas["count"])
                        for k, v in deltas.items()
                        if k != "count"
                    ),
                    "  ".join(
                        "{}: {:.3f}".format(k, v) for k, v in losses.items()
                    ),
                )
            )

    def _eval_checkpoint(
            self,
            checkpoint_path: str,
            writer: TensorboardWriter,
            checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        self._is_distributed = False
        if self._is_distributed:
            raise RuntimeError("Evaluation does not support distributed mode")

        # Map location CPU is almost always better than mapping to a CUDA device.
        if self.config.EVAL.SHOULD_LOAD_CKPT:
            ckpt_dict = self.load_checkpoint(
                checkpoint_path, map_location="cpu"
            )
        else:
            ckpt_dict = {}

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        if (
                len(self.config.VIDEO_OPTION) > 0
                and self.config.VIDEO_RENDER_TOP_DOWN
        ):
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        if config.VERBOSE:
            logger.info(f"env config: {config}")

        self._init_eval_envs(config)

        action_space = self.envs.action_spaces[0]
        if self.using_velocity_ctrl:
            # For navigation using a continuous action space for a task that
            # may be asking for discrete actions
            self.policy_action_space = action_space["VELOCITY_CONTROL"]
            action_shape = (2,)
            discrete_actions = False
        else:
            self.policy_action_space = action_space
            if is_continuous_action_space(action_space):
                # Assume NONE of the actions are discrete
                action_shape = (get_num_actions(action_space),)
                discrete_actions = False
            else:
                # For discrete pointnav
                action_shape = (1,)
                discrete_actions = True

        il_cfg = config.IL.BehaviorCloning
        policy_cfg = config.POLICY
        meta_cfg = self.config.META.MIL
        self._setup_actor_critic_agent(il_cfg)

        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic

        self._nbuffers = 2 if il_cfg.use_double_buffered_sampler else 1

        obs_space = self.obs_space
        if self._static_encoder:
            self._encoder = self.actor_critic.net.visual_encoder
            obs_space = spaces.Dict(
                {
                    "visual_features": spaces.Box(
                        low=np.finfo(np.float32).min,
                        high=np.finfo(np.float32).max,
                        shape=self._encoder.output_shape,
                        dtype=np.float32,
                    ),
                    **obs_space.spaces,
                }
            )

        self.rollouts = MILRolloutStorage(
            meta_cfg.num_gradient_updates,
            il_cfg.num_steps,
            self.envs.num_envs,
            obs_space,
            self.policy_action_space,
            policy_cfg.STATE_ENCODER.hidden_size,
            num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
            is_double_buffered=il_cfg.use_double_buffered_sampler,
            action_shape=action_shape,
            discrete_actions=discrete_actions,
        )
        self.rollouts.to(self.device)

        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode

        rgb_frames = [
            [] for _ in range(self.config.NUM_ENVIRONMENTS)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        number_of_eval_episodes = self.config.TEST_EPISODE_COUNT
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            if total_num_eps < number_of_eval_episodes:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps

        pbar = tqdm.tqdm(total=number_of_eval_episodes)
        logger.info("Sampling actions deterministically...")
        while (
                len(stats_episodes) < number_of_eval_episodes
                and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()
            _ = self._make_rollouts(il_cfg, meta_cfg, 0, eval=True)
            adapt_tasks_generator = self.rollouts.adapt_recurrent_generator(il_cfg.num_mini_batch)
            learner = self.actor_critic.clone()
            for adapt_task in adapt_tasks_generator:
                for num_gradient_update, task in enumerate(adapt_task):

                    if num_gradient_update == 0:  # If it is the first iteration, we don't have hidden states
                        (
                            inner_total_loss,
                            rnn_hidden_states,
                            inner_dist_entropy,
                            inner_action_loss
                        ) = self.agent.calculate_loss(task, learner, hidden_states=None)

                    elif num_gradient_update > 0:
                        (
                            inner_total_loss,
                            rnn_hidden_states,
                            inner_dist_entropy,
                            inner_action_loss
                        ) = self.agent.calculate_loss(task, learner, hidden_states=rnn_hidden_states)

                    learner.adapt(inner_total_loss)

            with torch.no_grad():
                (
                    actions,
                    test_recurrent_hidden_states,
                ) = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=True,
                )

                prev_actions.copy_(actions)  # type: ignore
            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            # For backwards compatibility, we also call .item() to convert to
            # an int
            if actions[0].shape[0] > 1:
                step_data = [
                    action_array_to_dict(self.policy_action_space, a)
                    for a in actions.to(device="cpu")
                ]
            else:
                step_data = [a.item() for a in actions.to(device="cpu")]

            outputs = self.envs.step(step_data)

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            batch = batch_obs(  # type: ignore
                observations,
                device=self.device,
                cache=self._obs_batching_cache,
            )
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            )

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                        next_episodes[i].scene_id,
                        next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not not_done_masks[i].item():
                    pbar.update()
                    episode_stats = {
                        "reward": current_episode_reward[i].item()
                    }
                    episode_stats.update(
                        self._extract_scalars_from_info(infos[i])
                    )
                    current_episode_reward[i] = 0
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats

                    if len(self.config.VIDEO_OPTION) > 0:
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i],
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=self._extract_scalars_from_info(infos[i]),
                            fps=self.config.VIDEO_FPS,
                            tb_writer=writer,
                            keys_to_include_in_name=self.config.EVAL_KEYS_TO_INCLUDE_IN_NAME,
                        )

                        rgb_frames[i] = []

                # episode continues
                elif len(self.config.VIDEO_OPTION) > 0:
                    # TODO move normalization / channel changing out of the policy and undo it here
                    frame = observations_to_image(
                        {k: v[i] for k, v in batch.items()}, infos[i]
                    )
                    if self.config.VIDEO_RENDER_ALL_INFO:
                        frame = overlay_frame(frame, infos[i])

                    rgb_frames[i].append(frame)

            not_done_masks = not_done_masks.to(device=self.device)
            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        num_episodes = len(stats_episodes)
        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                    sum(v[stat_key] for v in stats_episodes.values())
                    / num_episodes
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalar(
            "eval_reward/average_reward", aggregated_stats["reward"], step_id
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)

        self.envs.close()
