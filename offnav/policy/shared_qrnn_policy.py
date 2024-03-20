import torch
import torch.nn as nn
from gym import Space
from habitat import Config, logger
from habitat.tasks.nav.nav import EpisodicCompassSensor, EpisodicGPSSensor
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import build_rnn_state_encoder
from habitat_baselines.rl.ppo import Net

from offnav.policy.policy import IQLRNNPolicy, CriticHead, MLPCriticHead
from offnav.policy.transforms import get_transform
from offnav.policy.visual_encoder import VisualEncoder
from offnav.utils.utils import load_encoder


class RGBEncoder(Net):
    """
    Encoder class for be shared between all nets. Powerful one.
    """

    def __init__(
            self,
            policy_config: Config,
            run_type: str,
    ):
        super().__init__()
        self.policy_config = policy_config

        rgb_config = policy_config.RGB_ENCODER
        name = "resize"
        if rgb_config.use_augmentations and run_type == "train":
            name = rgb_config.augmentations_name
        if rgb_config.use_augmentations_test_time and run_type == "eval":
            name = rgb_config.augmentations_name
        self.visual_transform = get_transform(name, size=rgb_config.image_size)
        self.visual_transform.randomize_environments = (
            rgb_config.randomize_augmentations_over_envs
        )

        logger.info("******************************** Loading module: {} ********************************".format(
            self.__class__.__name__))

        self.visual_encoder = VisualEncoder(
            image_size=rgb_config.image_size,
            backbone=rgb_config.backbone,
            input_channels=3,
            resnet_baseplanes=rgb_config.resnet_baseplanes,
            resnet_ngroups=rgb_config.resnet_baseplanes // 2,
            avgpooled_image=rgb_config.avgpooled_image,
            drop_path_rate=rgb_config.drop_path_rate,
        )

        # load pretrained weights
        if rgb_config.pretrained_encoder is not None:
            msg = load_encoder(
                self.visual_encoder, rgb_config.pretrained_encoder
            )
            logger.info(
                "Using weights from {}: {}".format(
                    rgb_config.pretrained_encoder, msg
                )
            )

        # freeze backbone
        if rgb_config.freeze_backbone:
            for p in self.visual_encoder.backbone.parameters():
                p.requires_grad = False

        logger.info(
            "RGB encoder is {}".format(policy_config.RGB_ENCODER.backbone)
        )

    @property
    def output_size(self):
        return self.rnn_input_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind and self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return None

    def forward(self, observations, rnn_hidden_states):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """
        rgb_obs = observations["rgb"]

        N = rnn_hidden_states.size(1)

        if self.visual_encoder is not None:
            if len(rgb_obs.size()) == 5:
                observations["rgb"] = rgb_obs.contiguous().view(
                    -1, rgb_obs.size(2), rgb_obs.size(3), rgb_obs.size(4)
                )
            # visual encoder
            rgb = observations["rgb"]
            rgb = self.visual_transform(rgb, N)
            rgb = self.visual_encoder(rgb)

        return rgb


class SharedPolicyNet(Net):
    r"""A baseline sequence to sequence network that concatenates instruction,
    RGB, and depth encodings before decoding an action distribution with an RNN.
    Modules:
        Instruction encoder
        Depth encoder
        RGB encoder
        RNN state encoder
    """

    def __init__(
            self,
            observation_space: Space,
            policy_config: Config,
            num_actions: int,
            run_type: str,
            use_actions: bool,
            hidden_size: int,
            rnn_type: str,
            num_recurrent_layers: int,
            is_policy: bool = False,
    ):
        super().__init__()
        self.policy_config = policy_config
        self.use_actions = use_actions
        self.is_policy = is_policy
        self.shared_encoder = RGBEncoder(
            policy_config=policy_config,
            run_type=run_type
        )
        rnn_input_size = 0
        logger.info("******************************** Loading module: {} ********************************".format(
            self.__class__.__name__))

        rnn_input_size += policy_config.RGB_ENCODER.hidden_size

        self.visual_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                self.shared_encoder.visual_encoder.output_size,
                policy_config.RGB_ENCODER.hidden_size,
            ),
            nn.ReLU(True),
        )

        if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
            input_gps_dim = observation_space.spaces[
                EpisodicGPSSensor.cls_uuid
            ].shape[0]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            rnn_input_size += 32
            logger.info("\n\nSetting up GPS sensor")

        if EpisodicCompassSensor.cls_uuid in observation_space.spaces:
            assert (
                    observation_space.spaces[EpisodicCompassSensor.cls_uuid].shape[
                        0
                    ]
                    == 1
            ), "Expected compass with 2D rotation."
            input_compass_dim = 2  # cos and sin of the angle
            self.compass_embedding_dim = 32
            self.compass_embedding = nn.Linear(
                input_compass_dim, self.compass_embedding_dim
            )
            rnn_input_size += 32
            logger.info("\n\nSetting up Compass sensor")

        if ObjectGoalSensor.cls_uuid in observation_space.spaces:
            self._n_object_categories = (
                    int(
                        observation_space.spaces[ObjectGoalSensor.cls_uuid].high[0]
                    )
                    + 1
            )
            logger.info(
                "Object categories: {}".format(self._n_object_categories)
            )
            self.obj_categories_embedding = nn.Embedding(
                self._n_object_categories, 32
            )
            rnn_input_size += 32
            logger.info("\n\nSetting up Object Goal sensor")

        if policy_config.SEQ2SEQ.use_action and self.use_actions:
            self.action_embedding = nn.Embedding(num_actions + 1, 32)
            rnn_input_size += self.action_embedding.embedding_dim

        if policy_config.SEQ2SEQ.use_action and self.is_policy:
            self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)
            rnn_input_size += self.prev_action_embedding.embedding_dim

        self._rnn_input_size = rnn_input_size

        logger.info(
            "State enc: {} - {} - {} - {}".format(
                rnn_input_size, hidden_size, rnn_type, num_recurrent_layers
            )
        )

        self.state_encoder = build_rnn_state_encoder(
            rnn_input_size,
            hidden_size=hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )
        self._hidden_size = hidden_size
        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.shared_encoder.visual_encoder.is_blind and self.shared_encoder.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, actions, prev_actions, masks):

        x = []
        # Maybe try to pass it as an argument to forward function, but should be okey
        rgb = self.shared_encoder(observations, rnn_hidden_states)
        rgb = self.visual_fc(rgb)
        x.append(rgb)

        if EpisodicGPSSensor.cls_uuid in observations:
            obs_gps = observations[EpisodicGPSSensor.cls_uuid]
            if len(obs_gps.size()) == 3:
                obs_gps = obs_gps.contiguous().view(-1, obs_gps.size(2))
            x.append(self.gps_embedding(obs_gps))

        if EpisodicCompassSensor.cls_uuid in observations:
            obs_compass = observations["compass"]
            if len(obs_compass.size()) == 3:
                obs_compass = obs_compass.contiguous().view(
                    -1, obs_compass.size(2)
                )
            compass_observations = torch.stack(
                [
                    torch.cos(obs_compass),
                    torch.sin(obs_compass),
                ],
                -1,
            )
            compass_embedding = self.compass_embedding(
                compass_observations.float().squeeze(dim=1)
            )
            x.append(compass_embedding)

        if ObjectGoalSensor.cls_uuid in observations:
            object_goal = observations[ObjectGoalSensor.cls_uuid].long()
            if len(object_goal.size()) == 3:
                object_goal = object_goal.contiguous().view(
                    -1, object_goal.size(2)
                )
            x.append(self.obj_categories_embedding(object_goal).squeeze(dim=1))

        if self.policy_config.SEQ2SEQ.use_action and self.use_actions:
            actions_embedding = self.action_embedding(
                ((actions.float() + 1) * masks).long().squeeze(dim=-1)
            )
            x.append(actions_embedding)

        if self.policy_config.SEQ2SEQ.use_action and self.is_policy:
            prev_actions_embedding = self.prev_action_embedding(
                ((prev_actions.float() + 1) * masks).long().squeeze(dim=-1)
            )
            x.append(prev_actions_embedding)

        x = torch.cat(x, dim=1)

        x, rnn_hidden_states = self.state_encoder(
            x, rnn_hidden_states.contiguous(), masks
        )

        return x, rnn_hidden_states


class SharedQNet(Net):
    r"""A baseline sequence to sequence network that concatenates instruction,
    RGB, and depth encodings before decoding an action distribution with an RNN.
    Modules:
        Instruction encoder
        Depth encoder
        RGB encoder
        RNN state encoder
    """

    def __init__(
            self,
            observation_space: Space,
            policy_config: Config,
            num_actions: int,
            shared_encoder,
            use_actions: bool,
            hidden_size: int,
            rnn_type: str,
            num_recurrent_layers: int,
            is_policy: bool = False,
            mlp_critic=False,
            critic_hidden_dim=512,
    ):
        super().__init__()
        self.policy_config = policy_config
        self.use_actions = use_actions
        self.is_policy = is_policy
        self.shared_encoder = shared_encoder

        rnn_input_size = 0
        logger.info("******************************** Loading module: {} ********************************".format(
            self.__class__.__name__))

        rnn_input_size += policy_config.RGB_ENCODER.hidden_size

        self.visual_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                self.shared_encoder.visual_encoder.output_size,
                policy_config.RGB_ENCODER.hidden_size,
            ),
            nn.ReLU(True),
        )

        if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
            input_gps_dim = observation_space.spaces[
                EpisodicGPSSensor.cls_uuid
            ].shape[0]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            rnn_input_size += 32
            logger.info("\n\nSetting up GPS sensor")

        if EpisodicCompassSensor.cls_uuid in observation_space.spaces:
            assert (
                    observation_space.spaces[EpisodicCompassSensor.cls_uuid].shape[
                        0
                    ]
                    == 1
            ), "Expected compass with 2D rotation."
            input_compass_dim = 2  # cos and sin of the angle
            self.compass_embedding_dim = 32
            self.compass_embedding = nn.Linear(
                input_compass_dim, self.compass_embedding_dim
            )
            rnn_input_size += 32
            logger.info("\n\nSetting up Compass sensor")

        if ObjectGoalSensor.cls_uuid in observation_space.spaces:
            self._n_object_categories = (
                    int(
                        observation_space.spaces[ObjectGoalSensor.cls_uuid].high[0]
                    )
                    + 1
            )
            logger.info(
                "Object categories: {}".format(self._n_object_categories)
            )
            self.obj_categories_embedding = nn.Embedding(
                self._n_object_categories, 32
            )
            rnn_input_size += 32
            logger.info("\n\nSetting up Object Goal sensor")

        if policy_config.SEQ2SEQ.use_action and self.use_actions:
            self.action_embedding = nn.Embedding(num_actions + 1, 32)
            rnn_input_size += self.action_embedding.embedding_dim

        if policy_config.SEQ2SEQ.use_action and self.is_policy:
            self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)
            rnn_input_size += self.prev_action_embedding.embedding_dim

        self._rnn_input_size = rnn_input_size

        logger.info(
            "State enc: {} - {} - {} - {}".format(
                rnn_input_size, hidden_size, rnn_type, num_recurrent_layers
            )
        )

        self.state_encoder = build_rnn_state_encoder(
            rnn_input_size,
            hidden_size=hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )
        self._hidden_size = hidden_size
        if not mlp_critic:
            self.critic = CriticHead(self._hidden_size)
        else:
            self.critic = MLPCriticHead(
                self._hidden_size,
                critic_hidden_dim,
            )
        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.shared_encoder.visual_encoder.is_blind and self.shared_encoder.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, actions, prev_actions, masks):

        x = []
        # Maybe try to pass it as an argument to forward function, but should be okey
        rgb = self.shared_encoder(observations, rnn_hidden_states)
        rgb = self.visual_fc(rgb)
        x.append(rgb)

        if EpisodicGPSSensor.cls_uuid in observations:
            obs_gps = observations[EpisodicGPSSensor.cls_uuid]
            if len(obs_gps.size()) == 3:
                obs_gps = obs_gps.contiguous().view(-1, obs_gps.size(2))
            x.append(self.gps_embedding(obs_gps))

        if EpisodicCompassSensor.cls_uuid in observations:
            obs_compass = observations["compass"]
            if len(obs_compass.size()) == 3:
                obs_compass = obs_compass.contiguous().view(
                    -1, obs_compass.size(2)
                )
            compass_observations = torch.stack(
                [
                    torch.cos(obs_compass),
                    torch.sin(obs_compass),
                ],
                -1,
            )
            compass_embedding = self.compass_embedding(
                compass_observations.float().squeeze(dim=1)
            )
            x.append(compass_embedding)

        if ObjectGoalSensor.cls_uuid in observations:
            object_goal = observations[ObjectGoalSensor.cls_uuid].long()
            if len(object_goal.size()) == 3:
                object_goal = object_goal.contiguous().view(
                    -1, object_goal.size(2)
                )
            x.append(self.obj_categories_embedding(object_goal).squeeze(dim=1))

        if self.policy_config.SEQ2SEQ.use_action and self.use_actions:
            actions_embedding = self.action_embedding(
                ((actions.float() + 1) * masks).long().squeeze(dim=-1)
            )
            x.append(actions_embedding)

        if self.policy_config.SEQ2SEQ.use_action and self.is_policy:
            prev_actions_embedding = self.prev_action_embedding(
                ((prev_actions.float() + 1) * masks).long().squeeze(dim=-1)
            )
            x.append(prev_actions_embedding)

        x = torch.cat(x, dim=1)

        x, rnn_hidden_states = self.state_encoder(
            x, rnn_hidden_states.contiguous(), masks
        )

        value = self.critic(x)
        return value, rnn_hidden_states


# class SharedQNet(SharedPolicyNet):
#     def __init__(
#             self,
#             observation_space: Space,
#             policy_config: Config,
#             num_actions: int,
#             run_type: str,
#             shared_encoder,
#             use_actions: bool = False,
#             hidden_size: int = 512,
#             rnn_type: str = "GRU",
#             num_recurrent_layers: int = 1,
#             mlp_critic=False,
#             critic_hidden_dim=512,
#     ):
#         super().__init__(
#             observation_space=observation_space,
#             policy_config=policy_config,
#             num_actions=num_actions,
#             use_actions=use_actions,
#             run_type=run_type,
#             hidden_size=hidden_size,
#             rnn_type=rnn_type,
#             num_recurrent_layers=num_recurrent_layers,
#         )
#         self.shared_encoder = shared_encoder
#         if not mlp_critic:
#             self.critic = CriticHead(self.output_size)
#         else:
#             self.critic = MLPCriticHead(
#                 self.output_size,
#                 critic_hidden_dim,
#             )
#
#     def forward(self, observations, rnn_hidden_states, actions, prev_actions, masks):
#         r"""
#         instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
#         depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
#         rgb_embedding: [batch_size x RGB_ENCODER.output_size]
#         """
#         features, rnn_hidden_states = super().forward(observations, rnn_hidden_states, actions, prev_actions, masks)
#         value = self.critic(features)
#         return value, rnn_hidden_states


class SharedVNet(Net):
    def __init__(
            self,
            observation_space: Space,
            policy_config: Config,
            shared_encoder,
            output_activation=nn.Identity
    ):
        super().__init__()
        self.policy_config = policy_config
        self.shared_encoder = shared_encoder
        self._output_size = policy_config.QNET.output_size
        fc_input_size = 0
        logger.info("******************************** Loading module: {} ********************************".format(
            self.__class__.__name__))
        fc_input_size += policy_config.RGB_ENCODER.hidden_size
        logger.info(
            "RGB encoder is {}".format(policy_config.RGB_ENCODER.backbone)
        )

        self.visual_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                self.shared_encoder.visual_encoder.output_size,
                policy_config.RGB_ENCODER.hidden_size,
            ),
            nn.ReLU(True),
        )

        if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
            input_gps_dim = observation_space.spaces[
                EpisodicGPSSensor.cls_uuid
            ].shape[0]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            fc_input_size += 32
            logger.info("\n\nSetting up GPS sensor")

        if EpisodicCompassSensor.cls_uuid in observation_space.spaces:
            assert (observation_space.spaces[EpisodicCompassSensor.cls_uuid].shape[
                        0
                    ]
                    == 1
                    ), "Expected compass with 2D rotation."
            input_compass_dim = 2  # cos and sin of the angle
            self.compass_embedding_dim = 32
            self.compass_embedding = nn.Linear(
                input_compass_dim, self.compass_embedding_dim
            )
            fc_input_size += 32
            logger.info("\n\nSetting up Compass sensor")

        if ObjectGoalSensor.cls_uuid in observation_space.spaces:
            self._n_object_categories = (
                    int(
                        observation_space.spaces[ObjectGoalSensor.cls_uuid].high[0]
                    )
                    + 1
            )
            logger.info(
                "Object categories: {}".format(self._n_object_categories)
            )
            self.obj_categories_embedding = nn.Embedding(
                self._n_object_categories, 32
            )
            fc_input_size += 32
            logger.info("\n\nSetting up Object Goal sensor")

        self.fc_input_size = fc_input_size

        logger.info(
            "Fc size input/output: {} - {}".format(
                fc_input_size, self._output_size
            )
        )

        self.state_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                self.fc_input_size,
                self._output_size,
            ),
            output_activation(),
        )
        self.train()

    @property
    def output_size(self):
        return self._output_size

    @property
    def is_blind(self):
        return self.shared_encoder.visual_encoder.is_blind and self.shared_encoder.depth_encoder.is_blind

    def num_recurrent_layers(self):
        """
        This is because Net interface is fixed for use with RNNs
        :return:
        """
        pass

    def forward(self, observations, rnn_hidden_states):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """
        x = []
        # Maybe try to pass it as an argument to forward function, but should be okey
        rgb = self.shared_encoder(observations, rnn_hidden_states)
        rgb = self.visual_fc(rgb)
        x.append(rgb)

        if EpisodicGPSSensor.cls_uuid in observations:
            obs_gps = observations[EpisodicGPSSensor.cls_uuid]
            if len(obs_gps.size()) == 3:
                obs_gps = obs_gps.contiguous().view(-1, obs_gps.size(2))
            x.append(self.gps_embedding(obs_gps))

        if EpisodicCompassSensor.cls_uuid in observations:
            obs_compass = observations["compass"]
            if len(obs_compass.size()) == 3:
                obs_compass = obs_compass.contiguous().view(
                    -1, obs_compass.size(2)
                )
            compass_observations = torch.stack(
                [
                    torch.cos(obs_compass),
                    torch.sin(obs_compass),
                ],
                -1,
            )
            compass_embedding = self.compass_embedding(
                compass_observations.float().squeeze(dim=1)
            )
            x.append(compass_embedding)

        if ObjectGoalSensor.cls_uuid in observations:
            object_goal = observations[ObjectGoalSensor.cls_uuid].long()
            if len(object_goal.size()) == 3:
                object_goal = object_goal.contiguous().view(
                    -1, object_goal.size(2)
                )
            x.append(self.obj_categories_embedding(object_goal).squeeze(dim=1))

        x = torch.cat(x, dim=1)

        x = self.state_encoder(
            x
        )

        return x


@baseline_registry.register_policy
class ObjectNavSharedPolicy(IQLRNNPolicy):
    def __init__(
            self,
            observation_space: Space,
            action_space: Space,
            policy_config: Config,
            run_type: str,
            hidden_size: int,
            rnn_type: str,
            num_recurrent_layers: int,
    ):

        super().__init__(
            SharedPolicyNet(
                observation_space=observation_space,
                policy_config=policy_config,
                num_actions=action_space.n,
                run_type=run_type,
                use_actions=False,
                hidden_size=hidden_size,
                rnn_type=rnn_type,
                num_recurrent_layers=num_recurrent_layers,
                is_policy=True,
            ),
            action_space.n,
        )
        self.qf1 = SharedQNet(
            observation_space=observation_space,
            policy_config=policy_config,
            num_actions=action_space.n,
            shared_encoder=self.net.shared_encoder,
            use_actions=True,
            hidden_size=hidden_size,
            rnn_type=rnn_type,
            num_recurrent_layers=num_recurrent_layers,
        )
        self.target_qf1 = SharedQNet(
            observation_space=observation_space,
            policy_config=policy_config,
            num_actions=action_space.n,
            shared_encoder=self.net.shared_encoder,
            use_actions=True,
            hidden_size=hidden_size,
            rnn_type=rnn_type,
            num_recurrent_layers=num_recurrent_layers,
        )
        self.vf = SharedVNet(
            observation_space=observation_space,
            policy_config=policy_config,
            shared_encoder=self.net.shared_encoder,
        )

    @classmethod
    def from_config(cls, config: Config, observation_space, action_space):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            policy_config=config.POLICY,
            run_type=config.RUN_TYPE,
            hidden_size=config.POLICY.STATE_ENCODER.hidden_size,
            rnn_type=config.POLICY.STATE_ENCODER.rnn_type,
            num_recurrent_layers=config.POLICY.STATE_ENCODER.num_recurrent_layers,
        )

    @property
    def num_recurrent_layers(self):
        return self.net.num_recurrent_layers

    def freeze_visual_encoders(self):
        for param in self.net.shared_encoder.parameters():
            param.requires_grad_(False)

    def unfreeze_visual_encoders(self):
        for param in self.net.shared_encoder.parameters():
            param.requires_grad_(True)

    def freeze_state_encoder(self):
        for param in self.net.state_encoder.parameters():
            param.requires_grad_(False)

    def unfreeze_state_encoder(self):
        for param in self.net.state_encoder.parameters():
            param.requires_grad_(True)

    def freeze_actor(self):
        for param in self.action_distribution.parameters():
            param.requires_grad_(False)

    def unfreeze_actor(self):
        for param in self.action_distribution.parameters():
            param.requires_grad_(True)
