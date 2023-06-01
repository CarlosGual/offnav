import torch
import torch.nn as nn
from gym import Space
from habitat import Config, logger
from habitat.tasks.nav.nav import EpisodicCompassSensor, EpisodicGPSSensor
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import build_rnn_state_encoder
from habitat_baselines.rl.ppo import Net

from offnav.policy.policy import ILPolicy, IQLPolicy
from offnav.policy.transforms import get_transform
from offnav.policy.visual_encoder import VisualEncoder
from offnav.utils.utils import load_encoder


class ObjectNavQNet(Net):
    r"""A baseline ResNet network that concatenates instruction,
    RGB, and depth encodings before decoding a reward value with an FC Layer.
    Modules:
        Instruction encoder
        Depth encoder
        RGB encoder
    """

    def __init__(
            self,
            observation_space: Space,
            policy_config: Config,
            num_actions: int,
            run_type: str,
            use_actions: bool,
            output_activation=nn.Identity
    ):
        super().__init__()
        self.policy_config = policy_config
        self._output_size = policy_config.QNET.output_size
        self.use_actions = use_actions
        fc_input_size = 0

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

        self.visual_encoder = VisualEncoder(
            image_size=rgb_config.image_size,
            backbone=rgb_config.backbone,
            input_channels=3,
            resnet_baseplanes=rgb_config.resnet_baseplanes,
            resnet_ngroups=rgb_config.resnet_baseplanes // 2,
            avgpooled_image=rgb_config.avgpooled_image,
            drop_path_rate=rgb_config.drop_path_rate,
        )

        self.visual_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                self.visual_encoder.output_size,
                policy_config.RGB_ENCODER.hidden_size,
            ),
            nn.ReLU(True),
        )

        fc_input_size += policy_config.RGB_ENCODER.hidden_size
        logger.info(
            "RGB encoder is {}".format(policy_config.RGB_ENCODER.backbone)
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

        if policy_config.SEQ2SEQ.use_action and self.use_actions:
            self.action_embedding = nn.Embedding(num_actions, 32)
            fc_input_size += self.action_embedding.embedding_dim

        self.fc_input_size = fc_input_size

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
        return self.visual_encoder.is_blind and self.depth_encoder.is_blind

    def num_recurrent_layers(self):
        """
        This is because Net interface is fixed for use with RNNs
        :return:
        """
        pass

    def forward(self, observations, actions):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """
        rgb_obs = observations["rgb"]

        x = []

        if self.visual_encoder is not None:
            if len(rgb_obs.size()) == 5:
                observations["rgb"] = rgb_obs.contiguous().view(
                    -1, rgb_obs.size(2), rgb_obs.size(3), rgb_obs.size(4)
                )
            # visual encoder
            rgb = observations["rgb"]

            rgb = self.visual_transform(rgb)
            rgb = self.visual_encoder(rgb)
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
                actions.float().long().view(-1)
            )
            x.append(actions_embedding)

        x = torch.cat(x, dim=1)

        x = self.state_encoder(
            x
        )

        return x


class ObjectPolicyNet(ObjectNavQNet):
    def __init__(self,
                 observation_space: Space,
                 policy_config: Config,
                 num_actions: int,
                 run_type: str,
                 use_actions: bool = False,
                 min_log_std=None,
                 max_log_std=None
                 ):
        super().__init__(
            observation_space,
            policy_config,
            num_actions,
            run_type,
            use_actions,
            output_activation=nn.Tanh,
        )
        self.log_std_logits = nn.Parameter()
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, observations, actions):
        mean = super().forward(observations, actions)
        # log_std = torch.sigmoid(self.log_std_logits)
        # log_std = self.min_log_std + log_std * (
        #         self.max_log_std - self.min_log_std)
        # std = torch.exp(log_std)
        return mean  # , std


@baseline_registry.register_policy
class ObjectNavIQLPolicy(IQLPolicy):
    def __init__(
            self,
            observation_space: Space,
            action_space: Space,
            policy_config: Config,
            run_type: str,
    ):
        super().__init__(
            ObjectPolicyNet(
                observation_space=observation_space,
                policy_config=policy_config,
                num_actions=action_space.n,
                run_type=run_type,
                use_actions=False,
                min_log_std=-6,
                max_log_std=0
            ),
            action_space.n,
        )
        self.qf1 = ObjectNavQNet(
            observation_space=observation_space,
            policy_config=policy_config,
            num_actions=action_space.n,
            run_type=run_type,
            use_actions=True
        )
        self.qf2 = ObjectNavQNet(
            observation_space=observation_space,
            policy_config=policy_config,
            num_actions=action_space.n,
            run_type=run_type,
            use_actions=True
        )
        self.target_qf1 = ObjectNavQNet(
            observation_space=observation_space,
            policy_config=policy_config,
            num_actions=action_space.n,
            run_type=run_type,
            use_actions=True
        )
        self.target_qf2 = ObjectNavQNet(
            observation_space=observation_space,
            policy_config=policy_config,
            num_actions=action_space.n,
            run_type=run_type,
            use_actions=True
        )
        self.vf = ObjectNavQNet(
            observation_space=observation_space,
            policy_config=policy_config,
            num_actions=action_space.n,
            run_type=run_type,
            use_actions=False
        )

    @classmethod
    def from_config(cls, config: Config, observation_space, action_space):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            policy_config=config.POLICY,
            run_type=config.RUN_TYPE,
        )

    @property
    def num_recurrent_layers(self):
        return self.net.num_recurrent_layers

    def freeze_visual_encoders(self):
        for param in self.net.visual_encoder.parameters():
            param.requires_grad_(False)

    def unfreeze_visual_encoders(self):
        for param in self.net.visual_encoder.parameters():
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
