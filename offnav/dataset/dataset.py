#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import random
import numpy as np
from numpy import ndarray
from itertools import groupby
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Union, Tuple,
)

from habitat.config import Config
from habitat.core.dataset import Episode, T
from habitat.core.registry import registry
from habitat.core.simulator import AgentState, ShortestPathPoint
from habitat.core.utils import DatasetFloatJSONEncoder
from habitat.datasets.pointnav.pointnav_dataset import (
    CONTENT_SCENES_PATH_FIELD,
    DEFAULT_SCENE_PATH_PREFIX,
    PointNavDatasetV1,
)
from habitat.tasks.nav.object_nav_task import ObjectGoal, ObjectViewLocation

from offnav.task.object_nav_task import ObjectGoalNavEpisode, ReplayActionSpec


@registry.register_dataset(name="ObjectNav-v2")
class ObjectNavDatasetV2(PointNavDatasetV1):
    r"""Class inherited from PointNavDataset that loads Object Navigation dataset."""
    category_to_task_category_id: Dict[str, int]
    category_to_scene_annotation_category_id: Dict[str, int]
    episodes: List[ObjectGoalNavEpisode] = []  # type: ignore
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"
    goals_by_category: Dict[str, Sequence[ObjectGoal]]
    gibson_to_mp3d_category_map: Dict[str, str] = {'couch': 'sofa', 'toilet': 'toilet', 'bed': 'bed',
                                                   'tv': 'tv_monitor', 'potted plant': 'plant', 'chair': 'chair'}
    max_episode_steps: int = 500

    @staticmethod
    def dedup_goals(dataset: Dict[str, Any]) -> Dict[str, Any]:
        if len(dataset["episodes"]) == 0:
            return dataset

        goals_by_category = {}
        for i, ep in enumerate(dataset["episodes"]):
            dataset["episodes"][i]["object_category"] = ep["goals"][0][
                "object_category"
            ]
            ep = ObjectGoalNavEpisode(**ep)

            goals_key = ep.goals_key
            if goals_key not in goals_by_category:
                goals_by_category[goals_key] = ep.goals

            dataset["episodes"][i]["goals"] = []

        dataset["goals_by_category"] = goals_by_category

        return dataset

    def to_json(self) -> str:
        for i in range(len(self.episodes)):
            self.episodes[i].goals = []

        result = DatasetFloatJSONEncoder().encode(self)

        for i in range(len(self.episodes)):
            goals = self.goals_by_category[self.episodes[i].goals_key]
            if not isinstance(goals, list):
                goals = list(goals)
            self.episodes[i].goals = goals

        return result

    def __init__(self, config: Optional[Config] = None) -> None:
        self.goals_by_category = {}
        if config is not None:
            self.max_episode_steps = config.MAX_EPISODE_STEPS
        super().__init__(config)
        self.episodes = list(self.episodes)

    @staticmethod
    def __deserialize_goal(serialized_goal: Dict[str, Any]) -> ObjectGoal:
        g = ObjectGoal(**serialized_goal)

        for vidx, view in enumerate(g.view_points):
            view_location = ObjectViewLocation(**view)  # type: ignore
            view_location.agent_state = AgentState(**view_location.agent_state)  # type: ignore
            g.view_points[vidx] = view_location

        return g

    def from_json(
            self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        if "category_to_task_category_id" in deserialized:
            self.category_to_task_category_id = deserialized[
                "category_to_task_category_id"
            ]

        if "category_to_scene_annotation_category_id" in deserialized:
            self.category_to_scene_annotation_category_id = deserialized[
                "category_to_scene_annotation_category_id"
            ]

        if "category_to_mp3d_category_id" in deserialized:
            self.category_to_scene_annotation_category_id = deserialized[
                "category_to_mp3d_category_id"
            ]

        assert len(self.category_to_task_category_id) == len(
            self.category_to_scene_annotation_category_id
        )

        assert set(self.category_to_task_category_id.keys()) == set(
            self.category_to_scene_annotation_category_id.keys()
        ), "category_to_task and category_to_mp3d must have the same keys"

        if len(deserialized["episodes"]) == 0:
            return

        if "goals_by_category" not in deserialized:
            deserialized = self.dedup_goals(deserialized)

        for k, v in deserialized["goals_by_category"].items():
            self.goals_by_category[k] = [self.__deserialize_goal(g) for g in v]

        for i, episode in enumerate(deserialized["episodes"]):
            if "_shortest_path_cache" in episode:
                del episode["_shortest_path_cache"]

            if "scene_state" in episode:
                del episode["scene_state"]

            if "gibson" in episode["scene_id"]:
                episode["scene_id"] = "gibson_semantic/{}".format(episode["scene_id"].split("/")[-1])

            episode = ObjectGoalNavEpisode(**episode)
            episode.episode_id = str(i)
            episode.start_position = list(map(float, episode.start_position))
            episode.start_rotation = list(map(float, episode.start_rotation))

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                                       len(DEFAULT_SCENE_PATH_PREFIX):
                                       ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            episode.goals = self.goals_by_category[episode.goals_key]
            if episode.scene_dataset == "gibson":
                episode.object_category = self.gibson_to_mp3d_category_map[episode.object_category]

            if episode.reference_replay is not None:
                for i, replay_step in enumerate(episode.reference_replay):
                    replay_step["agent_state"] = None
                    episode.reference_replay[i] = ReplayActionSpec(**replay_step)

            if episode.shortest_paths is not None:
                for path in episode.shortest_paths:
                    for p_index, point in enumerate(path):
                        if point is None or isinstance(point, (int, str)):
                            point = {
                                "action": point,
                                "rotation": None,
                                "position": None,
                            }

                        path[p_index] = ShortestPathPoint(**point)

            if episode.reference_replay is not None and len(episode.reference_replay) > self.max_episode_steps:
                continue

            self.episodes.append(episode)  # type: ignore [attr-defined]


@registry.register_dataset(name="ObjectNav-meta")
class ObjectNavDatasetMeta(PointNavDatasetV1):
    r"""Class inherited from PointNavDataset that loads Object Navigation dataset."""
    category_to_task_category_id: Dict[str, int]
    category_to_scene_annotation_category_id: Dict[str, int]
    episodes: List[ObjectGoalNavEpisode] = []  # type: ignore
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"
    goals_by_category: Dict[str, Sequence[ObjectGoal]]
    gibson_to_mp3d_category_map: Dict[str, str] = {'couch': 'sofa', 'toilet': 'toilet', 'bed': 'bed',
                                                   'tv': 'tv_monitor', 'potted plant': 'plant', 'chair': 'chair'}
    max_episode_steps: int = 500

    @staticmethod
    def dedup_goals(dataset: Dict[str, Any]) -> Dict[str, Any]:
        if len(dataset["episodes"]) == 0:
            return dataset

        goals_by_category = {}
        for i, ep in enumerate(dataset["episodes"]):
            dataset["episodes"][i]["object_category"] = ep["goals"][0][
                "object_category"
            ]
            ep = ObjectGoalNavEpisode(**ep)

            goals_key = ep.goals_key
            if goals_key not in goals_by_category:
                goals_by_category[goals_key] = ep.goals

            dataset["episodes"][i]["goals"] = []

        dataset["goals_by_category"] = goals_by_category

        return dataset

    def to_json(self) -> str:
        for i in range(len(self.episodes)):
            self.episodes[i].goals = []

        result = DatasetFloatJSONEncoder().encode(self)

        for i in range(len(self.episodes)):
            goals = self.goals_by_category[self.episodes[i].goals_key]
            if not isinstance(goals, list):
                goals = list(goals)
            self.episodes[i].goals = goals

        return result

    def __init__(self, config: Optional[Config] = None) -> None:
        self.goals_by_category = {}
        if config is not None:
            self.max_episode_steps = config.MAX_EPISODE_STEPS
        super().__init__(config)
        self.episodes = list(self.episodes)

    def get_episode_iterator(self, *args: Any, **kwargs: Any) -> Iterator[T]:
        r"""Gets episode iterator with options. Options are specified in
        :ref:`EpisodeIterator` documentation.

        :param args: positional args for iterator constructor
        :param kwargs: keyword args for iterator constructor
        :return: episode iterator with specified behavior

        To further customize iterator behavior for your :ref:`Dataset`
        subclass, create a customized iterator class like
        :ref:`EpisodeIterator` and override this method.
        """
        return MetaEpisodeIterator(self.episodes, *args, **kwargs)

    @staticmethod
    def __deserialize_goal(serialized_goal: Dict[str, Any]) -> ObjectGoal:
        g = ObjectGoal(**serialized_goal)

        for vidx, view in enumerate(g.view_points):
            view_location = ObjectViewLocation(**view)  # type: ignore
            view_location.agent_state = AgentState(**view_location.agent_state)  # type: ignore
            g.view_points[vidx] = view_location

        return g

    def from_json(
            self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        if "category_to_task_category_id" in deserialized:
            self.category_to_task_category_id = deserialized[
                "category_to_task_category_id"
            ]

        if "category_to_scene_annotation_category_id" in deserialized:
            self.category_to_scene_annotation_category_id = deserialized[
                "category_to_scene_annotation_category_id"
            ]

        if "category_to_mp3d_category_id" in deserialized:
            self.category_to_scene_annotation_category_id = deserialized[
                "category_to_mp3d_category_id"
            ]

        assert len(self.category_to_task_category_id) == len(
            self.category_to_scene_annotation_category_id
        )

        assert set(self.category_to_task_category_id.keys()) == set(
            self.category_to_scene_annotation_category_id.keys()
        ), "category_to_task and category_to_mp3d must have the same keys"

        if len(deserialized["episodes"]) == 0:
            return

        if "goals_by_category" not in deserialized:
            deserialized = self.dedup_goals(deserialized)

        for k, v in deserialized["goals_by_category"].items():
            self.goals_by_category[k] = [self.__deserialize_goal(g) for g in v]

        for i, episode in enumerate(deserialized["episodes"]):
            if "_shortest_path_cache" in episode:
                del episode["_shortest_path_cache"]

            if "scene_state" in episode:
                del episode["scene_state"]

            if "gibson" in episode["scene_id"]:
                episode["scene_id"] = "gibson_semantic/{}".format(episode["scene_id"].split("/")[-1])

            episode = ObjectGoalNavEpisode(**episode)
            episode.episode_id = str(i)
            episode.start_position = list(map(float, episode.start_position))
            episode.start_rotation = list(map(float, episode.start_rotation))

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                                       len(DEFAULT_SCENE_PATH_PREFIX):
                                       ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            episode.goals = self.goals_by_category[episode.goals_key]
            if episode.scene_dataset == "gibson":
                episode.object_category = self.gibson_to_mp3d_category_map[episode.object_category]

            if episode.reference_replay is not None:
                for i, replay_step in enumerate(episode.reference_replay):
                    replay_step["agent_state"] = None
                    episode.reference_replay[i] = ReplayActionSpec(**replay_step)

            if episode.shortest_paths is not None:
                for path in episode.shortest_paths:
                    for p_index, point in enumerate(path):
                        if point is None or isinstance(point, (int, str)):
                            point = {
                                "action": point,
                                "rotation": None,
                                "position": None,
                            }

                        path[p_index] = ShortestPathPoint(**point)

            if episode.reference_replay is not None and len(episode.reference_replay) > self.max_episode_steps:
                continue

            self.episodes.append(episode)  # type: ignore [attr-defined]


class MetaEpisodeIterator(Iterator[T]):
    r"""Episode Iterator class that gives options for how a list of episodes
    should be iterated.

    Some of those options are desirable for the internal simulator to get
    higher performance. More context: simulator suffers overhead when switching
    between scenes, therefore episodes of the same scene should be loaded
    consecutively. However, if too many consecutive episodes from same scene
    are feed into RL model, the model will risk to overfit that scene.
    Therefore it's better to load same scene consecutively and switch once a
    number threshold is reached.

    Currently supports the following features:

    Cycling:
        when all episodes are iterated, cycle back to start instead of throwing
        StopIteration.
    Cycling with shuffle:
        when cycling back, shuffle episodes groups grouped by scene.
    Group by scene:
        episodes of same scene will be grouped and loaded consecutively.
    Set max scene repeat:
        set a number threshold on how many episodes from the same scene can be
        loaded consecutively.
    Sample episodes:
        sample the specified number of episodes.
    """

    def __init__(
            self,
            episodes: Sequence[T],
            cycle: bool = True,
            shuffle: bool = True,
            seed: int = None,
    ) -> None:
        r"""..

        :param episodes: list of episodes.
        :param cycle: if :py:`True`, cycle back to first episodes when
            StopIteration.
        :param shuffle: if :py:`True`, shuffle scene groups when cycle. No
            effect if cycle is set to :py:`False`. Will shuffle grouped scenes
            if :p:`group_by_scene` is :py:`True`.
        :param group_by_scene: if :py:`True`, group episodes from same scene.
        :param max_scene_repeat_episodes: threshold of how many episodes from the same
            scene can be loaded consecutively. :py:`-1` for no limit
        :param max_scene_repeat_steps: threshold of how many steps from the same
            scene can be taken consecutively. :py:`-1` for no limit
        :param num_episode_sample: number of episodes to be sampled. :py:`-1`
            for no sampling.
        :param step_repetition_range: The maximum number of steps within each scene is
            uniformly drawn from
            [1 - step_repeat_range, 1 + step_repeat_range] * max_scene_repeat_steps
            on each scene switch.  This stops all workers from swapping scenes at
            the same time
        """
        if seed:
            random.seed(seed)
            np.random.seed(seed)

        if not isinstance(episodes, list):
            episodes = list(episodes)

        self.episodes = episodes
        self.cycle = cycle
        self.shuffle = shuffle
        self._step_count = 0

        # Get episodes in dict form with scenes and goals ids
        self.episodes_dict = {}
        for episode in self.episodes:
            if episode.goals_key not in self.episodes_dict:
                self.episodes_dict[episode.goals_key] = []
            self.episodes_dict[episode.goals_key].append(episode)

        self._iterator = None
        # self._prev_task_id = None
        # self._task_id = None

    def set_task(self, task_id: str) -> bool:
        task_episodes = self.episodes_dict[task_id]
        # self._task_id = task_id
        self._iterator = iter(task_episodes)
        self._shuffle()
        return True

    def __iter__(self) -> "MetaEpisodeIterator":
        return self

    def __next__(self) -> Episode:
        r"""The main logic for handling how episodes will be iterated.

        :return: next episode.
        """
        assert self._iterator is not None, "Task not set. A call to set_task is required before calling next"

        next_episode = next(self._iterator, None)
        if next_episode is None:
            if not self.cycle:
                raise StopIteration

            self._iterator = iter(self.episodes)

            if self.shuffle:
                self._shuffle()

            next_episode = next(self._iterator)

        # if (
        #         self._prev_task_id != self._task_id
        #         and self._prev_task_id is not None
        # ):
        #     self._rep_count = 0
        #     self._step_count = 0
        #
        # self._prev_scene_id = self._task_id

        return next_episode

    def _shuffle(self) -> None:
        r"""Internal method that shuffles the remaining episodes.
        If self.group_by_scene is true, then shuffle groups of scenes.
        """
        assert self.shuffle
        episodes = list(self._iterator)

        random.shuffle(episodes)

        self._iterator = iter(episodes)  # type: ignore[arg-type]

    def step_taken(self) -> None:
        self._step_count += 1

