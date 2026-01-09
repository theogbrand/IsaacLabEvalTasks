# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Termination functions for the TCR bussing task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def task_done_tcr_bussing(
    env: ManagerBasedRLEnv,
    trash_item_1_cfg: SceneEntityCfg = SceneEntityCfg("trash_item_1"),
    trash_item_2_cfg: SceneEntityCfg = SceneEntityCfg("trash_item_2"),
    collection_box_cfg: SceneEntityCfg = SceneEntityCfg("collection_box"),
    max_distance_x: float = 0.10,
    max_distance_y: float = 0.10,
    max_distance_z: float = 0.08,
) -> torch.Tensor:
    """Determine if the TCR bussing task is complete.

    This function checks whether all success conditions for the task have been met:
    All trash items must be inside the collection box (within threshold distances).

    Args:
        env: The RL environment instance.
        trash_item_1_cfg: Configuration for the first trash item entity.
        trash_item_2_cfg: Configuration for the second trash item entity.
        collection_box_cfg: Configuration for the collection box entity.
        max_distance_x: Maximum x distance from trash to box center for task completion.
        max_distance_y: Maximum y distance from trash to box center for task completion.
        max_distance_z: Maximum z distance (height above box) for task completion.

    Returns:
        Boolean tensor indicating which environments have completed the task.
    """
    # Get object entities from the scene
    trash_1: RigidObject = env.scene[trash_item_1_cfg.name]
    trash_2: RigidObject = env.scene[trash_item_2_cfg.name]
    collection_box: RigidObject = env.scene[collection_box_cfg.name]

    # Get positions relative to environment origin
    trash_1_pos = trash_1.data.root_pos_w - env.scene.env_origins
    trash_2_pos = trash_2.data.root_pos_w - env.scene.env_origins
    box_pos = collection_box.data.root_pos_w - env.scene.env_origins

    # Calculate distances for trash item 1
    trash_1_to_box_x = torch.abs(trash_1_pos[:, 0] - box_pos[:, 0])
    trash_1_to_box_y = torch.abs(trash_1_pos[:, 1] - box_pos[:, 1])
    trash_1_to_box_z = trash_1_pos[:, 2] - box_pos[:, 2]

    # Calculate distances for trash item 2
    trash_2_to_box_x = torch.abs(trash_2_pos[:, 0] - box_pos[:, 0])
    trash_2_to_box_y = torch.abs(trash_2_pos[:, 1] - box_pos[:, 1])
    trash_2_to_box_z = trash_2_pos[:, 2] - box_pos[:, 2]

    # Check conditions for trash item 1
    done = trash_1_to_box_x < max_distance_x
    done = torch.logical_and(done, trash_1_to_box_y < max_distance_y)
    done = torch.logical_and(done, trash_1_to_box_z < max_distance_z)

    # Check conditions for trash item 2
    done = torch.logical_and(done, trash_2_to_box_x < max_distance_x)
    done = torch.logical_and(done, trash_2_to_box_y < max_distance_y)
    done = torch.logical_and(done, trash_2_to_box_z < max_distance_z)

    return done
