from __future__ import annotations

import gymnasium
import numpy as np
import highway_env
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.envs.common.observation import LidarObservation


class highwayEnvContinuos(AbstractEnv):

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 5,
                    "normalize": True
                },
                "action": {
                    "type": "ContinuousAction",
                    "longitudinal": True,
                    "lateral": True,
                    "steering_range": [-0.5, 0.5],
                    "acceleration_range": [0.0, 1.0]
                },
                "lanes_count": 3,
                "vehicles_count": 50,
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "duration": 150,  # [s]
                "ego_spacing": 2,
                "vehicles_density": 1,
                "collision_reward": -1,      # Stronger penalty for collisions
                "right_lane_reward": 0.2,       # Encourage rightmost lane
                "high_speed_reward": 0.5,       # Encourage maintaining target speed
                "lane_change_penalty": -0.2,    # Penalize unnecessary lane changes
                "smoothness_penalty": -0.2,     # Penalize erratic steering
                "reverse_penalty" : -1,
                "reward_speed_range": [20, 40], # Target speed range (m/s)
                "normalize_reward": True,
                "offroad_terminal": True,
                "safe_distance": 5.0,
                "min_speed_bonus": 0.2,
                "offroad_penalty" : -1
            }
        )
        return config

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                self.config["lanes_count"], speed_limit=30
            ),
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
        )

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"],
            )
            vehicle = self.action_type.vehicle_class(
                self.road, vehicle.position, vehicle.heading, vehicle.speed
            )
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(
                    self.road, spacing=1 / self.config["vehicles_density"]
                )
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _is_terminated(self) -> bool:
        return (
            self.vehicle.crashed
            or self.config["offroad_terminal"]
            and not self.vehicle.on_road
        )

    def _is_truncated(self) -> bool:
        return (
            self.time >= self.config['duration']
        )

    def _reward(self, action: Action) -> float:
        rewards = self._rewards(action)

        reward = (
                self.config.get("collision_reward", -1.0) * rewards["collision_reward"]
                + self.config.get("right_lane_reward", 0.2) * rewards["right_lane_reward"]
                + self.config.get("high_speed_reward", 0.5) * rewards["high_speed_reward"]
                + self.config.get("lane_change_penalty", -0.2) * rewards["lane_change_penalty"]
                + self.config.get("smoothness_penalty", -0.2) * rewards["smoothness_penalty"]
                + self.config.get("reverse_penalty", -1.0) * rewards["reverse_penalty"]
                + self.config.get("min_speed_bonus", 0.2) * rewards["min_speed_bonus"]
        )

        # Normalize total reward if enabled
        if self.config.get("normalize_reward", False):
            min_expected = (
                    self.config["collision_reward"]
                    + self.config["lane_change_penalty"]
                    + self.config["smoothness_penalty"]
                    + self.config.get("reverse_penalty", -1.0)
            )
            max_expected = (
                    self.config["right_lane_reward"]
                    + self.config["high_speed_reward"]
                    + self.config.get("min_speed_bonus", 0.2)
            )
            reward = utils.lmap(reward, [min_expected, max_expected], [0, 1])

        # Multiply by on_road_reward (if not off-road)
        reward *= rewards.get("on_road_reward", 1.0)

        return reward

    def _rewards(self, action: Action) -> dict[str, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = (
            self.vehicle.target_lane_index[2]
            if isinstance(self.vehicle, ControlledVehicle)
            else self.vehicle.lane_index[2]
        )

        # --- Speed Reward ---
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        speed_norm = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        speed_reward = np.clip(speed_norm, 0, 1)

        # --- Right Lane Reward ---
        right_lane_pos = lane / max(len(neighbours) - 1, 1)

        # --- Lane Change Penalty ---
        lane_changed = int(
            self.vehicle.lane_index[2] != getattr(self.vehicle, "previous_lane_index", self.vehicle.lane_index[2])
        )
        self.vehicle.previous_lane_index = self.vehicle.lane_index[2]

        # --- Steering Smoothness Penalty ---
        steer = float(action[1]) if isinstance(action, (np.ndarray, list, tuple)) else 0.0
        smoothness_penalty = abs(steer)

        # --- Reverse Penalty ---
        reverse_penalty = -1.0 if self.vehicle.speed < 0 else 0.0

        # --- Minimum Speed Bonus (e.g., > 5 m/s) ---
        min_speed_bonus = 0.2 if self.vehicle.speed > 5.0 else 0.0

        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": right_lane_pos,
            "high_speed_reward": speed_reward,
            "lane_change_penalty": float(lane_changed),
            "smoothness_penalty": smoothness_penalty,
            "on_road_reward": float(self.vehicle.on_road),
            "reverse_penalty": reverse_penalty,
            "min_speed_bonus": min_speed_bonus
        }

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()