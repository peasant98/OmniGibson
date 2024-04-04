import math
import numpy as np
from omnigibson.reward_functions.reward_function_base import BaseRewardFunction
from omnigibson.utils.motion_planning_utils import detect_robot_collision_in_sim
import omnigibson.utils.transform_utils as T

from scipy.spatial.transform import Rotation as R


class GraspReward(BaseRewardFunction):
    """
    Grasp reward
    """

    def __init__(
        self,
        obj_name,
        dist_coeff,
        grasp_reward,
        collision_penalty,
        eef_position_penalty_coef,
        eef_orientation_penalty_coef,
        regularization_coef,
    ):
        # Store internal vars
        self.prev_grasping = False
        self.prev_eef_pos = None
        self.prev_eef_rot = None
        self.obj_name = obj_name
        self.obj = None
        self.dist_coeff = dist_coeff
        self.grasp_reward = grasp_reward
        self.collision_penalty = collision_penalty
        self.eef_position_penalty_coef = eef_position_penalty_coef
        self.eef_orientation_penalty_coef = eef_orientation_penalty_coef
        self.regularization_coef = regularization_coef

        # Run super
        super().__init__()

    def _step(self, task, env, action):
        self.obj = env.scene.object_registry("name", f"{self.obj_name}_{env.id}") if self.obj is None else self.obj

        robot = env.robots[0]
        obj_in_hand = robot._ag_obj_in_hand[robot.default_arm]
        current_grasping = obj_in_hand == self.obj

        # Reward varying based on combination of whether the robot was previously grasping the desired and object
        # and is currently grasping the desired object
        reward = 0.0

        # Penalize large actions
        reward += -(np.sum(np.abs(action)) * self.regularization_coef)

        # Penalize based on the magnitude of the action
        eef_pos = robot.get_eef_position(robot.default_arm)
        if self.prev_eef_pos is not None:
            action_mag = T.l2_distance(self.prev_eef_pos, eef_pos)
            reward += -action_mag * self.eef_position_penalty_coef
        self.prev_eef_pos = eef_pos

        eef_rot = R.from_quat(robot.get_eef_orientation(robot.default_arm))
        if self.prev_eef_rot is not None:
            delta_rot = eef_rot * self.prev_eef_rot.inv()
            reward += delta_rot.magnitude() * self.eef_orientation_penalty_coef
        self.prev_eef_rot = eef_rot

        # Penalize robot for colliding with an object
        if detect_robot_collision_in_sim(robot, filter_objs=[self.obj]):
            reward -= self.collision_penalty

        # If we're not currently grasping
        if not current_grasping:
            # TODO: If we dropped the object recently, penalize for that
            obj_center = self.obj.get_position()
            dist = T.l2_distance(eef_pos, obj_center)
            reward += math.exp(-dist) * self.dist_coeff

        else:
            # We are currently grasping - first apply a grasp reward
            reward += self.grasp_reward

            # Then apply a distance reward to take us to a tucked position
            robot_center = robot.links["torso_lift_link"].get_position()
            obj_center = self.obj.get_position()
            dist = T.l2_distance(robot_center, obj_center)
            reward += math.exp(-dist) * self.dist_coeff

        self.prev_grasping = current_grasping

        return reward, {"grasp_success": current_grasping}

    def reset(self, task, env):
        """
        Reward function-specific reset

        Args:
            task (BaseTask): Task instance
            env (Environment): Environment instance
        """
        super().reset(task, env)
        self.prev_grasping = False
        self.prev_eef_pos = None
        self.prev_eef_rot = None
