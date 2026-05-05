import sapien
import torch
import numpy as np
from mani_skill.utils import sapien_utils, common
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.agents.robots import SO100
from mani_skill.utils.building import actors
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.registration import register_env
from mani_skill.sensors.camera import CameraConfig

@register_env("CustomPickCube-v1", max_episode_steps=200)
class CustomPickCubeEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["so100"]

    def __init__(
            self, robot_uids="so100", *args, **kwargs
        ):

        super().__init__(robot_uids=robot_uids, *args, **kwargs)
        
    @property
    def _default_sensor_configs(self):
        # registers one 128x128 camera looking at the robot, cube, and target
        # a smaller sized camera will be lower quality, but render faster
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig("base_camera", pose=pose, width=128, height=128, fov=np.pi / 2, near=0.01, far=100)
        ]
    @property
    def _default_human_render_camera_configs(self):
        # registers a more high-definition (512x512) camera used just for rendering when render_mode="rgb_array" or calling env.render_rgb_array()
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100)
    
    def _load_agent(self, options: dict):

        super()._load_agent(options, sapien.Pose(
            p=[0, 0, 0]
        ))
    
    def _load_scene(self, options):
        
        self.table_scene = TableSceneBuilder(
            env=self,
            robot_init_qpos_noise=0
        )
        
        self.table_scene.build()

        self.obj = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=np.array([12, 42, 160, 255]) / 255,
            name="cube",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[-0.40, 0.0, 0.02]),
        )
    
    def _initialize_episode(self, env_idx, options):
        
        with torch.device("cuda"):

            self.table_scene.initialize(env_idx)
            print(self.obj.pose.get_p())
            print(self.obj.pose.get_q())
    
    def evaluate(self):

        return {
            "success" : torch.linalg.norm(torch.tensor(1.1)) > 0
        }
    
    def compute_normalized_dense_reward(self, obs, action, info):

        return 0