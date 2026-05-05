import scene
import yourdfpy
import numpy as np
from pathlib import Path
import gymnasium as gym
import pyroki as pk
import jax.numpy as jnp
from ik_solver import _solve_ik_jax_batched
from mani_skill.utils.structs.pose import Pose



def close_gripper_action(current_qpos):
    current_qpos = current_qpos[-1] = np.pi/2
    return current_qpos

def open_gripper_action(current_qpos):
    current_qpos = current_qpos[-1] = np.pi/2
    return current_qpos

def plan_pick_cube():
    urdf_path = Path("SO100/so100.urdf")
    urdf = yourdfpy.URDF.load(urdf_path)
    target_link_name = "gripper"
    robot = pk.Robot.from_urdf(urdf)

    solution = _solve_ik_jax_batched(
        robot=robot,
        target_link_index=robot.links.names.index(target_link_name),
        target_wxyz=jnp.array([[0, 0, 1, 0]]),
        target_position=jnp.array([[0, 0, 0]]), 
    )
    return np.array(solution).flatten()

env = gym.make(
    "CustomPickCube-v1",
    num_envs=1,
    obs_mode="state_dict",
    render_mode="human"
)

obs, info = env.reset()  # fix: unpack both return values

solution = plan_pick_cube()
print("Action shape:", solution.shape)
print("Expected action space:", env.action_space)  # check these match!

terminated = False
truncated = False

while True:
    action = solution  # still static — see note below
    for step_size in range(10):
        obs, reward, terminated, truncated, info = env.step(None)
        env.render()
    '''
    env.step(open_gripper_action(obs['agent']['qpos']))
    env.render()
    env.step(close_gripper_action(obs['agent']['qpos']))
    env.render()
    '''
env.close()