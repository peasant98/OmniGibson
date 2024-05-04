from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

import os
import time

import matplotlib.pyplot as plt
import psutil
from omni.isaac.core import SimulationContext
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.stage import add_reference_to_stage, is_stage_loading
from tqdm import tqdm

INSTANCEABLE = True
FIX_JOINT = True
NUM_ROBOTS_PER_ROUND = 50
NUM_ROUNDS = 10
NUM_STEPS = 50
asset_path = (
    "/home/svl/Downloads/ur5e/ur5e_instanceable.usd" if INSTANCEABLE else "/home/svl/Downloads/ur5e/ur5e_visuals.usd"
)
if FIX_JOINT:
    asset_path = asset_path.replace(".usd", "_joint.usd")
simulation_context = SimulationContext()

create_prim("/DistantLight", "DistantLight")
create_prim("/ground", "GroundPlane")
# wait for things to load
simulation_app.update()
while is_stage_loading():
    simulation_app.update()

simulation_context.play()

num_robot = []
avg_mem_usage = []
load_time = []
fps = []

for n_round in range(NUM_ROUNDS):
    start_time = time.time()
    for i in tqdm(range(n_round * NUM_ROBOTS_PER_ROUND, (n_round + 1) * NUM_ROBOTS_PER_ROUND)):
        robot_prim = add_reference_to_stage(usd_path=asset_path, prim_path=f"/World/robot_{i}")

    # need to initialize physics getting any articulation..etc
    simulation_context.initialize_physics()

    robot_views = []
    memory_usage = []
    for i in tqdm(range(n_round * NUM_ROBOTS_PER_ROUND, (n_round + 1) * NUM_ROBOTS_PER_ROUND)):
        robot_views.append(RigidPrimView(f"/World/robot_{i}/base_link"))
        robot_views[-1].initialize()
        robot_views[-1].set_world_poses(positions=[[i * 2, 0, 0]])

    end_time = time.time()

    total_step_time = 0
    for i in tqdm(range(NUM_STEPS)):
        st = time.time()
        simulation_context.step(render=True)
        total_step_time += time.time() - st
        memory_usage.append(psutil.Process(os.getpid()).memory_info().rss / 1024**3)

    fps.append(NUM_STEPS / total_step_time)
    load_time.append(end_time - start_time)
    num_robot.append((n_round + 1) * NUM_ROBOTS_PER_ROUND)
    avg_mem_usage.append(sum(memory_usage) / NUM_STEPS)

plt.plot(num_robot, load_time, label="load time")
# plt.plot(num_robot, fps, label="fps")
plt.plot(num_robot, avg_mem_usage, label="mem usage")
plt.legend()
plt.savefig(f"/home/svl/Downloads/result_{INSTANCEABLE}_{FIX_JOINT}.png")
plt.show()


simulation_context.stop()
simulation_app.close()
