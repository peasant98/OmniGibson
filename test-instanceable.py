import os
import time

import psutil
from tqdm import tqdm

import omnigibson as og
from omnigibson.objects import USDObject
from omnigibson.scenes import Scene

INSTANCEABLE = True
NUM_ROBOTS = 200
NUM_STEPS = 10
asset_path = (
    # "/home/svl/Downloads/ur5e/ur5e_instanceable.usd"
    "/home/svl/Documents/OmniGibson/omnigibson/data/assets/models/fetch/fetch/fetch.usd"
    # "/home/svl/Documents/OmniGibson/omnigibson/data/assets/models/tiago/tiago_dual_omnidirectional_stanford/tiago_dual_omnidirectional_stanford_33_instanceable.usd"
    # "/home/svl/Documents/OmniGibson/omnigibson/data/assets/models/tiago/tiago_dual_omnidirectional_stanford/tiago_dual_omnidirectional_stanford.usd"
    if INSTANCEABLE
    else
    # "/home/svl/Downloads/ur5e/ur5e_visuals.usd"
    "/home/svl/Documents/OmniGibson/omnigibson/data/assets/models/fetch/fetch/fetch_backup.usd"
    # "/home/svl/Documents/OmniGibson/omnigibson/data/assets/models/tiago/tiago_dual_omnidirectional_stanford/tiago_dual_omnidirectional_stanford_33.usd"
)

og.launch()

scene = Scene()
og.sim.import_scene(scene)

start = time.time()
robots = []
for i in tqdm(range(NUM_ROBOTS)):
    robots.append(USDObject(f"robot_{i}", asset_path))
    og.sim.import_object(robots[-1])
    robots[-1].set_position([i * 2, 0, 0])
og.sim.play()
end_import = time.time()

# from IPython import embed; embed()
memory_usage, vram_usage = [], []
for i in tqdm(range(NUM_STEPS)):
    og.sim.step()
    # profiling
    # memory usage in GB
    memory_usage.append(psutil.Process(os.getpid()).memory_info().rss / 1024**3)
    # VRAM usage in GB
    # for gpu in nvidia_smi.getInstance().DeviceQuery()["gpu"]:
    #     found = False
    #     for process in gpu["processes"]:
    #         if process["pid"] == os.getpid():
    #             vram_usage.append(process["used_memory"] / 1024)
    #             found = True
    #             break
    #     if found:
    #         break


print("=" * 80)
print(f"Instanceable: {INSTANCEABLE}")
print(f"Number of robots: {NUM_ROBOTS}")
print(f"Import time: {end_import - start} s")
# memory
print(f"Max memory usage: {max(memory_usage)} GB")
print(f"Average memory usage: {sum(memory_usage) / len(memory_usage)} GB")
# VRAM
# print(f"Max VRAM usage: {max(vram_usage)} GB")
# print(f"Average VRAM usage: {sum(vram_usage) / len(vram_usage)} GB")
print("=" * 80)

og.shutdown()
