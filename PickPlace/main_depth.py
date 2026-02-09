from isaacsim import SimulationApp

# 1. 启动仿真
simulation_app = SimulationApp({"headless": False})

import os
import time
import numpy as np
from PIL import Image

from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera

from env_setup import PickPlaceTask

# 2. 环境初始化
my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()

# 3. 添加任务
my_task = PickPlaceTask(name="my_pick_place_task")
my_world.add_task(my_task)

# 4. 添加深度相机
camera = Camera(
    prim_path="/World/DepthCamera",
    name="depth_camera",
    position=np.array([0.0, 0.5, 5.0]),
    orientation=np.array([1.0, 0.0, 0.0, 0.0]),
    frequency=30,
    resolution=(640, 480),
)
my_world.scene.add(camera)

# 5. 重置与相机初始化
my_world.reset()
if hasattr(camera, "initialize"):
    camera.initialize()

if hasattr(camera, "add_depth_to_sensors"):
    camera.add_depth_to_sensors()
elif hasattr(camera, "set_depth_enabled"):
    camera.set_depth_enabled(True)

# 6. 输出目录
out_dir = os.path.join(os.getcwd(), "_out_depth_camera")
os.makedirs(out_dir, exist_ok=True)

reset_needed = False
last_save_time = time.time()
frame_idx = 0

# 深度图可视化范围（米）
depth_max_m = 10.0

print("[MainDepth] Simulation started... Saving RGB + depth every 1s.")

while simulation_app.is_running():
    my_world.step(render=True)

    if my_world.is_stopped() and not reset_needed:
        reset_needed = True

    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            reset_needed = False

        now = time.time()
        if now - last_save_time >= 1.0:
            # RGB
            rgb = None
            if hasattr(camera, "get_rgba"):
                rgba = camera.get_rgba()
                if rgba is not None:
                    if rgba.dtype == np.uint8:
                        rgb = rgba[..., :3]
                    else:
                        rgb = (rgba[..., :3] * 255).astype(np.uint8)
            elif hasattr(camera, "get_rgb"):
                rgb = camera.get_rgb()
                if rgb is not None and rgb.dtype != np.uint8:
                    rgb = (rgb * 255).astype(np.uint8)

            # Depth
            depth = camera.get_depth() if hasattr(camera, "get_depth") else None

            if rgb is not None:
                Image.fromarray(rgb).save(os.path.join(out_dir, f"{frame_idx:06d}_rgb.png"))

            if depth is not None:
                depth_clean = np.array(depth, copy=True)
                depth_clean[np.isinf(depth_clean)] = 0.0
                depth_clean = np.clip(depth_clean, 0.0, depth_max_m)
                depth_uint16 = (depth_clean / depth_max_m * 65535).astype(np.uint16)
                Image.fromarray(depth_uint16, mode="I;16").save(
                    os.path.join(out_dir, f"{frame_idx:06d}_depth.png")
                )

            frame_idx += 1
            last_save_time = now

simulation_app.close()
