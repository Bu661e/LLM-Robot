# main.py
from isaacsim import SimulationApp
import numpy as np

# 1. 启动仿真
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from env_setup import PickPlaceTask
from robot_api import LLMRobotWrapper

# 2. 环境初始化
my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()

# 3. 添加任务
my_task = PickPlaceTask(name="my_pick_place_task")
my_world.add_task(my_task)

# 4. 重置与预热
my_world.reset()

reset_needed = False
task_executed = False

print("[Main] Simulation started...")

while simulation_app.is_running():
    my_world.step(render=True)

    if my_world.is_stopped() and not reset_needed:
        reset_needed = True

    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            task_executed = False
            reset_needed = False
        
        # --- 核心逻辑 ---
        if not task_executed:
            
            if my_task._robot is None:
                continue

            robot = LLMRobotWrapper(
                world=my_world,
                robot=my_task._robot
            )
            # ================================================================
            # # LLM 生成
            # print("\n[Main] Planning Task: Red Cube and Blue Cube -> Stacking")
            # red_cube_source_pos = [0.5, 0.0, 0.025]  # 红色方块中心位置 (Z轴为方块高度的一半)
            # blue_cube_source_pos = [0.5, 0.4, 0.025] # 蓝色方块中心位置
            
            # # 提取目标坐标 (放置位置)
            # target_obj_pos = [0, 0.4, 0.03]  # 堆叠位置

            # # 调用新的 API (传入坐标)
            # robot.pick_and_place(
            #     pick_position=red_cube_source_pos, 
            #     place_position=target_obj_pos
            # )
            # robot.pick_and_place(
            #     pick_position=blue_cube_source_pos, 
            #     place_position=target_obj_pos + np.array([0, 0, 0.05])
            # )
            # ================================================================

            perception_data = {"timestamp":"2026-01-27T06:00:00Z","frame_id":"World","detected_objects":[{"label":"red_cube","instance_id":0,"layout":{"translate":[0.5,0,0.025],"rotation":[1,0,0,0],"scale":[0.05,0.05,0.05]}},{"label":"blue_cube","instance_id":1,"layout":{"translate":[0.5,0.4,0.025],"rotation":[1,0,0,0],"scale":[0.05,0.05,0.05]}}]}

            # Extract object data from perception
            red_cube = next(obj for obj in perception_data["detected_objects"] if obj["label"] == "red_cube")
            blue_cube = next(obj for obj in perception_data["detected_objects"] if obj["label"] == "blue_cube")

            # Source (blue_cube) current center for picking
            pick_position = blue_cube["layout"]["translate"]  # [0.5, 0.4, 0.025]

            # Target (red_cube) parameters for placement calculation
            target_center_z = red_cube["layout"]["translate"][2]  # 0.025
            target_height = red_cube["layout"]["scale"][2]        # 0.05
            source_height = blue_cube["layout"]["scale"][2]       # 0.05

            # Calculate final placement Z-coordinate
            target_top_z = target_center_z + (target_height / 2)  # 0.025 + 0.025 = 0.05
            source_half_h = source_height / 2                      # 0.025
            place_z = target_top_z + source_half_h + 0.03          # 0.05 + 0.025 + 0.03 = 0.105

            # Target center (X,Y) remains same as red_cube's center
            place_position = [red_cube["layout"]["translate"][0],  # X: 0.5
                            red_cube["layout"]["translate"][1],  # Y: 0.0
                            place_z]                             # Z: 0.105

            # Execute pick-and-place
            robot.pick_and_place(pick_position, place_position)


            # ================================================================
            print("[Main] Task Sequence Completed.")
            task_executed = True

simulation_app.close()