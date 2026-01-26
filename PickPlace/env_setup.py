from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.api.scenes import Scene
import numpy as np
from isaacsim.core.utils.stage import get_stage_units


class PickPlaceTask(BaseTask):
    """
    一个简单的 Pick & Place 任务类：
    - 固定位置的 Franka 机械臂
    - 固定位置的红色方块 (抓取目标)
    - 固定位置的蓝色方块 (放置目标/参考点)
    """
    def __init__(self, name="pick_place_task"):
        super().__init__(name=name, offset=None)
        
        # 缓存对象引用
        self._robot = None
        self._red_cube = None
        self._blue_cube = None
        
        # 定义关键参数
        self._stage_units = get_stage_units()
        self._cube_size = 0.05  # 5cm
        
        
        # 字典用于存储观测数据
        self._observations = {}

    def set_up_scene(self, scene: Scene) -> None:
        """
        初始化场景：加载机器人和方块
        """
        super().set_up_scene(scene)

        # 1. 添加 Franka 机器人

        try:
            from isaacsim.robot.manipulators.examples.franka import Franka
            self._robot = scene.add(
                Franka(
                    prim_path="/World/Franka",
                    name="franka",
                    position=np.array([0, 0, 0])
                )
            )
        except Exception as e:
            print(f"Failed to load Franka robot: {e}")
            print("Continuing without robot for now...")
            self._robot = None


        self._cube_size = 0.05  # 5cm
        # 2. 添加红色方块 (抓取物)
        self._red_cube = scene.add(
            DynamicCuboid(
                prim_path="/World/red_cube",
                name="red_cube",
                position=np.array([0.5, 0.0, self._cube_size / 2.0]),
                scale=np.array([self._cube_size] * 3) / self._stage_units,
                color=np.array([1.0, 0.0, 0.0]), # Red
            )
        )

        # 3. 添加蓝色方块 (目标点)
        self._blue_cube = scene.add(
            DynamicCuboid(
                prim_path="/World/blue_cube",
                name="blue_cube",
                position=np.array([0.5, 0.4, self._cube_size / 2.0]),
                scale=np.array([self._cube_size] * 3) / self._stage_units,
                color=np.array([0.0, 0.0, 1.0]), # Blue
            )
        )

        # 注册到 Task 的对象管理中 (这是 BaseTask 的推荐做法)
        self._task_objects[self._robot.name] = self._robot
        self._task_objects[self._red_cube.name] = self._red_cube
        self._task_objects[self._blue_cube.name] = self._blue_cube
        
        print(f"[PickPlaceTask] Scene setup complete.")
        print(f"  Red Cube: {np.array([0.5, 0.0, self._cube_size / 2.0])}")
        print(f"  Blue Cube: {np.array([0.5, 0.4, self._cube_size / 2.0])}")

    def get_observations(self) -> dict:
        """
        获取当前仿真步的观测数据
        这个返回值可以直接喂给 RobotAPI 作为 'Perception Data'
        """
        # 1. 机器人状态
        joint_pos = self._robot.get_joint_positions()
        ee_pos, ee_rot = self._robot.end_effector.get_world_pose()
        
        self._observations["robot"] = {
            "joint_positions": joint_pos,
            "end_effector_pos": ee_pos,
            "end_effector_rot": ee_rot
        }

        # 2. 红色方块位姿 (Ground Truth)
        cube_pos, cube_rot = self._red_cube.get_world_pose()
        self._observations["red_cube"] = {
            "pos": cube_pos,
            "rot": cube_rot,
            "color": "red"
        }

        # 3. 蓝色方块位姿
        target_pos, target_rot = self._blue_cube.get_world_pose()
        self._observations["blue_cube"] = {
            "pos": target_pos,
            "rot": target_rot,
            "color": "blue"
        }

        return self._observations

    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        """
        每个物理步之前调用，可用于更新逻辑
        """
        super().pre_step(time_step_index, simulation_time)
        # 这里可以添加一些动态干扰或检查逻辑，目前留空
        return

    def post_reset(self) -> None:
        """
        当 World.reset() 被调用后执行
        通常用于重置机器人的夹爪状态或物体位置
        """
        from isaacsim.robot.manipulators.grippers import ParallelGripper
        
        # 1. 如果机器人有夹爪，重置为张开状态
        if self._robot.gripper and isinstance(self._robot.gripper, ParallelGripper):
            self._robot.gripper.open()
            print("[PickPlaceTask] Gripper opened after reset.")
            
        # 2. 强制将物体放回初始位置 (防止它们在 Reset 后乱飞)
        self._red_cube.set_world_pose(position=np.array([0.5, 0.0, self._cube_size / 2.0]))
        self._blue_cube.set_world_pose(position=np.array([0.5, 0.4, self._cube_size / 2.0]) )
        
        print("[PickPlaceTask] Objects reset to initial positions.")

    def calculate_metrics(self) -> dict:
        """
        (可选) 计算任务指标，比如距离目标的误差
        """
        # 获取当前位置
        current_cube_pos, _ = self._red_cube.get_world_pose()
        target_pos, _ = self._blue_cube.get_world_pose()
        
        # 简单的欧氏距离：红色方块是否被放到了蓝色方块上？
        # 注意：这里我们只看 XY 平面的距离
        distance = np.linalg.norm(current_cube_pos[:2] - target_pos[:2])
        
        return {
            "distance_to_target": distance,
            "success": distance < 0.05 # 如果误差小于 5cm 视为成功
        }