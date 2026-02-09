from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.api.scenes import Scene
import numpy as np
from isaacsim.core.utils.stage import get_stage_units


class PickPlaceTaskWithCam(BaseTask):
    """
    在原有 Pick & Place 场景基础上，增加一个全局深度摄像头
    - 固定位置的 Franka 机械臂
    - 固定位置的红色方块 (抓取目标)
    - 固定位置的蓝色方块 (放置目标/参考点)
    - 全局深度相机 (RGB + Depth)
    """
    def __init__(self, name="pick_place_task_with_cam"):
        super().__init__(name=name, offset=None)
        
        # 缓存对象引用
        self._robot = None
        self._red_cube = None
        self._blue_cube = None
        self._camera = None
        
        # 定义关键参数
        self._stage_units = get_stage_units()
        self._cube_size = 0.05  # 5cm
        
        # 字典用于存储观测数据
        self._observations = {}

    def _create_global_camera(self, scene: Scene):
        """
        创建全局深度相机
        """
        try:
            from isaacsim.sensors.camera import Camera
        except Exception:
            from omni.isaac.sensor import Camera

        self._camera = Camera(
            prim_path="/World/GlobalCamera",
            name="global_camera",
            position=np.array([0, 0.5, 5]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            frequency=30,
            resolution=(640, 480),
        )
        scene.add(self._camera)

        if hasattr(self._camera, "initialize"):
            self._camera.initialize()

        if hasattr(self._camera, "add_depth_to_sensors"):
            self._camera.add_depth_to_sensors()
        elif hasattr(self._camera, "set_depth_enabled"):
            self._camera.set_depth_enabled(True)

        if hasattr(self._camera, "set_world_pose"):
            self._camera.set_world_pose(
                position=np.array([0.8, 0.0, 0.8]),
                orientation=np.array([0.7071, 0.0, 0.7071, 0.0]),
            )

        if hasattr(self._camera, "name"):
            self._task_objects[self._camera.name] = self._camera

        print("[PickPlaceTaskWithCam] Global camera created at [0.8, 0.0, 0.8].")

    def set_up_scene(self, scene: Scene) -> None:
        """
        初始化场景：加载机器人、方块和全局深度相机
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
                color=np.array([1.0, 0.0, 0.0]),  # Red
            )
        )

        # 3. 添加蓝色方块 (目标点)
        self._blue_cube = scene.add(
            DynamicCuboid(
                prim_path="/World/blue_cube",
                name="blue_cube",
                position=np.array([0.5, 0.4, self._cube_size / 2.0]),
                scale=np.array([self._cube_size] * 3) / self._stage_units,
                color=np.array([0.0, 0.0, 1.0]),  # Blue
            )
        )

        # 4. 添加全局深度相机
        self._create_global_camera(scene)

        # 注册到 Task 的对象管理中 (这是 BaseTask 的推荐做法)
        if self._robot is not None:
            self._task_objects[self._robot.name] = self._robot
        self._task_objects[self._red_cube.name] = self._red_cube
        self._task_objects[self._blue_cube.name] = self._blue_cube
        
        print("[PickPlaceTaskWithCam] Scene setup complete.")
        print(f"  Red Cube: {np.array([0.5, 0.0, self._cube_size / 2.0])}")
        print(f"  Blue Cube: {np.array([0.5, 0.4, self._cube_size / 2.0])}")

    def get_observations(self) -> dict:
        """
        获取当前仿真步的观测数据
        """
        # 1. 机器人状态
        if self._robot is not None:
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

        # 4. 相机观测
        if self._camera is not None:
            rgb = None
            depth = None

            if hasattr(self._camera, "get_rgba"):
                rgba = self._camera.get_rgba()
                rgb = rgba[..., :3] if rgba is not None else None
            elif hasattr(self._camera, "get_rgb"):
                rgb = self._camera.get_rgb()

            if hasattr(self._camera, "get_depth"):
                depth = self._camera.get_depth()

            self._observations["camera"] = {
                "rgb": rgb,
                "depth": depth,
            }

        return self._observations

    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        """
        每个物理步之前调用，可用于更新逻辑
        """
        super().pre_step(time_step_index, simulation_time)
        return

    def post_reset(self) -> None:
        """
        当 World.reset() 被调用后执行
        通常用于重置机器人的夹爪状态或物体位置
        """
        from isaacsim.robot.manipulators.grippers import ParallelGripper
        
        if self._robot is not None and self._robot.gripper and isinstance(self._robot.gripper, ParallelGripper):
            self._robot.gripper.open()
            print("[PickPlaceTaskWithCam] Gripper opened after reset.")
            
        self._red_cube.set_world_pose(position=np.array([0.5, 0.0, self._cube_size / 2.0]))
        self._blue_cube.set_world_pose(position=np.array([0.5, 0.4, self._cube_size / 2.0]))
        
        print("[PickPlaceTaskWithCam] Objects reset to initial positions.")

    def calculate_metrics(self) -> dict:
        """
        (可选) 计算任务指标，比如距离目标的误差
        """
        current_cube_pos, _ = self._red_cube.get_world_pose()
        target_pos, _ = self._blue_cube.get_world_pose()
        
        distance = np.linalg.norm(current_cube_pos[:2] - target_pos[:2])
        
        return {
            "distance_to_target": distance,
            "success": distance < 0.05
        }
