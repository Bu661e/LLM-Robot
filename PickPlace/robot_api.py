# robot_api.py
import numpy as np
from omni.isaac.franka.controllers import PickPlaceController

class LLMRobotWrapper:
    def __init__(self, world, robot):
        self.world = world
        self.robot = robot
        
        self.controller = PickPlaceController(
            name="pick_place_controller",
            gripper=robot.gripper,
            robot_articulation=robot
        )

    def _execute_controller_loop(self, pick_pos, place_pos, pick_rot=None):
        """
        内部私有方法：运行控制器的 While 循环
        """
        self.controller.reset()
        
        # 默认抓取姿态：垂直向下 (针对 Franka)
        # 如果调用时没有指定旋转，则使用默认值
        if pick_rot is None:
            end_effector_orientation = np.array([0, 1, 0, 0]) 
        else:
            end_effector_orientation = pick_rot

        print(f"[RobotAPI] Executing Motion:")
        print(f"  -> Pick At:  {pick_pos}")
        print(f"  -> Place At: {place_pos}")

        while self.world.is_playing():
            # 1. 获取当前关节状态
            current_joints = self.robot.get_joint_positions()
            
            # 2. 计算控制指令
            actions = self.controller.forward(
                picking_position=pick_pos,
                placing_position=place_pos,
                current_joint_positions=current_joints,
                end_effector_orientation=end_effector_orientation,
                end_effector_offset=np.array([0, 0, 0.0]) # 控制器内部的微调偏移，通常设为0
            )
            
            # 3. 应用动作
            self.robot.apply_action(actions)
            self.world.step(render=True)
            
            # 4. 判断结束
            if self.controller.is_done():
                print("[RobotAPI] Action Completed.")
                break
        
        # 动作完成后稳定一下
        for _ in range(10):
            self.world.step(render=True)

    def pick_and_place(self, pick_position, place_position, rotation=None):
        """
        直接接收坐标数据的 API
        
        Args:
            pick_position (list or np.array): [x, y, z] 抓取点的世界坐标
            place_position (list or np.array): [x, y, z] 放置点的世界坐标
            rotation (list or np.array, optional): [w, x, y, z] 抓取时的末端姿态（四元数）
        """
        # 确保输入是 Numpy 数组，方便计算
        pick_pos = np.array(pick_position)
        place_pos = np.array(place_position)
        
        if rotation is not None:
            rot = np.array(rotation)
        else:
            rot = None

        # 调用执行循环
        self._execute_controller_loop(pick_pos, place_pos, rot)
        return True