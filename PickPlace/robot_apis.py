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

    # ... 保留你之前的 __init__ 和 _execute_controller_loop ...

    def move_to(self, target_position, rotation=None):
        """
        仅移动末端执行器到目标位置，不执行抓取或放置动作。
        常用于空机位移或观察特定点。
        """
        target_pos = np.array(target_position)
        # 技巧：将 pick 和 place 设为相同位置，控制器会直接走到该点并认为任务完成
        self._execute_controller_loop(target_pos, target_pos, rotation)
        print(f"[RobotAPI] Moved to {target_position}")
        return True

    def pick_at(self, pick_position, lift_height=0.2, rotation=None):
        """
        只执行抓取动作，并在抓取后将物体举起到指定高度。
        这对于需要“拿着物体移动”的长程任务非常有用。
        """
        pick_pos = np.array(pick_position)
        # 设定放置点为抓取点上方的悬停位置
        place_pos = pick_pos + np.array([0, 0, lift_height])
        self._execute_controller_loop(pick_pos, place_pos, rotation)
        return True

    def place_at(self, place_position, rotation=None):
        """
        假设当前已经抓持物体，将其放置到目标位置。
        """
        # 获取当前位置作为“抓取点”，这样控制器会跳过抓取阶段直接去 Place
        current_pos, _ = self.robot.end_effector.get_world_pose()
        self._execute_controller_loop(current_pos, np.array(place_position), rotation)
        return True

    def get_robot_state(self):
        """
        返回机器人当前的语义状态，供 LLM 判断下一步动作。
        """
        pos, rot = self.robot.end_effector.get_world_pose()
        joint_pos = self.robot.get_joint_positions()
        # 简单的夹爪状态逻辑：根据指间距判断是否闭合
        gripper_pos = self.robot.gripper.get_joint_positions()
        is_holding = np.mean(gripper_pos) < 0.01 
        
        state = {
            "ee_position": pos.tolist(),
            "ee_orientation": rot.tolist(),
            "joint_positions": joint_pos.tolist(),
            "is_holding_object": is_holding
        }
        return state

    def reset_robot(self):
        """
        将机器人复位到初始姿态。
        在 LLM 发现逻辑错误或任务结束时非常重要。
        """
        self.controller.reset()
        # 这里可以使用你定义的默认 Home 位姿
        home_pos = np.array([0.4, 0.0, 0.5]) 
        self.move_to(home_pos)
        print("[RobotAPI] Robot Reset to Home.")
        return True