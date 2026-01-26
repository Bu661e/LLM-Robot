# [1] 补全头部缺失的 Import (Lines 1-3)
import numpy as np
import typing
# 这是一个预留行，可能用于 import carb 或其他系统库，但在当前逻辑中 numpy 和 typing 是必须的

from isaacsim.core.api.controllers import BaseController
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka.controllers.stacking_controller import StackingController
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.robot.manipulators.examples.franka.controllers.pick_place_controller import PickPlaceController
from isaacsim.robot.manipulators.controllers import pick_place_controller
from isaacsim.robot.manipulators.controllers import stacking_controller
#from omni.isaac.franka.controllers import FrankaPickPlaceController


class ArmPickController(BaseController):
    def __init__(
        self,
        name: str,
        gripper: ParallelGripper,
        articulation: Articulation,
        picking_order_cube_names: typing.List[str],
        robot_observation_name: str,
    ) -> None:
        # Use the Franka StackingController as base
        super().__init__(name=name)
        self.pick_place_controller = PickPlaceController(
            name="pick_place_controller",
            gripper=gripper,
            robot_articulation=articulation)

        self.picking_order_cube_names = picking_order_cube_names
        self.robot_observation_name = robot_observation_name
        self.current_cube_numth = 0
        self.current_height = [0.0] * 3  # 对应3种颜色的目标位置高度
        # new add
        self.last_completed_cube_numth = -1  # 跟踪上一个完成的方块
        self.current_cube_start_time = 0  # 当前方块开始处理的时间
        self.max_cube_time = 1000  # 每个方块的最大处理时间 (步数)

    def forward(
        self,
        observations: dict,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
        end_effector_offset: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """ 控制器主循环：根据当前观测，决定机械臂的动作，并增加夹爪轨迹控制。"""

        # 1. 检查是否所有方块都已处理完成
        if self.current_cube_numth >= len(self.picking_order_cube_names):
            # 所有方块已完成，输出空动作
            target_joint_positions = [None] * observations[self.robot_observation_name]['joint_positions'].shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)

        # 2. 获取当前方块的颜色索引
        current_cube_name = self.picking_order_cube_names[self.current_cube_numth]
        color_idx = observations[current_cube_name]['color']
        placing_target_postion = observations['target_positions'][color_idx]
        placing_target_postion[2] = self.current_height[color_idx]+ observations[current_cube_name]['size'][2]/2 # 计算放置位置的高度

        # 调用PickPlaceController的forward方法
        cube_current_position = observations[current_cube_name]['position']        # 通过事件状态或空间距离判断夹爪动作
        robot_current_joint_position = observations[self.robot_observation_name]['joint_positions']
        actions = self.pick_place_controller.forward(
            picking_position=cube_current_position,
            placing_position=placing_target_postion,
            current_joint_positions=robot_current_joint_position,
            end_effector_orientation=end_effector_orientation,
            end_effector_offset=end_effector_offset,
        )

        if self.pick_place_controller.is_done():
            print(f"PickPlaceController reports done for cube {self.current_cube_numth} (color index {color_idx})")
            # 更新该颜色位置的高度 (加上方块高度)
            cube_size = observations[current_cube_name]['size']
            self.current_height[color_idx] += cube_size[2]
            self.last_completed_cube_numth = self.current_cube_numth

            # 移动到下一个方块
            self.current_cube_numth += 1
            
            # [2] 补全：每次完成一个方块后，必须重置内部的 pick_place_controller 状态，以便处理下一个
            self.pick_place_controller.reset()

        # [3] 补全：返回计算出的动作，否则机器人在仿真中不会移动
        return actions

    # [4] 补全：Reset 方法
    # 主程序 main_task_armpickplace.py 在 Line 57 调用了 my_controller.reset()
    def reset(self) -> None:
        """重置控制器状态"""
        self.current_cube_numth = 0
        self.current_height = [0.0] * 3
        self.last_completed_cube_numth = -1
        self.pick_place_controller.reset()
        print("ArmPickController has been reset.")