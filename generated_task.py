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
