1. Shutdown panda and move it to the desired grasp pose then Activate panda

2. Follow StartHere.md

3. Grasp object
ros2 run pymoveit2 panda_gripper_control

4. Run the heuristic fetch code
## 1. Vertical Retrieval
ros2 run pymoveit2 vertical_retrieval.py
## 2. Gripper Approach Direction Retreival
ros2 run pymoveit2 approach_direction_retrieval.py
