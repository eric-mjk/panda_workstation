## This is a forked repo

Original repo

panda-ros2 :  `https://github.com/tenfoldpaper/panda_ros2`

multipanda-ros2 : `https://github.com/tenfoldpaper/multipanda_ros2`



## Changelog

- Deleted the mongo_db variable in launch file
- Working setup for franka panda moveit launch (modified srdf and launch file for correct franka hand attachment)
- Edited the ros2_control for fake robot hardware (effort -> joint position) since initially there where no joint position update with effort in fake hardware
