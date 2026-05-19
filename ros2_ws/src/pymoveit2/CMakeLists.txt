cmake_minimum_required(VERSION 3.5)
project(pymoveit2)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install Python package
ament_python_install_package(pymoveit2)

# Install examples
set(EXAMPLES_DIR examples)
set(CUSTOM_SCRIPTS_DIR custom_scripts)
set(THINKGRASP_DIR thinkgrasp)
install(PROGRAMS
    ${EXAMPLES_DIR}/ex_allow_collisions.py
    ${EXAMPLES_DIR}/ex_clear_planning_scene.py
    ${EXAMPLES_DIR}/ex_collision_mesh.py
    ${EXAMPLES_DIR}/ex_collision_primitive.py
    ${EXAMPLES_DIR}/ex_fk.py
    ${EXAMPLES_DIR}/ex_gripper.py
    ${EXAMPLES_DIR}/ex_ik.py
    ${EXAMPLES_DIR}/ex_joint_goal.py
    ${EXAMPLES_DIR}/ex_orientation_path_constraint.py
    ${EXAMPLES_DIR}/ex_pose_goal.py
    ${EXAMPLES_DIR}/ex_servo.py
    ${CUSTOM_SCRIPTS_DIR}/panda_gripper_control.py
    ${CUSTOM_SCRIPTS_DIR}/sim_panda_gripper_control.py
    ${CUSTOM_SCRIPTS_DIR}/panda_joint_goal.py
    ${CUSTOM_SCRIPTS_DIR}/panda_pose_goal.py
    ${CUSTOM_SCRIPTS_DIR}/panda_ready.py
    ${THINKGRASP_DIR}/send_grasp_request.py
    ${THINKGRASP_DIR}/send_grasp_request_realrobot.py
    ${THINKGRASP_DIR}/grasp_pipeline_realrobot.py
    ${THINKGRASP_DIR}/sim_grasp_pipeline_realrobot.py
    DESTINATION lib/${PROJECT_NAME}
)

install(FILES
    ${THINKGRASP_DIR}/__init__.py
    ${THINKGRASP_DIR}/client.py
    ${THINKGRASP_DIR}/client_realrobot.py
    DESTINATION lib/${PROJECT_NAME}
)

# Setup the project
ament_package()
