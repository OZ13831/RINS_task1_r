# Copyright 2023 Clearpath Robotics, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# @author Roni Kreinin (rkreinin@clearpathrobotics.com)

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

pkg_task1_real = get_package_share_directory('task1_real')
pkg_dis_tutorial3 = get_package_share_directory('dis_tutorial3')
pkg_turtlebot4_viz = get_package_share_directory('turtlebot4_viz')

ARGUMENTS = [
    DeclareLaunchArgument('namespace', default_value='', description='Robot namespace'),
    DeclareLaunchArgument('rviz', default_value='true', choices=['true', 'false'], description='Start rviz.'),
    DeclareLaunchArgument('use_sim_time', default_value='false', choices=['true', 'false'], description='use_sim_time'),
    DeclareLaunchArgument('map', default_value=PathJoinSubstitution([pkg_task1_real, 'maps', 'map1.yaml']), description='Full path to map yaml file to load'),
    DeclareLaunchArgument('localization_params', default_value=PathJoinSubstitution([pkg_task1_real, 'config', 'localization.yaml']), description='Localization parameters file'),
    DeclareLaunchArgument('nav2_params', default_value=PathJoinSubstitution([pkg_task1_real, 'config', 'nav2.yaml']), description='Nav2 parameters file')
]

def generate_launch_description():    
    # Launch Files
    localization_launch = PathJoinSubstitution([pkg_dis_tutorial3, 'launch', 'localization.launch.py'])
    nav2_launch = PathJoinSubstitution([pkg_dis_tutorial3, 'launch', 'nav2.launch.py'])
    rviz_launch = PathJoinSubstitution([pkg_turtlebot4_viz, 'launch', 'view_navigation.launch.py'])

    # Localization
    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([localization_launch]),
        launch_arguments=[
            ('namespace', LaunchConfiguration('namespace')),
            ('use_sim_time', LaunchConfiguration('use_sim_time')),
            ('map', LaunchConfiguration('map')),
            ('params', LaunchConfiguration('localization_params')),
        ]
    )

    # Nav2
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([nav2_launch]),
        launch_arguments=[
            ('namespace', LaunchConfiguration('namespace')),
            ('use_sim_time', LaunchConfiguration('use_sim_time')),
            ('params_file', LaunchConfiguration('nav2_params')),
        ]
    )

    laser_filter = Node(
        package='laser_filters',
        executable='scan_to_scan_filter_chain',
        parameters=[
            PathJoinSubstitution([
                pkg_task1_real,
                'config',
                'laser_filter_chain.yaml',
            ])
        ],
        remappings=[
            ('/scan_filtered', 'scan_filtered')
        ]
    )

    rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([rviz_launch]),
        launch_arguments=[
            ('namespace', LaunchConfiguration('namespace')),
            ('use_sim_time', LaunchConfiguration('use_sim_time'))
        ],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )


    # Create launch description and add actions
    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(localization)
    ld.add_action(nav2)
    ld.add_action(laser_filter)
    ld.add_action(rviz)
    return ld