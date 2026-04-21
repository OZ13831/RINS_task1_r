#! /usr/bin/env python3
# Mofidied from Samsung Research America
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


from enum import Enum
import time
import math

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Quaternion, PoseStamped, PoseWithCovarianceStamped, PointStamped
from visualization_msgs.msg import Marker
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import Spin, NavigateToPose
from turtle_tf2_py.turtle_tf2_broadcaster import quaternion_from_euler

from irobot_create_msgs.action import Dock, Undock
from irobot_create_msgs.msg import DockStatus

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration as rclpyDuration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data

from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point

class TaskResult(Enum):
    UNKNOWN = 0
    SUCCEEDED = 1
    CANCELED = 2
    FAILED = 3

amcl_pose_qos = QoSProfile(
          durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
          reliability=QoSReliabilityPolicy.RELIABLE,
          history=QoSHistoryPolicy.KEEP_LAST,
          depth=1)
FACE_APPROACH_OFFSET = 1.2 
FACE_DEDUP_RADIUS = 0.8
class RobotCommander(Node):

    def __init__(self, node_name='robot_commander', namespace=''):
        super().__init__(node_name=node_name, namespace=namespace)
        
        self.pose_frame_id = 'map'
        
        # Flags and helper variables
        self.goal_handle = None
        self.result_future = None
        self.feedback = None
        self.status = None
        self.initial_pose_received = False
        self.is_docked = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.seen_faces: list[tuple[float, float]] = []
        self.pending_faces: list[tuple[float, float]] = []  # drained by main loop

        # ROS2 subscribers
        self.create_subscription(DockStatus, 'dock_status', self._dockCallback, qos_profile_sensor_data)
        self.localization_pose_sub = self.create_subscription(PoseWithCovarianceStamped, 'amcl_pose', self._amclPoseCallback, amcl_pose_qos)
        self.create_subscription(Marker, "/people_marker", self._faceMarkerCallback, qos_profile_sensor_data)

        # ROS2 publishers
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped, 'initialpose', 10)
        
        # ROS2 Action clients
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.spin_client = ActionClient(self, Spin, 'spin')
        self.undock_action_client = ActionClient(self, Undock, 'undock')
        self.dock_action_client = ActionClient(self, Dock, 'dock')

        self.get_logger().info(f"Robot commander has been initialized!")
        
    def destroyNode(self):
        self.nav_to_pose_client.destroy()
        super().destroy_node()     

    def goToPose(self, pose, behavior_tree=''):
        """Send a `NavToPose` action request."""
        self.debug("Waiting for 'NavigateToPose' action server")
        while not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
            self.info("'NavigateToPose' action server not available, waiting...")

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        goal_msg.behavior_tree = behavior_tree

        self.info('Navigating to goal: ' + str(pose.pose.position.x) + ' ' +
                  str(pose.pose.position.y) + '...')
        send_goal_future = self.nav_to_pose_client.send_goal_async(goal_msg,
                                                                   self._feedbackCallback)
        rclpy.spin_until_future_complete(self, send_goal_future)
        self.goal_handle = send_goal_future.result()

        if not self.goal_handle.accepted:
            self.error('Goal to ' + str(pose.pose.position.x) + ' ' +
                       str(pose.pose.position.y) + ' was rejected!')
            return False

        self.result_future = self.goal_handle.get_result_async()
        return True

    def spin(self, spin_dist=1.57, time_allowance=10):
        self.debug("Waiting for 'Spin' action server")
        while not self.spin_client.wait_for_server(timeout_sec=1.0):
            self.info("'Spin' action server not available, waiting...")
        goal_msg = Spin.Goal()
        goal_msg.target_yaw = spin_dist
        goal_msg.time_allowance = Duration(sec=time_allowance)

        self.info(f'Spinning to angle {goal_msg.target_yaw}....')
        send_goal_future = self.spin_client.send_goal_async(goal_msg, self._feedbackCallback)
        rclpy.spin_until_future_complete(self, send_goal_future)
        self.goal_handle = send_goal_future.result()

        if not self.goal_handle.accepted:
            self.error('Spin request was rejected!')
            return False

        self.result_future = self.goal_handle.get_result_async()
        return True
    
    def undock(self):
        """Perform Undock action."""
        self.info('Undocking...')
        self.undock_send_goal()

        while not self.isUndockComplete():
            time.sleep(0.1)

    def undock_send_goal(self):
        goal_msg = Undock.Goal()
        self.undock_action_client.wait_for_server()
        goal_future = self.undock_action_client.send_goal_async(goal_msg)

        rclpy.spin_until_future_complete(self, goal_future)

        self.undock_goal_handle = goal_future.result()

        if not self.undock_goal_handle.accepted:
            self.error('Undock goal rejected')
            return

        self.undock_result_future = self.undock_goal_handle.get_result_async()

    def isUndockComplete(self):
        """
        Get status of Undock action.

        :return: ``True`` if undocked, ``False`` otherwise.
        """
        if self.undock_result_future is None or not self.undock_result_future:
            return True

        rclpy.spin_until_future_complete(self, self.undock_result_future, timeout_sec=0.1)

        if self.undock_result_future.result():
            self.undock_status = self.undock_result_future.result().status
            if self.undock_status != GoalStatus.STATUS_SUCCEEDED:
                self.info(f'Goal with failed with status code: {self.status}')
                return True
        else:
            return False

        self.info('Undock succeeded')
        return True

    def cancelTask(self):
        """Cancel pending task request of any type."""
        self.info('Canceling current task.')
        if self.result_future:
            future = self.goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, future)
        return

    def isTaskComplete(self):
        """Check if the task request of any type is complete yet."""
        if not self.result_future:
            # task was cancelled or completed
            return True
        rclpy.spin_until_future_complete(self, self.result_future, timeout_sec=0.10)
        if self.result_future.result():
            self.status = self.result_future.result().status
            if self.status != GoalStatus.STATUS_SUCCEEDED:
                self.debug(f'Task with failed with status code: {self.status}')
                return True
        else:
            # Timed out, still processing, not complete yet
            return False

        self.debug('Task succeeded!')
        return True

    def getFeedback(self):
        """Get the pending action feedback message."""
        return self.feedback

    def getResult(self):
        """Get the pending action result message."""
        if self.status == GoalStatus.STATUS_SUCCEEDED:
            return TaskResult.SUCCEEDED
        elif self.status == GoalStatus.STATUS_ABORTED:
            return TaskResult.FAILED
        elif self.status == GoalStatus.STATUS_CANCELED:
            return TaskResult.CANCELED
        else:
            return TaskResult.UNKNOWN

    def waitUntilNav2Active(self, navigator='bt_navigator', localizer='amcl'):
        """Block until the full navigation system is up and running."""
        self._waitForNodeToActivate(localizer)
        if not self.initial_pose_received:
            time.sleep(1)
        self._waitForNodeToActivate(navigator)
        self.info('Nav2 is ready for use!')
        return

    def _waitForNodeToActivate(self, node_name):
        # Waits for the node within the tester namespace to become active
        self.debug(f'Waiting for {node_name} to become active..')
        node_service = f'{node_name}/get_state'
        state_client = self.create_client(GetState, node_service)
        while not state_client.wait_for_service(timeout_sec=1.0):
            self.info(f'{node_service} service not available, waiting...')

        req = GetState.Request()
        state = 'unknown'
        while state != 'active':
            self.debug(f'Getting {node_name} state...')
            future = state_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            if future.result() is not None:
                state = future.result().current_state.label
                self.debug(f'Result of get_state: {state}')
            time.sleep(2)
        return
    
    def YawToQuaternion(self, angle_z = 0.):
        quat_tf = quaternion_from_euler(0, 0, angle_z)

        # Convert a list to geometry_msgs.msg.Quaternion
        quat_msg = Quaternion(x=quat_tf[0], y=quat_tf[1], z=quat_tf[2], w=quat_tf[3])
        return quat_msg

    def _amclPoseCallback(self, msg):
        self.debug('Received amcl pose')
        self.initial_pose_received = True
        self.current_pose = msg.pose
        return

    def _feedbackCallback(self, msg):
        self.debug('Received action feedback message')
        self.feedback = msg.feedback
        return
    
    def _dockCallback(self, msg: DockStatus):
        self.is_docked = msg.is_docked

    def setInitialPose(self, pose):
        msg = PoseWithCovarianceStamped()
        msg.pose.pose = pose
        msg.header.frame_id = self.pose_frame_id
        msg.header.stamp = 0
        self.info('Publishing Initial Pose')
        self.initial_pose_pub.publish(msg)
        return

    def info(self, msg):
        self.get_logger().info(msg)
        return

    def warn(self, msg):
        self.get_logger().warn(msg)
        return

    def error(self, msg):
        self.get_logger().error(msg)
        return

    def debug(self, msg):
        self.get_logger().debug(msg)
        return

# -- FACE FUNCTIONS --
    def _faceMarkerCallback(self, msg: Marker):
        """
        IMPORTANT: never call goToPose / spin from inside a callback —
        the executor is already spinning and will raise RuntimeError.
        Instead we just push the map-frame coordinate onto pending_faces;
        the main loop drains it between waypoints.
        """
        map_point = self._transform_to_map(msg)
        if map_point is None:
            return

        mx, my = map_point

        if self.is_face_already_seen(mx, my):
            return  # already queued or visited

        # Register immediately so duplicate markers don't flood the queue
        self._register_face(mx, my)
        self.pending_faces.append((mx, my))
        self.info(f"Face queued at map ({mx:.2f}, {my:.2f}).")
 
    # ── TF transform helper ────────────────────────────────────────────────────
 
    def _transform_to_map(self, msg: Marker) -> tuple[float, float] | None:
        """
        Transform the marker's pose from its source frame (usually /base_link)
        into the map frame.
 
        Returns (map_x, map_y) or None if the transform is unavailable.
        """
        source_frame = msg.header.frame_id.lstrip("/")   # strip leading slash
        target_frame = "map"
 
        try:
            # Look up the latest available transform (timeout 0.5 s)
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=rclpyDuration(seconds=0.5),
            )
        except Exception as e:
            self.warn(f"TF lookup failed ({source_frame} → map): {e}")
            return None
 
        # Wrap the marker position in a PointStamped so tf2 can transform it
        pt = PointStamped()
        pt.header = msg.header
        pt.point = msg.pose.position
 
        try:
            transformed = do_transform_point(pt, transform)
        except Exception as e:
            self.warn(f"TF transform failed: {e}")
            return None
 
        return transformed.point.x, transformed.point.y
 
    # ── Deduplication ──────────────────────────────────────────────────────────
 
    def is_face_already_seen(self, map_x: float, map_y: float) -> bool:
        """
        Return True if any previously-seen face is within FACE_DEDUP_RADIUS
        metres of (map_x, map_y).
        """
        for (sx, sy) in self.seen_faces:
            dist = math.hypot(map_x - sx, map_y - sy)
            if dist < FACE_DEDUP_RADIUS:
                return True
        return False
 
    def _register_face(self, map_x: float, map_y: float):
        """Record a face location so we never visit it again."""
        self.seen_faces.append((map_x, map_y))
        self.info(
            f"Registered face #{len(self.seen_faces)} at map "
            f"({map_x:.2f}, {map_y:.2f}).  "
            f"Total seen: {len(self.seen_faces)}"
        )
 
    # ── Face approach ──────────────────────────────────────────────────────────
 
    def navigate_to_face(self, map_x: float, map_y: float):
        """
        Build a goal pose in front of the detected face and send it to Nav2.
        Registration is already done by the callback before this is called.
        Returns True if the goal was accepted.
        """
        robot_x, robot_y = 0.0, 0.0
        if hasattr(self, "current_pose") and self.current_pose is not None:
            robot_x = self.current_pose.pose.position.x
            robot_y = self.current_pose.pose.position.y

        yaw_to_face = math.atan2(map_y - robot_y, map_x - robot_x)

        goal_x = map_x - FACE_APPROACH_OFFSET * math.cos(yaw_to_face)
        goal_y = map_y - FACE_APPROACH_OFFSET * math.sin(yaw_to_face)

        self.info(
            f"Approaching face: stopping at ({goal_x:.2f}, {goal_y:.2f}), "
            f"facing {math.degrees(yaw_to_face):.1f}°"
        )

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = "map"
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = goal_x
        goal_pose.pose.position.y = goal_y
        goal_pose.pose.orientation = self.YawToQuaternion(yaw_to_face)

        return self.goToPose(goal_pose)    
    
def main(args=None):
    
    rclpy.init(args=args)
    rc = RobotCommander()

    # Wait until Nav2 and Localizer are available
    rc.waitUntilNav2Active()

    # Check if the robot is docked, only continue when a message is recieved
    while rc.is_docked is None:
        rclpy.spin_once(rc, timeout_sec=0.5)

    # If it is docked, undock it first
    if rc.is_docked:
        rc.undock()

    #positions = [(1.294688288157884, -0.09709422018877921, -0.4138), (2.27034218163281, -1.7852592861839378, -2.0504), (1.31824248371742, -2.7561294271451726, -2.1587), (0.09428066039197096, -2.69596867399077, 1.3093), (-1.2348800314905855, -1.1274794638447014, -3.1163), (-2.212052807813255, 0.5644651064332631, 0.4210), (1.499812278130206, 1.5054969739486326, -0.9273), (-0.8546707403125438, 2.5830822945769194, -0.6689), (-1.613572287451908, -1.4718939457440627, 2.4253)]
    positions = [(0.09090706790517887, -2.823764257425934, -0.7057637729997648)]

    for (x, y, z) in positions:

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = rc.get_clock().now().to_msg()
        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y
        goal_pose.pose.orientation = rc.YawToQuaternion(z)

        rc.goToPose(goal_pose)

        while not rc.isTaskComplete():
            rclpy.spin_once(rc, timeout_sec=0.1)

        result = rc.getResult()
        rc.info(f"Waypoint result: {result}")

        # ── Drain any faces detected during transit ──────────────────────
        while rc.pending_faces:
            fx, fy = rc.pending_faces.pop(0)
            rc.info(f"Visiting queued face at ({fx:.2f}, {fy:.2f})")
            ok = rc.navigate_to_face(fx, fy)
            if ok:
                while not rc.isTaskComplete():
                    rclpy.spin_once(rc, timeout_sec=0.1)
                rc.info(f"Face visit result: {rc.getResult()}")
            else:
                rc.warn("Face approach goal rejected — skipping.")


    rc.destroyNode()

    # And a simple example
if __name__=="__main__":
    main()