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
import wave
from matplotlib.pyplot import rc
from piper import PiperVoice, SynthesisConfig
from pydub import AudioSegment
from pydub.playback import play
import numpy as np

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Quaternion, PoseStamped, PoseWithCovarianceStamped, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import Spin, NavigateToPose
from turtle_tf2_py.turtle_tf2_broadcaster import quaternion_from_euler
from std_msgs.msg import String

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
FACE_APPROACH_OFFSET = 0.4
FACE_DEDUP_RADIUS = 0.4

class RobotCommander(Node):

    def __init__(self, node_name='robot_commander', namespace=''):
        super().__init__(node_name=node_name, namespace=namespace)
        
        self.pose_frame_id = 'map'
        
        # Flags and helper variables
        self.goal_handle = None
        self.result_future = None
        self._cancel_future = None
        self.feedback = None
        self.status = None
        self.initial_pose_received = False
        self.is_docked = None
        self.going_to_face = False # če je zaznal obraz in gre proti njemu je TRue, sicer False
        self.going_to_ring = False
        self.recent_color = None
        self.safe = True  # becomes False if we send a new goal before the previous one finishes; used to suppress feedback/result of old goal

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.seen_faces: list[tuple[float, float]] = []
        self.seen_rings: list[tuple[float, float]] = []
        self.one_face_positions: list[tuple[float, float]] = []  # for averaging multiple detections of the same face
        self.one_ring_positions: list[tuple[float, float]] = []  # for averaging multiple detections of the same ring
        self._pending_face: tuple[float, float] | None = None  # set by callback, read by main loop
        self._pending_ring: tuple[float, float] | None = None  # set by callback, read by main loop
        self._interrupted_waypoint: tuple[float, float, float, float] | None = None  # (x, y, qz, qw)
        self.classified_colors = {"black": 0, "red": 0, "green": 0, "blue": 0}
        # ROS2 subscribers
        self.create_subscription(DockStatus, 'dock_status', self._dockCallback, qos_profile_sensor_data)
        self.localization_pose_sub = self.create_subscription(PoseWithCovarianceStamped, 'amcl_pose', self._amclPoseCallback, amcl_pose_qos)
        self.create_subscription(MarkerArray, "/people_markers", self._faceMarkerCallback, qos_profile_sensor_data)
        self.create_subscription(Marker, "/ring_xy_marker", self._ringMarkerCallback, qos_profile_sensor_data)
        self.create_subscription(String, "/ring_color", self._ringColorCallback, qos_profile_sensor_data)
        self.create_subscription(String, "/ring_predicted_color", self._ringColorCallback, qos_profile_sensor_data)
        # ROS2 publishers
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped, 'initialpose', 10)
        self.seen_pub = self.create_publisher(MarkerArray, '/seen_markers', QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1))
        self.speak_pub = self.create_publisher(String, '/speak', 10)
        self.one_publish = self.create_publisher(MarkerArray, '/one_markers', QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1))

        # ROS2 Action clients
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.spin_client = ActionClient(self, Spin, 'spin')
        self.undock_action_client = ActionClient(self, Undock, 'undock')
        self.dock_action_client = ActionClient(self, Dock, 'dock')

        self.get_logger().info(f"Robot commander has been initialized!")
        
    def destroyNode(self):
        self.nav_to_pose_client.destroy()
        self.speak_pub.destroy()
        super().destroy_node()     

    def goToPose(self, pose, behavior_tree=''):
        """Send a `NavToPose` action request."""
        self.debug("Waiting for 'NavigateToPose' action server")
        self.safe = False
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
        self.result_future.add_done_callback(self._goalDoneCallback)
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
        self.info('Canceling current task.')
        if self.result_future and self.goal_handle:
            self._cancel_future = self.goal_handle.cancel_goal_async()
        self.safe = True


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
    
    def _goalDoneCallback(self, future):
        """Callback when goToPose goal completes."""
        self.going_to_face = False
        self.going_to_ring = False
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
    def publish_seen(self):
        """
        Publish a MarkerArray containing a sphere for every confirmed seen face.
        Uses TRANSIENT_LOCAL so late-joining RViz subscribers receive the full
        history immediately upon connecting.
        """
        marker_array = MarkerArray()

        for i, (fx, fy) in enumerate(self.seen_faces):
            # ── sphere at the face position ──────────────────────────────────
            sphere = Marker()
            sphere.header.frame_id = "map"
            sphere.header.stamp = self.get_clock().now().to_msg()
            sphere.ns = "seen_faces"
            sphere.id = i * 2          # even ids → spheres
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = fx
            sphere.pose.position.y = fy
            sphere.pose.position.z = 0.3        # raise slightly above ground
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = 0.25
            sphere.scale.y = 0.25
            sphere.scale.z = 0.25
            sphere.color.r = 1.0
            sphere.color.g = 0.3
            sphere.color.b = 0.0
            sphere.color.a = 1.0
            sphere.lifetime.sec = 0             # 0 → marker lives forever
            marker_array.markers.append(sphere)

            # ── text label above the sphere ──────────────────────────────────
            label = Marker()
            label.header.frame_id = "map"
            label.header.stamp = sphere.header.stamp
            label.ns = "seen_faces_labels"
            label.id = i * 2 + 1       # odd ids → labels
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = fx
            label.pose.position.y = fy
            label.pose.position.z = 0.65
            label.pose.orientation.w = 1.0
            label.scale.z = 0.18        # text height in metres
            label.color.r = 1.0
            label.color.g = 1.0
            label.color.b = 1.0
            label.color.a = 1.0
            label.text = f"Face #{i + 1}"
            label.lifetime.sec = 0
            marker_array.markers.append(label)

        for i, (fx, fy) in enumerate(self.seen_rings):
            # ── sphere at the face position ──────────────────────────────────
            sphere = Marker()
            sphere.header.frame_id = "map"
            sphere.header.stamp = self.get_clock().now().to_msg()
            sphere.ns = "seen_rings"
            sphere.id = i * 2          # even ids → spheres
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = fx
            sphere.pose.position.y = fy
            sphere.pose.position.z = 0.3        # raise slightly above ground
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = 0.25
            sphere.scale.y = 0.25
            sphere.scale.z = 0.25
            sphere.color.r = 0.0
            sphere.color.g = 0.8
            sphere.color.b = 0.1
            sphere.color.a = 1.0
            sphere.lifetime.sec = 0             # 0 → marker lives forever
            marker_array.markers.append(sphere)

            # ── text label above the sphere ──────────────────────────────────
            label = Marker()
            label.header.frame_id = "map"
            label.header.stamp = sphere.header.stamp
            label.ns = "seen_rings_labels"
            label.id = i * 2 + 1       # odd ids → labels
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = fx
            label.pose.position.y = fy
            label.pose.position.z = 0.65
            label.pose.orientation.w = 1.0
            label.scale.z = 0.18        # text height in metres
            label.color.r = 1.0
            label.color.g = 1.0
            label.color.b = 1.0
            label.color.a = 1.0
            label.text = f"Ring #{i + 1}"
            label.lifetime.sec = 0
            marker_array.markers.append(label)

        self.seen_pub.publish(marker_array)
        self.info(f"Published {len(self.seen_rings)} seen-ring marker(s) to /seen_markers")

    def _faceMarkerCallback(self, msg: MarkerArray):
        for marker in msg.markers:
            if marker.action != Marker.ADD:
                continue
            if marker.ns != "detected_faces":
                continue
            if marker.type != Marker.SPHERE:
                continue

            map_point = self._transform_point_to_map(
                marker.pose.position.x,
                marker.pose.position.y,
                marker.pose.position.z,
                marker.header,
            )
            if map_point is None:
                continue

            mx, my = map_point
            if self.is_face_already_seen(mx, my):
                continue

            avg_x, avg_y = self._register_face(mx, my)
            if avg_x is not None and self._pending_face is None:
                self._pending_face = (avg_x, avg_y)
                self.info(f"Face detected at map ({avg_x:.2f}, {avg_y:.2f}), queued.")
 
    def _ringMarkerCallback(self, msg: Marker):
        marker = msg
        if marker.action != Marker.ADD:
            return

        map_point = self._transform_point_to_map(
            marker.pose.position.x,
            marker.pose.position.y,
            marker.pose.position.z,
            marker.header,
        )
        if map_point is None:
            return

        mx, my = map_point

        if self.is_ring_already_seen(mx, my):
            return

        if self.recent_color in self.classified_colors:
            self.classified_colors[self.recent_color] += 1

        avg_x, avg_y = self._register_ring(mx, my)
        if avg_x is not None and self._pending_ring is None:
            self._pending_ring = (avg_x, avg_y)
            self.info(f"Ring detected at map ({avg_x:.2f}, {avg_y:.2f}), queued.")


    def _ringColorCallback(self, msg: String):
        self.recent_color = msg.data
        print(f"Classified colors: {self.classified_colors}")
        print("One ring positions:", len(self.one_ring_positions))
        print("Goint to ring:", self.going_to_ring)

    # ── TF transform helper ────────────────────────────────────────────────────
 
    def _transform_point_to_map(self, x: float, y: float, z: float, header) -> tuple[float, float] | None:
        """
        Transform a point from its source frame
        into the map frame.
 
        Returns (map_x, map_y) or None if the transform is unavailable.
        """
        source_frame = header.frame_id.lstrip("/") if header.frame_id else "map"
        target_frame = "map"

        if source_frame == target_frame:
            return float(x), float(y)
 
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
        pt.header = header
        pt.point.x = float(x)
        pt.point.y = float(y)
        pt.point.z = float(z)
 
        try:
            transformed = do_transform_point(pt, transform)
        except Exception as e:
            self.warn(f"TF transform failed: {e}")
            return None
 
        return transformed.point.x, transformed.point.y
 
    # ── Deduplication ──────────────────────────────────────────────────────────
    def is_ring_already_seen(self, map_x: float, map_y: float) -> bool:
        """
        Return True if any previously-seen ring is within FACE_DEDUP_RADIUS
        metres of (map_x, map_y).
        """


        for ring in self.seen_rings:
            dist = math.hypot(map_x - ring[0], map_y - ring[1])
            if dist < FACE_DEDUP_RADIUS:
                return True
        return False


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
    
    def _register_ring(self, map_x: float, map_y: float):
        """Record a ring location so we never visit it again."""
        if not self.going_to_ring:
            self.one_ring_positions.append((map_x, map_y))
            # print()
            # print("TU SM")
            # print()

        if len(self.one_ring_positions) > 15 and not self.going_to_face:
            self.going_to_ring = True
            median_x = np.median([p[0] for p in self.one_ring_positions])
            median_y = np.median([p[1] for p in self.one_ring_positions])
            self.seen_rings.append((median_x, median_y))
            self.info(
                f"Registered ring at map "
                f"({map_x:.2f}, {map_y:.2f}).  "
                f"Total seen: {len(self.seen_rings)}"
            )

            self.one_ring_positions.clear()
            self.publish_seen()
            return median_x, median_y
        return None, None


    def  publish_one_seen(self):
        marker_array = MarkerArray()

        for i, (fx, fy) in enumerate(self.one_face_positions):
            # ── sphere at the face position ──────────────────────────────────
            sphere = Marker()
            sphere.header.frame_id = "map"
            sphere.header.stamp = self.get_clock().now().to_msg()
            sphere.ns = "seen_faces"
            sphere.id = i * 2          # even ids → spheres
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = fx
            sphere.pose.position.y = fy
            sphere.pose.position.z = 0.3        # raise slightly above ground
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = 0.25
            sphere.scale.y = 0.25
            sphere.scale.z = 0.25
            sphere.color.r = 1.0
            sphere.color.g = 0.3
            sphere.color.b = 0.0
            sphere.color.a = 1.0
            sphere.lifetime.sec = 0             # 0 → marker lives forever
            marker_array.markers.append(sphere)

            # ── text label above the sphere ──────────────────────────────────
            label = Marker()
            label.header.frame_id = "map"
            label.header.stamp = sphere.header.stamp
            label.ns = "seen_faces_labels"
            label.id = i * 2 + 1       # odd ids → labels
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = fx
            label.pose.position.y = fy
            label.pose.position.z = 0.65
            label.pose.orientation.w = 1.0
            label.scale.z = 0.18        # text height in metres
            label.color.r = 1.0
            label.color.g = 1.0
            label.color.b = 1.0
            label.color.a = 1.0
            label.text = f"Face #{i + 1}"
            label.lifetime.sec = 0
            marker_array.markers.append(label)
        self.one_publish.publish(marker_array)

    def _register_face(self, map_x: float, map_y: float):
        """Record a face location so we never visit it again."""
        #self.seen_faces.append((map_x, map_y))
        if not self.going_to_face:
            self.one_face_positions.append((map_x, map_y))
            self.publish_one_seen()

        if len(self.one_face_positions) > 25 and not self.going_to_ring:
            self.going_to_face = True
            median_x = np.median([p[0] for p in self.one_face_positions])
            median_y = np.median([p[1] for p in self.one_face_positions])
            self.seen_faces.append((median_x, median_y))
            self.one_face_positions.clear()
            self.info(

                f"Registered face #{len(self.seen_faces)} at map "
                f"({map_x:.2f}, {map_y:.2f}).  "
                f"Total seen: {len(self.seen_faces)}"
            )
            self.publish_seen()
            return median_x, median_y
        return None, None
 
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
def make_pose(rc, x, y, qz, qw):
    pose = PoseStamped()
    pose.header.frame_id = "map"
    pose.header.stamp = rc.get_clock().now().to_msg()
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=qz, w=qw)
    return pose
def main(args=None):
    positions = [
    #(-0.1507724543037919, -0.23072254256144112, 0.010314225539944995, 0.9999468069610059),
    (-0.9198282301753373, -0.10762984551283342, 0.8100609432123049, 0.5863456900856276),
    (-1.5866502778489406, 0.7317233527968411, -0.5793155795278797, 0.8151033427218152),
    (-2.2618125454904896, -0.06564286960134255, 0.958429748974962, 0.28532861104311164),
    (-3.1314396350047202, 0.4414679464953455, -0.6794658614896312, 0.7337071234969396),
    (-3.2835393036433405, -0.10195187443663478, 0.5270695181045316, 0.8498221714482723),
    (-3.716156298638488, -0.20608935460419006, -0.9949289061152515, 0.10058067297601933),
    ]
    rclpy.init(args=args)
    rc = RobotCommander()
    # Wait until Nav2 and Localizer are available
    rc.waitUntilNav2Active()

    # If it is docked, undock it first
    if rc.is_docked:
        print("robot is docked")
        rc.undock()
        print("robot is undocked")
    print("Not docked", rc.is_docked)
    
    interupted = False
    count_faces = 0
    count_rings = 0

    print("starting execution")
    while count_faces < 3 or count_rings < 2:
        for (x, y, qz, qw) in positions:
            goal_pose = make_pose(rc, x, y, qz, qw)
            rc.goToPose(goal_pose)
            print("Going to waypoint: ", x, y)

            while not rc.isTaskComplete():
                rclpy.spin_once(rc, timeout_sec=0.1)

            #     if rc._pending_ring is not None:
            #         ring_x, ring_y = rc._pending_ring
            #         rc._pending_ring = None

            #         rc._interrupted_waypoint = (x, y, qz, qw)
            #         rc.cancelTask()
            #         rc.navigate_to_face(ring_x, ring_y)

            #         while not rc.isTaskComplete():
            #             rclpy.spin_once(rc, timeout_sec=0.1)
            #         count_rings += 1
            #         rc.info("MADE IT TO RING")
            #         temp = None
            #         maxCount = -1
            #         print("CLASSIFIED COLORS: ", rc.classified_colors)
            #         for color, count in rc.classified_colors.items():
            #             if count > maxCount:
            #                 maxCount = count
            #                 temp = color
            #         if temp is not None:
            #             rc.info(f"Most likely color is {temp} with {maxCount} classifications.")
            #             speak_msg = String()
            #             speak_msg.data = f"I see a {temp} ring."
            #             rc.speak_pub.publish(speak_msg)
            #         #rc.seen_rings.append((ring_x, ring_y))
            #         for color, count in rc.classified_colors.items():
            #             rc.classified_colors[color] = 0
            #         sleep_duration = 0.3
            #         rc.info(f"Robot staying in place for {sleep_duration} seconds...")
            #         time.sleep(sleep_duration)
            #         rc.info("Ring visit done, resuming waypoint...")
                
            #         interupted = True


            #     if rc._pending_face is not None:
            #         face_x, face_y = rc._pending_face
            #         rc._pending_face = None

            #         rc._interrupted_waypoint = (x, y, qz, qw)
            #         rc.cancelTask()
            #         rc.navigate_to_face(face_x, face_y)

            #         while not rc.isTaskComplete():
            #             rclpy.spin_once(rc, timeout_sec=0.1)
            #         rc.info("MADE IT TO FACE")
            #         rc.speak_pub.publish(String(data="Hello, human!"))
            #         sleep_duration = 0.4
            #         rc.info(f"Robot staying in place for {sleep_duration} seconds...")
            #         time.sleep(sleep_duration)
            #         rc.info("Face visit done, resuming waypoint...")
            #         count_faces += 1
            #         interupted = True
            #     if count_faces >= 3 and count_rings >= 2:
            #         break

            #     if interupted:
            #         ix, iy, iqz, iqw = rc._interrupted_waypoint
            #         rc._interrupted_waypoint = None
            #         goal_pose = make_pose(rc, ix, iy, iqz, iqw)
            #         rc.goToPose(goal_pose)

            #         interupted = False

            result = rc.getResult()
            rc.info(f"Waypoint result: {result}")
            if count_faces >= 3 and count_rings >= 2:
                break

    rc.destroyNode()
    # And a simple example
if __name__=="__main__":
    main()