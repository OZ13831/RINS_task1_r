#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy
from rclpy.duration import Duration
from rclpy.time import Time
from retinaface import RetinaFace
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
import tf2_ros
from tf2_ros import TransformException
from tf2_geometry_msgs import do_transform_point

from visualization_msgs.msg import Marker, MarkerArray

from cv_bridge import CvBridge, CvBridgeError
import cv2


class detect_faces(Node):

	def __init__(self):
		super().__init__('detect_faces')

		self.declare_parameters(
			namespace='',
			parameters=[
				('device', ''),
				('target_frame', 'map'),
				('assumed_face_width_m', 0.16),
				('inference_scale', 0.5),
				('show_image', True),
				('top_crop_ratio', 0.3),
				('bottom_crop_ratio', 0.3),
			],
		)

		marker_topic = "/people_markers"

		self.detection_color = (0, 0, 255)
		self.device = self.get_parameter('device').get_parameter_value().string_value
		self.target_frame = self.get_parameter('target_frame').get_parameter_value().string_value
		self.assumed_face_width_m = self.get_parameter('assumed_face_width_m').get_parameter_value().double_value
		self.inference_scale = self.get_parameter('inference_scale').get_parameter_value().double_value
		self.show_image = self.get_parameter('show_image').get_parameter_value().bool_value
		self.top_crop_ratio = self.get_parameter('top_crop_ratio').get_parameter_value().double_value
		self.bottom_crop_ratio = self.get_parameter('bottom_crop_ratio').get_parameter_value().double_value
		self.inference_scale = max(0.1, min(self.inference_scale, 1.0))
		self.top_crop_ratio = max(0.0, min(self.top_crop_ratio, 0.45))
		self.bottom_crop_ratio = max(0.0, min(self.bottom_crop_ratio, 0.45))

		self.bridge = CvBridge()

		self.fx = None
		self.fy = None
		self.cx = None
		self.cy = None
		self.camera_frame_id = None

		self.tf_buffer = tf2_ros.Buffer()
		self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

		self.rgb_image_sub = self.create_subscription(Image, "/gemini/color/image_raw", self.rgb_callback, qos_profile_sensor_data)
		self.camera_info_sub = self.create_subscription(CameraInfo, "/gemini/color/camera_info", self.camera_info_callback, qos_profile_sensor_data)

		self.marker_pub = self.create_publisher(MarkerArray, marker_topic, QoSReliabilityPolicy.BEST_EFFORT)

		self.faces = []
		self.processing = False

		self.get_logger().info(f"Node has been initialized! Will publish face markers to {marker_topic} in frame {self.target_frame}.")

	def camera_info_callback(self, data):
		if len(data.k) < 9:
			return

		self.fx = float(data.k[0])
		self.fy = float(data.k[4])
		self.cx = float(data.k[2])
		self.cy = float(data.k[5])
		self.camera_frame_id = data.header.frame_id

	def rgb_callback(self, data):
		if self.processing:
			return

		self.processing = True

		self.faces = []
		marker_array = MarkerArray()

		clear_marker = Marker()
		clear_marker.action = Marker.DELETEALL
		marker_array.markers.append(clear_marker)
		
		cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		height, width = cv_image.shape[:2]

		top_crop_px = int(height * self.top_crop_ratio)
		bottom_crop_px = int(height * self.bottom_crop_ratio)
		y_start = top_crop_px
		y_end = max(y_start + 1, height - bottom_crop_px)
		cropped_image = cv_image[y_start:y_end, :]
		try:

			if cropped_image.size == 0:
				self.marker_pub.publish(marker_array)
				return

			scaled_image = cv2.resize(cropped_image, (0, 0), fx=self.inference_scale, fy=self.inference_scale)
			res = RetinaFace.detect_faces(scaled_image)

			camera_frame = self.camera_frame_id if self.camera_frame_id is not None else data.header.frame_id
			transform = None
			if self.fx is not None and self.fy is not None and self.cx is not None and self.cy is not None:
				try:
					transform = self.tf_buffer.lookup_transform(
						self.target_frame,
						camera_frame,
						data.header.stamp,
						timeout=Duration(seconds=0.01)
					)
				except TransformException:
					try:
						transform = self.tf_buffer.lookup_transform(
							self.target_frame,
							camera_frame,
							Time(),
							timeout=Duration(seconds=0.01)
						)
					except TransformException as fallback_ex:
						self.get_logger().warn(
							f"Could not lookup transform {camera_frame} -> {self.target_frame}: {fallback_ex}"
						)

			if isinstance(res, dict):
				for marker_id, face_data in enumerate(res.values()):
					bbox = face_data.get("facial_area")
					if bbox is None or len(bbox) != 4:
						continue

					x1, y1, x2, y2 = [int(v / self.inference_scale) for v in bbox]
					y1 += y_start
					y2 += y_start

					x1 = max(0, min(x1, width - 1))
					y1 = max(0, min(y1, height - 1))
					x2 = max(0, min(x2, width - 1))
					y2 = max(0, min(y2, height - 1))

					cv_image = cv2.rectangle(cv_image, (x1, y1), (x2, y2), self.detection_color, 3)

					cx = int((x1 + x2) / 2)
					cy = int((y1 + y2) / 2)

					cv_image = cv2.circle(cv_image, (cx, cy), 5, self.detection_color, -1)

					self.faces.append((cx, cy))

					if self.fx is None or self.fy is None or self.cx is None or self.cy is None or transform is None:
						continue

					bbox_width_px = max(1.0, float(x2 - x1))
					z_cam = (self.assumed_face_width_m * self.fx) / bbox_width_px
					x_cam = ((float(cx) - self.cx) * z_cam) / self.fx
					y_cam = ((float(cy) - self.cy) * z_cam) / self.fy

					face_point_cam = PointStamped()
					face_point_cam.header.frame_id = camera_frame
					face_point_cam.header.stamp = data.header.stamp
					face_point_cam.point.x = x_cam
					face_point_cam.point.y = y_cam
					face_point_cam.point.z = z_cam

					face_point_map = do_transform_point(face_point_cam, transform)

					marker = Marker()
					marker.header.frame_id = self.target_frame
					marker.header.stamp = data.header.stamp
					marker.ns = "detected_faces"
					marker.id = marker_id
					marker.type = Marker.SPHERE
					marker.action = Marker.ADD
					marker.pose.position.x = float(face_point_map.point.x)
					marker.pose.position.y = float(face_point_map.point.y)
					marker.pose.position.z = float(face_point_map.point.z)
					marker.pose.orientation.w = 1.0
					marker.scale.x = 0.15
					marker.scale.y = 0.15
					marker.scale.z = 0.15
					marker.color.r = 1.0
					marker.color.g = 0.2
					marker.color.b = 0.2
					marker.color.a = 1.0

					marker_array.markers.append(marker)

			self.marker_pub.publish(marker_array)

			if self.show_image:
				cv2.imshow("image", cropped_image)
				key = cv2.waitKey(1)
				if key == 27:
					print("exiting")
					exit()

		except CvBridgeError as e:
			print(e)
			self.marker_pub.publish(marker_array)
		finally:
			self.processing = False


def main():
	print('Face detection node starting.')

	rclpy.init(args=None)
	node = detect_faces()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()
