#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.duration import Duration as rclpyDuration
from rclpy.time import Time
import message_filters
from ultralytics import YOLO
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String
from geometry_msgs.msg import Point
from builtin_interfaces.msg import Duration as RosDuration
import tf2_ros
from tf2_ros import Buffer, TransformListener
from tf2_ros import TransformException
from tf2_geometry_msgs import do_transform_point

from visualization_msgs.msg import Marker, MarkerArray

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import torch


def cleanup_mask(mask):
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
	erosion = cv2.erode(mask, kernel, iterations=1)
	dilation = cv2.dilate(erosion, kernel, iterations=1)
	return dilation


def predict_color(ring_img, mask=None):
	if mask is None or np.sum(mask) == 0:
		return ""
	mask = cleanup_mask(mask)

	hist_folder = "/home/gamma/colcon_ws/avg_hist2"
	if not os.path.exists(hist_folder):
		os.makedirs(hist_folder)

	if mask.shape[:2] != ring_img.shape[:2]:
		mask = cv2.resize(mask, (ring_img.shape[1], ring_img.shape[0]))

	hsv = cv2.cvtColor(ring_img, cv2.COLOR_BGR2HSV)
	h, s, v = cv2.split(hsv)

	v_masked = v[mask > 0]

	if np.mean(v_masked) < 85:
		return "black"

	hist = cv2.calcHist([h], [0], mask, [180], [0, 180])
	hist = cv2.normalize(hist, hist).flatten()

	histograms = os.listdir(hist_folder)
	best_match = None
	second_best = None

	best_score = float("inf")
	for hist_file in histograms:
		if hist_file.endswith(".npy"):
			hist_path = os.path.join(hist_folder, hist_file)
			hist_data = np.load(hist_path)
			score = np.sqrt(0.5 * np.sum((np.sqrt(hist) - np.sqrt(hist_data)) ** 2))
			color_name = hist_file.split(".")[0]
			if score < best_score:
				second_best = best_score
				best_score = score
				best_match = color_name

	iowe_ratio = second_best / best_score if second_best is not None else float(1)
	colors = ["red", "green", "blue"]
	for color in colors:
		if best_match and best_match.startswith(color) and iowe_ratio > 1.2:
			return color
	return ""


class detect_faces(Node):

	def __init__(self):
		super().__init__('detect_faces')

		self.declare_parameters(
			namespace='',
			parameters=[
				('device', ''),
				('model_path', 'yolov8n-face.pt'),
				('upper_ratio', 0.5),
				('depth_topic', '/gemini/depth/image_raw'),
				('camera_info_topic', '/gemini/color/camera_info'),
				('map_topic', '/map'),
				('map_bounds_padding', 0.3),
				('map_bounds_hard', False),
				('wall_distance_m', 0.25),
				('target_frame', 'map'),
				('show_image', True),
				('top_crop_ratio', 0.3),
				('bottom_crop_ratio', 0.3),
			],
		)

		marker_topic = "/people_markers"

		self.detection_color = (0, 0, 255)
		self.device = self.get_parameter('device').get_parameter_value().string_value
		self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
		self.upper_ratio = self.get_parameter('upper_ratio').get_parameter_value().double_value
		self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
		self.camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
		self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
		self.map_bounds_padding = self.get_parameter('map_bounds_padding').get_parameter_value().double_value
		self.map_bounds_hard = self.get_parameter('map_bounds_hard').get_parameter_value().bool_value
		self.wall_distance_m = self.get_parameter('wall_distance_m').get_parameter_value().double_value
		self.target_frame = self.get_parameter('target_frame').get_parameter_value().string_value
		self.show_image = self.get_parameter('show_image').get_parameter_value().bool_value
		self.top_crop_ratio = self.get_parameter('top_crop_ratio').get_parameter_value().double_value
		self.bottom_crop_ratio = self.get_parameter('bottom_crop_ratio').get_parameter_value().double_value
		self.top_crop_ratio = max(0.0, min(self.top_crop_ratio, 0.45))
		self.bottom_crop_ratio = max(0.0, min(self.bottom_crop_ratio, 0.45))

		self.bridge = CvBridge()

		self.latest_depth_msg = None
		self.latest_depth_image = None
		self.full_depth_image_m = None
		self.full_depth_header = None
		self.depth_y_start = 0
		self.depth_y_end = 0
		self.fx = None
		self.fy = None
		self.cx = None
		self.cy = None
		self.camera_frame_id = None
		self.map_min_x = None
		self.map_max_x = None
		self.map_min_y = None
		self.map_max_y = None
		self.map_resolution = None
		self.map_width = None
		self.map_height = None
		self.map_origin_x = None
		self.map_origin_y = None
		self.map_data = None

		self.tf_buffer = tf2_ros.Buffer()
		self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

		map_qos = QoSProfile(
			durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
			reliability=QoSReliabilityPolicy.RELIABLE,
			history=QoSHistoryPolicy.KEEP_LAST,
			depth=1,
		)

		#self.rgb_image_sub = self.create_subscription(Image, "/gemini/color/image_raw", self.rgb_callback, qos_profile_sensor_data)
		#self.depth_sub = self.create_subscription(Image, self.depth_topic, self.depth_callback, qos_profile_sensor_data)
		self.camera_info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, qos_profile_sensor_data)
		self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic, self.map_callback, map_qos)

		self.rgb_sync_sub = message_filters.Subscriber(self, Image, "/gemini/color/image_raw", qos_profile=qos_profile_sensor_data)
		self.depth_sync_sub = message_filters.Subscriber(self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data)
		self.rgb_depth_sync = message_filters.ApproximateTimeSynchronizer(
			[self.rgb_sync_sub, self.depth_sync_sub],
			queue_size=10,
			slop=0.1
		)
		self.rgb_depth_sync.registerCallback(self.rgb_depth_callback)

		self.mode = "face"
		self.mode_sub = self.create_subscription(String, "/mode", self.mode_callback, qos_profile_sensor_data)

		self.marker_pub = self.create_publisher(Marker, marker_topic, QoSReliabilityPolicy.BEST_EFFORT)
		self.color_pub = self.create_publisher(String, "/ring_color", qos_profile_sensor_data)
		self.ring_pub = self.create_publisher(Marker, "/rings", qos_profile_sensor_data)

		if torch.cuda.is_available():
			self.device = "cuda:0"
			self.get_logger().info("CUDA is available. Forcing YOLO to use cuda:0.")
		else:
			self.device = "cpu"
			self.get_logger().warn("CUDA not available. Falling back to CPU for YOLO.")

		self.model = YOLO(self.model_path)

		self.processing = False

		self.get_logger().info(
			f"Node has been initialized! Will publish face markers to {marker_topic} in frame {self.target_frame} using depth topic {self.depth_topic}."
		)
		self.get_logger().info(f"Waiting for map on {self.map_topic} to enable map bounds checking.")

	def rgb_depth_callback(self, rgb_msg, depth_msg):
		self.depth_callback(depth_msg)
		self.rgb_callback(rgb_msg)

	def mode_callback(self, msg):
		mode = msg.data.strip().lower()
		if mode in ("face", "ring"):
			self.mode = mode

	def map_callback(self, data):
		# Cache map boundaries for quick inside-map checks.
		print("WAVE")
		if data.info.width == 0 or data.info.height == 0 or data.info.resolution <= 0.0:
			print("Map update ignored: invalid dimensions or resolution.")
			return

		origin_x = float(data.info.origin.position.x)
		origin_y = float(data.info.origin.position.y)
		width_m = float(data.info.width) * float(data.info.resolution)
		height_m = float(data.info.height) * float(data.info.resolution)

		self.map_min_x = origin_x
		self.map_max_x = origin_x + width_m
		self.map_min_y = origin_y
		self.map_max_y = origin_y + height_m
		self.map_resolution = float(data.info.resolution)
		self.map_width = int(data.info.width)
		self.map_height = int(data.info.height)
		self.map_origin_x = origin_x
		self.map_origin_y = origin_y
		self.map_data = data.data
		print(f"Map updated: size=({self.map_width}x{self.map_height}), res={self.map_resolution}, origin=({self.map_origin_x:.2f},{self.map_origin_y:.2f}).")

	def _is_near_wall(self, x, y):
		if (
			self.map_data is None
			or self.map_resolution is None
			or self.map_width is None
			or self.map_height is None
			or self.map_origin_x is None
			or self.map_origin_y is None
		):
			print(f"map_resolution: {self.map_resolution}, map_width: {self.map_width}, map_height: {self.map_height}, map_origin_x: {self.map_origin_x}, map_origin_y: {self.map_origin_y}")
			return False

		resolution = float(self.map_resolution)
		if resolution <= 0.0:
			print("Invalid map resolution, cannot check for walls.")
			return False

		grid_x = int((x - self.map_origin_x) / resolution)
		grid_y = int((y - self.map_origin_y) / resolution)
		if grid_x < 0 or grid_y < 0 or grid_x >= self.map_width or grid_y >= self.map_height:
			print(f"Point ({x:.2f}, {y:.2f}) is outside map grid, cannot check for walls.")
			print(f"Computed grid coordinates: ({grid_x}, {grid_y}), map size: ({self.map_width}, {self.map_height}).")
			return False

		radius_cells = int(max(0.0, float(self.wall_distance_m)) / resolution)
		min_x = max(0, grid_x - radius_cells)
		max_x = min(self.map_width - 1, grid_x + radius_cells)
		min_y = max(0, grid_y - radius_cells)
		max_y = min(self.map_height - 1, grid_y + radius_cells)

		for iy in range(min_y, max_y + 1):
			row_offset = iy * self.map_width
			for ix in range(min_x, max_x + 1):
				value = int(self.map_data[row_offset + ix])
				if value >= 50:
					return True

		return False

	def _is_inside_map(self, x, y):
		# Skip filtering when map bounds are not ready.
		if None in (self.map_min_x, self.map_max_x, self.map_min_y, self.map_max_y):
			return True

		padding = max(0.0, float(self.map_bounds_padding))
		return (
			(self.map_min_x - padding) <= x <= (self.map_max_x + padding)
			and (self.map_min_y - padding) <= y <= (self.map_max_y + padding)
		)

	def camera_info_callback(self, data):
		if len(data.k) < 9:
			print("Camera info ignored: intrinsics matrix too small.")
			return

		self.fx = float(data.k[0])
		self.fy = float(data.k[4])
		self.cx = float(data.k[2])
		self.cy = float(data.k[5])
		self.camera_frame_id = data.header.frame_id
		# print(f"Camera intrinsics updated: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}.")

	def depth_callback(self, data):
		try:
			depth_raw = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
		except Exception as exc:
			self.get_logger().warn(f"Failed to parse depth image: {exc}")
			return

		if depth_raw is None or depth_raw.size == 0:
			print("Depth image empty, skipping.")
			self.latest_depth_msg = None
			self.latest_depth_image = None
			self.full_depth_image_m = None
			self.full_depth_header = None
			self.depth_y_start = 0
			self.depth_y_end = 0
			return

		depth_height = depth_raw.shape[0]
		top_crop_px = int(depth_height * self.top_crop_ratio)
		bottom_crop_px = int(depth_height * self.bottom_crop_ratio)
		y_start = top_crop_px
		y_end = max(y_start + 1, depth_height - bottom_crop_px)
		cropped_depth = depth_raw[y_start:y_end, :]
		if cropped_depth.size == 0:
			print("Depth crop empty, skipping.")
			self.latest_depth_msg = None
			self.latest_depth_image = None
			self.full_depth_image_m = None
			self.full_depth_header = None
			self.depth_y_start = 0
			self.depth_y_end = 0
			return

		if data.encoding == "16UC1":
			self.full_depth_image_m = depth_raw.astype(np.float32) / 1000.0
		else:
			self.full_depth_image_m = depth_raw.astype(np.float32)
		self.full_depth_header = data.header

		self.latest_depth_msg = data
		self.latest_depth_image = cropped_depth
		self.depth_y_start = y_start
		self.depth_y_end = y_end

	def estimate_rim_point(self, depth_img, u, v, r):
		h, w = depth_img.shape[:2]

		if r < 3:
			return None

		yy, xx = np.ogrid[:h, :w]
		dist = np.sqrt((xx - u) ** 2 + (yy - v) ** 2)
		inner = 0.72 * r
		outer = 1.15 * r
		rim_mask = (dist >= inner) & (dist <= outer)

		x1 = max(0, int(u - 1.25 * r))
		y1 = max(0, int(v - 1.25 * r))
		x2 = min(w, int(u + 1.25 * r))
		y2 = min(h, int(v + 1.25 * r))

		roi_depth = depth_img[y1:y2, x1:x2]
		roi_mask = rim_mask[y1:y2, x1:x2]

		yy_local, xx_local = np.where(roi_mask)
		depth_values = roi_depth[yy_local, xx_local]

		finite_mask = np.isfinite(depth_values)
		range_mask = (depth_values > 0.15) & (depth_values < 10.0)
		valid_mask = finite_mask & range_mask

		if np.count_nonzero(valid_mask) < 20:
			return None

		valid_depths = depth_values[valid_mask]
		valid_x = xx_local[valid_mask] + x1
		valid_y = yy_local[valid_mask] + y1

		rim_depth = float(np.percentile(valid_depths, 35))
		nearest_idx = int(np.argmin(np.abs(valid_depths - rim_depth)))
		rim_u = int(valid_x[nearest_idx])
		rim_v = int(valid_y[nearest_idx])

		return rim_u, rim_v, rim_depth

	def pixel_to_camera_point(self, u, v, z):
		x = (u - self.cx) * z / self.fx
		y = (v - self.cy) * z / self.fy
		return x, y, z

	def keep_outer_circles(self, circles):
		if circles is None or len(circles) == 0:
			return []

		sorted_circles = sorted(circles, key=lambda c: c[2], reverse=True)
		kept = []

		for candidate in sorted_circles:
			x, y, r = candidate
			matched = False
			for kept_circle in kept:
				kx, ky, kr = kept_circle
				center_distance = np.hypot(float(x - kx), float(y - ky))
				same_center_threshold = max(6.0, 0.35 * float(max(r, kr)))
				if center_distance <= same_center_threshold:
					matched = True
					break

			if not matched:
				kept.append(candidate)

		return kept

	def publish_ring_markers(self, ring_points_map, stamp):
		if not ring_points_map:
			return

		pt = ring_points_map[0]
		marker = Marker()
		marker.header.frame_id = self.target_frame
		marker.header.stamp = stamp
		marker.ns = "detected_rings"
		marker.id = 0
		marker.type = Marker.SPHERE
		marker.action = Marker.ADD
		marker.pose.position.x = float(pt[0])
		marker.pose.position.y = float(pt[1])
		marker.pose.position.z = float(pt[2])
		marker.pose.orientation.w = 1.0
		marker.scale.x = 0.12
		marker.scale.y = 0.12
		marker.scale.z = 0.12
		marker.color.a = 0.9
		marker.color.r = 1.0
		marker.color.g = 0.4
		marker.color.b = 0.0
		marker.lifetime = RosDuration(sec=0, nanosec=600000000)

		self.ring_pub.publish(marker)

	def _depth_to_meters(self, depth_value, encoding):
		if depth_value is None:
			return None

		value = float(depth_value)
		if not np.isfinite(value) or value <= 0.0:
			return None

		if encoding == "16UC1":
			return value / 1000.0

		return value

	def _read_depth_at_pixel(self, image_x, image_y, image_width, image_height):
		if self.latest_depth_msg is None or self.latest_depth_image is None:
			return None
		if self.depth_y_end <= self.depth_y_start:
			return None

		depth_height, depth_width = self.latest_depth_image.shape[:2]
		if image_width <= 1 or image_height <= 1 or depth_width <= 1 or depth_height <= 1:
			return None

		color_top_crop_px = int(image_height * self.top_crop_ratio)
		color_bottom_crop_px = int(image_height * self.bottom_crop_ratio)
		color_y_start = color_top_crop_px
		color_y_end = max(color_y_start + 1, image_height - color_bottom_crop_px)
		if image_y < color_y_start or image_y >= color_y_end:
			return None

		depth_x = int(round((float(image_x) / float(image_width - 1)) * float(depth_width - 1)))
		color_crop_height = max(1, color_y_end - color_y_start)
		relative_y = float(image_y - color_y_start) / float(max(1, color_crop_height - 1))
		depth_y = int(round(relative_y * float(depth_height - 1)))
		depth_x = max(0, min(depth_x, depth_width - 1))
		depth_y = max(0, min(depth_y, depth_height - 1))

		window_radius = 1
		x0 = max(0, depth_x - window_radius)
		x1 = min(depth_width, depth_x + window_radius + 1)
		y0 = max(0, depth_y - window_radius)
		y1 = min(depth_height, depth_y + window_radius + 1)

		patch = self.latest_depth_image[y0:y1, x0:x1].astype(np.float32)
		encoding = self.latest_depth_msg.encoding
		if encoding == "16UC1":
			patch = patch / 1000.0

		valid = np.isfinite(patch) & (patch > 0.0)
		if not np.any(valid):
			return None
		depth_m = float(np.mean(patch[valid]))
		depth_y_full = self.depth_y_start + depth_y
		return float(depth_x), float(depth_y_full), float(depth_m)

	def _project_pixel_to_camera(self, image_x, image_y, image_width, image_height):
		if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
			return None

		depth_sample = self._read_depth_at_pixel(image_x, image_y, image_width, image_height)
		if depth_sample is None:
			return None

		depth_x, depth_y, depth_m = depth_sample
		x_cam = ((depth_x - self.cx) * depth_m) / self.fx
		y_cam = ((depth_y - self.cy) * depth_m) / self.fy
		z_cam = depth_m
		return float(x_cam), float(y_cam), float(z_cam)

	def rgb_callback(self, data):
		if self.processing:
			return

		self.processing = True
		transform_stamp = data.header.stamp

		marker = Marker()

		# clear_marker = Marker()
		# clear_marker.action = Marker.DELETEALL
		# marker_array.markers.append(clear_marker)

		if self.latest_depth_msg is None or self.latest_depth_image is None:
			print("No depth data available for RGB frame, skipping.")
			self.processing = False
			return

		cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		height, width = cv_image.shape[:2]
		try:

			depth_msg = self.latest_depth_msg
			camera_frame = self.camera_frame_id if self.camera_frame_id else data.header.frame_id
			if depth_msg is not None and depth_msg.header.frame_id:
				camera_frame = depth_msg.header.frame_id			
			transform = None
			try:
				transform = self.tf_buffer.lookup_transform(
					self.target_frame,
					camera_frame,
					Time.from_msg(transform_stamp),
					timeout=rclpyDuration(seconds=0.05)
				)
			except TransformException as fallback_ex:
				self.get_logger().warn(
					f"Could not lookup transform {camera_frame} -> {self.target_frame}: {fallback_ex}"
				)

			if self.mode == "face":
				top_crop_px = int(height * self.top_crop_ratio)
				bottom_crop_px = int(height * self.bottom_crop_ratio)
				y_start = top_crop_px
				y_end = max(y_start + 1, height - bottom_crop_px)
				cropped_image = cv_image[y_start:y_end, :]
				if cropped_image.size == 0:
					print("RGB crop empty, skipping.")
					return
				print("Running YOLO face detection...")
				res = self.model.predict(cropped_image, imgsz=256, verbose=False, device=self.device)
				if not res:
					print("No faces detected by YOLO.")
				if res and res[0].boxes is not None and len(res[0].boxes) > 0:
					boxes = res[0].boxes
					if boxes.conf is not None and boxes.conf.numel() > 0:
						best_idx = int(boxes.conf.argmax().item())
					else:
						best_idx = 0
					bbox = boxes.xyxy[best_idx].cpu().tolist()
					x1, y1, x2, y2 = [int(v) for v in bbox]
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

					if transform is not None:
						point_cam = self._project_pixel_to_camera(cx, cy, width, height)
						if point_cam is not None:
							face_point_cam = PointStamped()
							face_point_cam.header.frame_id = camera_frame
							face_point_cam.header.stamp = data.header.stamp
							face_point_cam.point.x = point_cam[0]
							face_point_cam.point.y = point_cam[1]
							face_point_cam.point.z = point_cam[2]

							face_point_map = do_transform_point(face_point_cam, transform)

							if not self._is_inside_map(face_point_map.point.x, face_point_map.point.y):
								if self.map_bounds_hard:
									self.get_logger().debug(
										"Face outside map bounds, skipping marker."
									)
									print("Face outside map bounds (hard), skipping.")
									return

							if not self._is_near_wall(face_point_map.point.x, face_point_map.point.y):
								print("Face not near wall, skipping.")
								return

							marker = Marker()
							marker.header.frame_id = self.target_frame
							marker.header.stamp = data.header.stamp
							marker.ns = "detected_faces"
							marker.id = 0
							marker.type = Marker.SPHERE
							marker.action = Marker.ADD
							marker.pose.position.x = float(face_point_map.point.x)
							marker.pose.position.y = float(face_point_map.point.y)
							marker.pose.position.z = float(face_point_map.point.z)
							marker.pose.orientation.w = 1.0
							marker.scale.x = 0.15
							marker.scale.y = 0.15
							marker.scale.z = 0.15

							marker.color.r = 0.2
							marker.color.g = 1.0
							marker.color.b = 0.2
							marker.color.a = 1.0

							self.marker_pub.publish(marker)
							print(f"Published face marker at map coordinates ({face_point_map.point.x:.2f}, {face_point_map.point.y:.2f}, {face_point_map.point.z:.2f}).")
							if self.show_image:
								cv2.imshow("image", cv_image)
								key = cv2.waitKey(1)
							if key == 27:
								print("exiting")
								exit()
					else:
						print("No transform available, skipping face.")

			elif self.mode == "ring":
				ring_image = cv_image.copy()
				ring_points_map = []
				if (
					self.full_depth_image_m is not None
					and self.full_depth_header is not None
					and None not in (self.fx, self.fy, self.cx, self.cy)
					and transform is not None
				):
					image_height, image_width = ring_image.shape[:2]
					upper_height = max(1, int(image_height * self.upper_ratio))
					ring_mask = np.zeros((image_height, image_width), dtype=np.uint8)
					upper_region = ring_image[:upper_height, :]
					cv_image_gray = cv2.cvtColor(upper_region, cv2.COLOR_BGR2GRAY)
					circles = cv2.HoughCircles(
						cv_image_gray,
						cv2.HOUGH_GRADIENT_ALT,
						dp=1.5,
						minDist=45,
						param1=220,
						param2=0.85,
						minRadius=13,
						maxRadius=90,
					)

					if circles is not None:
						circles = np.round(circles[0, :]).astype(int)
						circles = self.keep_outer_circles(circles)
						for x, y, r in circles:
							outer_r = max(1, int(1.1 * r))
							inner_r = max(1, int(0.85 * r))
							cv2.circle(ring_mask, (x, y), outer_r, 255, -1)
							cv2.circle(ring_mask, (x, y), inner_r, 0, -1)

							if (
								y >= self.full_depth_image_m.shape[0]
								or x >= self.full_depth_image_m.shape[1]
								or y < 0
								or x < 0
							):
								continue

							rim_sample = self.estimate_rim_point(self.full_depth_image_m, x, y, r)
							if rim_sample is None:
								continue

							rim_u, rim_v, z_rim = rim_sample
							px, py, pz = self.pixel_to_camera_point(rim_u, rim_v, z_rim)

							ring_point_cam = PointStamped()
							ring_point_cam.header.frame_id = camera_frame
							ring_point_cam.header.stamp = data.header.stamp
							ring_point_cam.point.x = float(px)
							ring_point_cam.point.y = float(py)
							ring_point_cam.point.z = float(pz)

							try:
								ring_point_map = do_transform_point(ring_point_cam, transform)
							except Exception:
								continue

							if not self._is_inside_map(ring_point_map.point.x, ring_point_map.point.y):
								if self.map_bounds_hard:
									continue
							if not self._is_near_wall(ring_point_map.point.x, ring_point_map.point.y):
								continue

							ring_points_map.append(
								(
									ring_point_map.point.x,
									ring_point_map.point.y,
									ring_point_map.point.z,
								)
							)
							break

						if ring_points_map:
							self.publish_ring_markers(ring_points_map, data.header.stamp)
							display_image = cv2.bitwise_and(ring_image, ring_image, mask=ring_mask)
							predicted_color = predict_color(display_image, ring_mask)
							if predicted_color:
								print(f"Predicted ring color: {predicted_color} published to /ring_color topic.")
								self.color_pub.publish(String(data=predicted_color))
								
							print(f"Predicted ring color: {predicted_color} at map points: {ring_points_map}")
							if self.show_image:
								cv2.imshow("image", display_image)
								key = cv2.waitKey(1)
								if key == 27:
									print("exiting")
									exit()

		except CvBridgeError as e:
			print(e)
			self.marker_pub.publish(marker)
		except Exception as e:
			self.get_logger().warn(f"Face detection callback failed: {e}")
			self.marker_pub.publish(marker)
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
