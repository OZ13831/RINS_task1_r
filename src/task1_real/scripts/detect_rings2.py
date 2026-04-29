#!/usr/bin/python3

import os
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import OccupancyGrid

from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PointStamped, Point
from builtin_interfaces.msg import Duration as RosDuration
from tf2_ros import Buffer, TransformListener, TransformException
from tf2_geometry_msgs import do_transform_point
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from rclpy.duration import Duration
from rclpy.time import Time

qos_profile = QoSProfile(
          durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
          reliability=QoSReliabilityPolicy.RELIABLE,
          history=QoSHistoryPolicy.KEEP_LAST,
          depth=1)




# morphological operation to cleanup mask 
# could be better, but it works for now
def cleanup_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    # kernel = np.array([
    #     [0, 0, 1, 0, 0],
    #     [0, 1, 0, 1, 0],
    #     [1, 0, 0, 0, 1],
    #     [0, 1, 0, 1, 0],
    #     [0, 0, 1, 0, 0]], dtype=np.uint8)
    erosion = cv2.erode(mask,kernel,iterations = 1)
    dilation = cv2.dilate(erosion,kernel,iterations = 1)
    return dilation

def predict_color(ring_img, mask=None):
    if mask is None or np.sum(mask) == 0:
        return ""
    mask = cleanup_mask(mask)


    # Create histograms folder if it doesn't exist
    hist_folder = "/home/gamma/colcon_ws/avg_hist2"
    if not os.path.exists(hist_folder):
        os.makedirs(hist_folder)
    
    # Ensure mask matches image dimensions
    if mask.shape[:2] != ring_img.shape[:2]:
        mask = cv2.resize(mask, (ring_img.shape[1], ring_img.shape[0]))
    
    # print("Ring image pixel:", ring_img[29][107])
    
    hsv = cv2.cvtColor(ring_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Apply mask to get only masked values
    h_masked = h[mask > 0]
    s_masked = s[mask > 0]
    v_masked = v[mask > 0]
    
    print("HSV (masked):", np.mean(h_masked), np.mean(s_masked), np.mean(v_masked))
    if np.mean(v_masked) < 85:
        return "black"
    
    hist = cv2.calcHist([h], [0], mask, [180], [0, 180])
    hist = cv2.normalize(hist, hist).flatten() 


    histograms = os.listdir(hist_folder)
    best_match = None
    second_best = None 

    best_score = float('inf')
    for hist_file in histograms:
        if hist_file.endswith(".npy"):
            hist_path = os.path.join(hist_folder, hist_file)
            hist_data = np.load(hist_path)
            score = np.sqrt(0.5 * np.sum((np.sqrt(hist) - np.sqrt(hist_data))**2))
            color_name = hist_file.split(".")[0]
            # print(f"Comparing to {color_name}, score: {score}")
            if score < best_score:
                second_best = best_score
                best_score = score
                best_match = color_name
    

    iowe_ratio = second_best / best_score if second_best is not None else float(1)
    print(f"Best match: {best_match} with score {best_score}, IOWE ratio: {iowe_ratio}")
    colors = ['red', 'green', 'blue']
    for color in colors:
        if best_match.startswith(color) and iowe_ratio > 1.2:
            return color
    return ""


class RingDetector(Node):
    def __init__(self):
        super().__init__('transform_point')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('upper_ratio', 0.5),
                ('map_topic', '/map'),
                ('map_bounds_padding', 0.25),
                ('map_bounds_hard', False),
                ('wall_distance_m', 0.7),
                ('target_frame', 'map'),
            ],
        )

        self.upper_ratio = self.get_parameter('upper_ratio').get_parameter_value().double_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.map_bounds_padding = self.get_parameter('map_bounds_padding').get_parameter_value().double_value
        self.map_bounds_hard = self.get_parameter('map_bounds_hard').get_parameter_value().bool_value
        self.wall_distance_m = self.get_parameter('wall_distance_m').get_parameter_value().double_value
        self.target_frame = self.get_parameter('target_frame').get_parameter_value().string_value
        # An object we use for converting images between ROS format and OpenCV format
        self.bridge = CvBridge()
        self.circles = None
        self.depth_image = None
        self.depth_header = None

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

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

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        map_qos = QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscribe to the image and/or depth topic
        self.rgb_sync_sub = message_filters.Subscriber(self, Image, "/gemini/color/image_raw", qos_profile=qos_profile_sensor_data)
        self.depth_sync_sub = message_filters.Subscriber(self, Image, "/gemini/depth/image_raw", qos_profile=qos_profile_sensor_data)
        self.rgb_depth_sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sync_sub, self.depth_sync_sub],
            queue_size=10,
            slop=0.1
        )
        self.rgb_depth_sync.registerCallback(self.rgb_depth_callback)
        self.depth_info_sub = self.create_subscription(CameraInfo, "/gemini/depth/camera_info", self.depth_info_callback, qos_profile_sensor_data)
        self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic, self.map_callback, map_qos)
        

        self.color_pub = self.create_publisher(String, "/ring_color", qos_profile_sensor_data)
        self.ring_pub = self.create_publisher(Marker, "/rings", qos_profile)
        # cv2.namedWindow("detected_circles", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("detected_circles", 960, 540)

        #cv2.namedWindow("Binary Image", cv2.WINDOW_NORMAL)
        #cv2.namedWindow("Detected contours", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("Detected rings", cv2.WINDOW_NORMAL)

        self.get_logger().info(f"Waiting for map on {self.map_topic} to enable map bounds checking.")

    def map_callback(self, data):
        # Cache map boundaries for quick inside-map checks.
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

    def _is_inside_map(self, x, y):
        # Skip filtering when map bounds are not ready.
        if None in (self.map_min_x, self.map_max_x, self.map_min_y, self.map_max_y):
            return True

        padding = max(0.0, float(self.map_bounds_padding))
        return (
            (self.map_min_x - padding) <= x <= (self.map_max_x + padding)
            and (self.map_min_y - padding) <= y <= (self.map_max_y + padding)
        )

    def _is_near_wall(self, x, y):
        if (
            self.map_data is None
            or self.map_resolution is None
            or self.map_width is None
            or self.map_height is None
            or self.map_origin_x is None
            or self.map_origin_y is None
        ):
            return False

        resolution = float(self.map_resolution)
        if resolution <= 0.0:
            return False

        grid_x = int((x - self.map_origin_x) / resolution)
        grid_y = int((y - self.map_origin_y) / resolution)
        if grid_x < 0 or grid_y < 0 or grid_x >= self.map_width or grid_y >= self.map_height:
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

    def depth_callback(self, data):
        try:
            if data.encoding == '16UC1':
                depth_raw = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
                self.depth_image = depth_raw.astype(np.float32) / 1000.0
            else:
                depth_raw = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
                self.depth_image = depth_raw.astype(np.float32)

            self.depth_header = data.header
        except CvBridgeError as ex:
            self.get_logger().warn(f"Depth conversion failed: {ex}")

    def rgb_depth_callback(self, rgb_msg, depth_msg):
        self.depth_callback(depth_msg)
        self.image_callback(rgb_msg)

    def depth_info_callback(self, msg: CameraInfo):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        # print(f"Camera intrinsics updated: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}.")

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
        marker = Marker()

        # clear_marker = Marker()
        # clear_marker.action = Marker.DELETEALL
        # marker_array.markers.append(clear_marker)

        if len(ring_points_map) > 0:
            pt = ring_points_map[0]
            marker.header.frame_id = self.target_frame
            marker.header.stamp = stamp
            marker.ns = 'detected_rings'
            marker.id = 0
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = pt[0]
            marker.pose.position.y = pt[1]
            marker.pose.position.z = pt[2]
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.12
            marker.scale.y = 0.12
            marker.scale.z = 0.12
            marker.color.a = 0.9
            marker.color.r = 1.0
            marker.color.g = 0.4
            marker.color.b = 0.0
            marker.lifetime = RosDuration(sec=0, nanosec=600000000)
    

        print(f"Publishing ring markers: {len(ring_points_map)} ring(s) detected.")
        self.ring_pub.publish(marker)

    def image_callback(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        display_image = cv_image.copy()
        image_height = cv_image.shape[0]
        image_width = cv_image.shape[1]
        upper_height = max(1, int(image_height * self.upper_ratio))
        ring_mask = np.zeros((image_height, image_width), dtype=np.uint8)

        # Keep and process only upper ratio
        upper_region = cv_image[:upper_height, :]
        cv_image_gray = cv2.cvtColor(upper_region, cv2.COLOR_BGR2GRAY)
        self.circles = cv2.HoughCircles(
            cv_image_gray,
            cv2.HOUGH_GRADIENT_ALT,
            dp=1.5,
            minDist=45,
            param1=220,
            param2=0.85,
            minRadius=13,
            maxRadius=90,
        )

        ring_points_map = []

        if self.depth_header is not None and self.depth_header.frame_id:
            camera_frame = self.depth_header.frame_id
        else:
            camera_frame = data.header.frame_id

        transform = None
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                camera_frame,
                Time.from_msg(data.header.stamp),
                timeout=Duration(seconds=0.05)
            )
        except TransformException as fallback_ex:
            self.get_logger().debug(
                f"Could not lookup transform {camera_frame} -> {self.target_frame}: {fallback_ex}"
            )

        if self.circles is not None:
            print(f"Rings: HoughCircles found {len(self.circles[0])} circle(s).")

            circles = np.round(self.circles[0, :]).astype(int)
            circles = self.keep_outer_circles(circles)
            for x, y, r in circles:
                x1 = max(0, x - r)
                y1 = max(0, y - r)
                x2 = min(cv_image_gray.shape[1] - 1, x + r)
                y2 = min(cv_image_gray.shape[0] - 1, y + r)
                cv2.rectangle(cv_image_gray, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(cv_image_gray, (x, y), 2, (0, 0, 255), -1)

                outer_r = max(1, int(1.1 * r))
                inner_r = max(1, int(0.85 * r))
                cv2.circle(ring_mask, (x, y), outer_r, 255, -1)
                cv2.circle(ring_mask, (x, y), inner_r, 0, -1)

                if self.depth_image is None or self.depth_header is None:
                    print(f"Ring at ({x},{y}) skipped: no depth image.")
                    continue
                if None in (self.fx, self.fy, self.cx, self.cy):
                    print(f"Ring at ({x},{y}) skipped: no camera intrinsics.")
                    continue
                if y >= self.depth_image.shape[0] or x >= self.depth_image.shape[1] or y < 0 or x < 0:
                    print(f"Ring at ({x},{y}) skipped: depth bounds.")
                    continue
                if transform is None:
                    print("Ring skipped: no transform available.")
                    continue

                rim_sample = self.estimate_rim_point(self.depth_image, x, y, r)
                if rim_sample is None:
                    print(f"Ring at ({x},{y}) skipped: no rim depth.")
                    continue

                rim_u, rim_v, z_rim = rim_sample

                cv2.circle(cv_image_gray, (rim_u, rim_v), 2, (255, 255, 255), -1)
                cv2.circle(display_image, (rim_u, rim_v), 3, (255, 255, 255), -1)

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
                        self.get_logger().debug(
                            "Ring outside map bounds, skipping marker."
                        )
                        print("Ring skipped: outside map bounds (hard).")
                        continue
                    self.get_logger().debug(
                        "Ring outside map bounds, keeping marker due to soft mode."
                    )

                if not self._is_near_wall(ring_point_map.point.x, ring_point_map.point.y):
                    self.get_logger().debug(
                        "Ring not near wall, skipping marker."
                    )
                    print("Ring skipped: not near wall.")
                    continue

                ring_points_map.append((
                    ring_point_map.point.x,
                    ring_point_map.point.y,
                    ring_point_map.point.z,
                ))

            self.publish_ring_markers(ring_points_map, data.header.stamp)
            display_image = cv2.bitwise_and(display_image, display_image, mask=ring_mask)
            predicted_color = predict_color(display_image, ring_mask)
            if predicted_color:
                self.color_pub.publish(String(data=predicted_color))


            cv2.imshow("detected_circles", display_image)
            key = cv2.waitKey(1)
            if key == 27:
                rclpy.shutdown()

def main():

    rclpy.init(args=None)
    rd_node = RingDetector()

    rclpy.spin(rd_node)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()