#!/usr/bin/env python3

import rospy
import numpy as np
import tf
from math import sin, cos, atan2, sqrt, pi

import gtsam
from gtsam.symbol_shorthand import X, L

from nav_msgs.msg import Odometry
from my_slam_interfaces.msg import LandmarkArray
from visualization_msgs.msg import Marker, MarkerArray

def normalize_angle(angle):
    while angle > np.pi:
        angle -= 2.0 * np.pi
    while angle < -np.pi:
        angle += 2.0 * np.pi
    return angle

class GraphSLAM:
    def __init__(self):
        rospy.init_node('graph_slam_node')

        # --- GTSAM Setup ---
        parameters = gtsam.ISAM2Params()
        # parameters.relinearizeThreshold = 0.1 # Commented out to use defaults
        # parameters.relinearizeSkip = 1        # Commented out to use defaults
        self.isam = gtsam.ISAM2(parameters)

        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimates = gtsam.Values()
        self.current_estimate = gtsam.Values()

        # --- Noise Models ---
        self.odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.05, 0.05]))
        self.measurement_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1])) 
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, 0.01]))

        # --- Initialization ---
        self.pose_index = 0
        
        # Initialize Prior
        start_pose = gtsam.Pose2(0, 0, 0)
        self.graph.add(gtsam.PriorFactorPose2(X(0), start_pose, prior_noise))
        self.initial_estimates.insert(X(0), start_pose)
        self.current_estimate.insert(X(0), start_pose)
        
        # --- FIX: Persistent State Variable ---
        # We keep a copy of the latest pose here so we never lose it
        # regardless of what the optimizer clears.
        self.last_pose_estimate = start_pose

        # --- Landmark Management ---
        self.landmark_registry = {} 
        self.next_landmark_id = 0
        self.MAX_LANDMARKS = 6
        self.RADIUS_TOLERANCE = 0.05
        self.GATE_THRESHOLD = 1.0 

        # --- ROS Communication ---
        self.last_odom_time = None
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/landmarks', LandmarkArray, self.landmark_callback)
        self.map_pub = rospy.Publisher('/slam/map_markers', MarkerArray, queue_size=10)
        
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.tf_listener = tf.TransformListener()

        rospy.Timer(rospy.Duration(0.05), self.publish_tf)
        rospy.loginfo("GraphSLAM Node (GTSAM/ISAM2) Started.")

    def odom_callback(self, msg):
        if self.last_odom_time is None:
            self.last_odom_time = msg.header.stamp
            return
        dt = (msg.header.stamp - self.last_odom_time).to_sec()
        if dt <= 0: return
        self.last_odom_time = msg.header.stamp
        
        v = msg.twist.twist.linear.x
        omega = msg.twist.twist.angular.z
        
        dx = v * dt 
        dy = 0.0 
        dtheta = omega * dt

        # 1. Update Graph Logic
        self.pose_index += 1
        
        odom_transform = gtsam.Pose2(dx, dy, dtheta)
        self.graph.add(gtsam.BetweenFactorPose2(X(self.pose_index - 1), X(self.pose_index), odom_transform, self.odom_noise))
        
        # 2. Predict Initial Estimate (The FIX)
        # Instead of querying GTSAM containers (which might be empty/cleared),
        # we use our persistent python variable.
        prev_pose = self.last_pose_estimate
        predicted_pose = prev_pose.compose(odom_transform)
        
        self.initial_estimates.insert(X(self.pose_index), predicted_pose)
        
        # Update our persistent tracker
        self.last_pose_estimate = predicted_pose

    def landmark_callback(self, msg):
        # Use the persistent estimate for calculations
        robot_pose = self.last_pose_estimate

        for landmark_obs in msg.landmarks:
            obs_r = sqrt(landmark_obs.x**2 + landmark_obs.y**2)
            obs_bearing = atan2(landmark_obs.y, landmark_obs.x)
            obs_radius = landmark_obs.radius

            best_match_id, min_dist = self.data_association(robot_pose, obs_r, obs_bearing, obs_radius)

            if best_match_id != -1:
                self.graph.add(gtsam.BearingRangeFactor2D(
                    X(self.pose_index), 
                    L(best_match_id), 
                    gtsam.Rot2(obs_bearing), 
                    obs_r, 
                    self.measurement_noise
                ))
            elif len(self.landmark_registry) < self.MAX_LANDMARKS:
                new_id = self.next_landmark_id
                self.next_landmark_id += 1
                
                point_in_body = gtsam.Point2(landmark_obs.x, landmark_obs.y)
                point_global = robot_pose.transformFrom(point_in_body)
                
                self.initial_estimates.insert(L(new_id), point_global)
                
                self.graph.add(gtsam.BearingRangeFactor2D(
                    X(self.pose_index), 
                    L(new_id), 
                    gtsam.Rot2(obs_bearing), 
                    obs_r, 
                    self.measurement_noise
                ))
                
                self.landmark_registry[new_id] = {'radius': obs_radius}
                rospy.loginfo(f"Initialized Landmark L{new_id}")

        # --- OPTIMIZATION STEP ---
        # Only optimize if we actually have new data in the graph/estimates
        if self.graph.size() > 0 or self.initial_estimates.size() > 0:
            try:
                self.isam.update(self.graph, self.initial_estimates)
                self.current_estimate = self.isam.calculateEstimate()
                
                # Update our persistent tracker with the OPTIMIZED position
                # This "closes the loop" so the next odometry step starts from the corrected spot
                if self.current_estimate.exists(X(self.pose_index)):
                    self.last_pose_estimate = self.current_estimate.atPose2(X(self.pose_index))
            except RuntimeError as e:
                rospy.logwarn(f"GTSAM Update failed: {e}")

            # IMPORTANT: Clear buffers OUTSIDE the try/except block
            # This ensures we don't get stuck in a loop if the update fails
            self.graph.resize(0)
            self.initial_estimates.clear()

        self.publish_map_markers()

    def data_association(self, robot_pose, r, bearing, obs_radius):
        best_id = -1
        min_dist = self.GATE_THRESHOLD

        for lm_id in self.landmark_registry.keys():
            # Check if this landmark exists in our current estimate
            if self.current_estimate.exists(L(lm_id)):
                lm_pos = self.current_estimate.atPoint2(L(lm_id))
            elif self.initial_estimates.exists(L(lm_id)):
                 lm_pos = self.initial_estimates.atPoint2(L(lm_id))
            else:
                continue

            local_point = robot_pose.transformTo(lm_pos)
            pred_r = np.linalg.norm(local_point)
            pred_bearing = atan2(local_point[1], local_point[0])

            r_err = r - pred_r
            bearing_err = normalize_angle(bearing - pred_bearing)
            
            dist_metric = sqrt(r_err**2 + (bearing_err)**2)
            map_radius = self.landmark_registry[lm_id]['radius']
            radius_diff = abs(obs_radius - map_radius)

            if dist_metric < min_dist and radius_diff < self.RADIUS_TOLERANCE:
                min_dist = dist_metric
                best_id = lm_id

        return best_id, min_dist

    def publish_tf(self, event):
        # Use our persistent estimate for smooth TF
        # (Or use current_estimate if you prefer jumps when loops close)
        current_pose = self.last_pose_estimate 
        
        x = current_pose.x()
        y = current_pose.y()
        theta = current_pose.theta()

        T_map_robot_mat = tf.transformations.compose_matrix(
            translate=(x, y, 0),
            angles=(0, 0, theta)
        )
        try:
            (odom_trans, odom_rot) = self.tf_listener.lookupTransform('/odom', '/base_link', rospy.Time(0))
            T_odom_robot_mat = tf.transformations.compose_matrix(
                translate=odom_trans,
                angles=tf.transformations.euler_from_quaternion(odom_rot)
            )
            T_map_odom_mat = np.dot(T_map_robot_mat, np.linalg.inv(T_odom_robot_mat))
            map_odom_trans = tf.transformations.translation_from_matrix(T_map_odom_mat)
            map_odom_quat = tf.transformations.quaternion_from_matrix(T_map_odom_mat)
            self.tf_broadcaster.sendTransform(map_odom_trans, map_odom_quat, rospy.Time.now(), "odom", "map")
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            pass

    def publish_map_markers(self):
        marker_array = MarkerArray()
        
        for lm_id, props in self.landmark_registry.items():
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "graph_slam"
            marker.id = lm_id
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.scale.x = props['radius'] * 2
            marker.scale.y = props['radius'] * 2
            marker.scale.z = 0.2
            marker.lifetime = rospy.Duration(0)

            # CHECK 1: Is it in the Optimized Estimate? (The Goal)
            if self.current_estimate.exists(L(lm_id)):
                pos = self.current_estimate.atPoint2(L(lm_id))
                marker.pose.position.x = pos[0]
                marker.pose.position.y = pos[1]
                marker.color.r, marker.color.g, marker.color.b, marker.color.a = (1.0, 0.0, 1.0, 0.8) # PURPLE = SUCCESS
            
            # CHECK 2: Is it waiting in the Initial buffer? (Pending)
            elif self.initial_estimates.exists(L(lm_id)):
                pos = self.initial_estimates.atPoint2(L(lm_id))
                marker.pose.position.x = pos[0]
                marker.pose.position.y = pos[1]
                marker.color.r, marker.color.g, marker.color.b, marker.color.a = (1.0, 1.0, 0.0, 0.8) # YELLOW = PENDING
                
            # CHECK 3: Is it nowhere? (Solver Rejected/Crashed)
            else:
                # We use the robot's current position just to show it exists somewhere
                # This tells you "I saw it, but the math failed"
                current_pose = self.last_pose_estimate
                marker.pose.position.x = current_pose.x()
                marker.pose.position.y = current_pose.y()
                marker.color.r, marker.color.g, marker.color.b, marker.color.a = (1.0, 0.0, 0.0, 0.5) # RED = ERROR
            
            marker.pose.position.z = 0.1
            marker.pose.orientation.w = 1.0
            marker_array.markers.append(marker)
                
        self.map_pub.publish(marker_array)

if __name__ == '__main__':
    try:
        slam = GraphSLAM()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
