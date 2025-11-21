#!/usr/bin/env python3

import rospy
import numpy as np
import tf
from math import sin, cos, atan2, sqrt, pi

# --- GTSAM IMPORTS ---
import gtsam
from gtsam.symbol_shorthand import X, L # X for poses, L for landmarks

from nav_msgs.msg import Odometry
from my_slam_interfaces.msg import LandmarkArray
from visualization_msgs.msg import Marker, MarkerArray

def normalize_angle(angle):
    """ Normalize an angle to the range [-pi, pi] """
    while angle > np.pi:
        angle -= 2.0 * np.pi
    while angle < -np.pi:
        angle += 2.0 * np.pi
    return angle

class GraphSLAM:
    def __init__(self):
        rospy.init_node('graph_slam_node')

        '''
        # --- GTSAM Setup ---
        parameters = gtsam.ISAM2Params()
        parameters.relinearizeThreshold = 0.1
        parameters.relinearizeSkip = 1
        self.isam = gtsam.ISAM2(parameters)
	'''
	# --- GTSAM Setup ---
        # ISAM2 is the incremental optimizer
        parameters = gtsam.ISAM2Params()
        
        # COMMENT OUT THESE LINES FOR NOW TO USE DEFAULTS
        # parameters.setRelinearizeThreshold(0.1)
        # parameters.setRelinearizeSkip(1)
        
        # OPTIONAL: Print available attributes to debug the correct names for later
        # rospy.loginfo(f"ISAM2 Parameters API: {dir(parameters)}")

        self.isam = gtsam.ISAM2(parameters)

        # Containers for new factors and estimates to be added in the next update step
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimates = gtsam.Values()
        self.current_estimate = gtsam.Values()

        # --- Noise Models (The GraphSLAM equivalent of R and Q) ---
        # Odometry Noise: [x, y, theta]
        self.odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.05, 0.05]))
        # Measurement Noise: [bearing, range] 
        # Note: GTSAM usually does [bearing, range], your EKF did [range, bearing]. Watch order.
        self.measurement_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1])) 
        # Prior Noise (Locking the start position)
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, 0.01]))

        # --- Initialization ---
        self.pose_index = 0
        
        # Add Prior Factor to lock X(0) at (0,0,0)
        self.graph.add(gtsam.PriorFactorPose2(X(0), gtsam.Pose2(0, 0, 0), prior_noise))
        self.initial_estimates.insert(X(0), gtsam.Pose2(0, 0, 0))
        self.current_estimate.insert(X(0), gtsam.Pose2(0, 0, 0))

        # --- Landmark Management ---
        # GTSAM Points don't store 'radius', so we keep a separate dict for that metadata
        # Mapping: landmark_id (int) -> {'radius': float, 'count': int}
        self.landmark_registry = {} 
        self.next_landmark_id = 0
        self.MAX_LANDMARKS = 6
        
        # Tuning parameters
        self.RADIUS_TOLERANCE = 0.05
        self.GATE_THRESHOLD = 1.0 # Euclidean distance threshold for data association

        # --- ROS Communication ---
        self.last_odom_time = None
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/landmarks', LandmarkArray, self.landmark_callback)
        self.map_pub = rospy.Publisher('/slam/map_markers', MarkerArray, queue_size=10)
        
        # --- TF ---
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
        
        # 1. Extract Control Input (Odometry)
        v = msg.twist.twist.linear.x
        omega = msg.twist.twist.angular.z
        
        # Calculate displacement in robot frame
        # (Approximation for short dt: arc motion ~ straight line)
        dx = v * dt 
        dy = 0.0 
        dtheta = omega * dt

        # 2. Update Graph Logic
        self.pose_index += 1
        
        # Create the Odometry Factor (Constraints between X_prev and X_curr)
        # In GTSAM, Pose2(dx, dy, dtheta) represents the relative transform
        odom_transform = gtsam.Pose2(dx, dy, dtheta)
        self.graph.add(gtsam.BetweenFactorPose2(X(self.pose_index - 1), X(self.pose_index), odom_transform, self.odom_noise))
        
        # 3. Predict Initial Estimate for the new pose
        # Get the optimized estimate of the previous pose
        if self.current_estimate.exists(X(self.pose_index - 1)):
            prev_pose = self.current_estimate.atPose2(X(self.pose_index - 1))
        else:
            # Fallback if optimizer hasn't run yet
            prev_pose = self.initial_estimates.atPose2(X(self.pose_index - 1))

        # Compose predicted pose
        predicted_pose = prev_pose.compose(odom_transform)
        self.initial_estimates.insert(X(self.pose_index), predicted_pose)

    def landmark_callback(self, msg):
        # Only process landmarks if we have a valid pose estimate to attach them to
        if not self.initial_estimates.exists(X(self.pose_index)) and not self.current_estimate.exists(X(self.pose_index)):
            return

        # Get current robot pose (best guess)
        if self.current_estimate.exists(X(self.pose_index)):
            robot_pose = self.current_estimate.atPose2(X(self.pose_index))
        else:
            robot_pose = self.initial_estimates.atPose2(X(self.pose_index))

        for landmark_obs in msg.landmarks:
            # Input data
            # Note: GTSAM BearingRange factor expects (bearing, range)
            obs_r = sqrt(landmark_obs.x**2 + landmark_obs.y**2)
            obs_bearing = atan2(landmark_obs.y, landmark_obs.x)
            obs_radius = landmark_obs.radius

            # --- Data Association ---
            best_match_id, min_dist = self.data_association(robot_pose, obs_r, obs_bearing, obs_radius)

            if best_match_id != -1:
                # --- KNOWN LANDMARK: Add Constraint ---
                self.graph.add(gtsam.BearingRangeFactor2D(
                    X(self.pose_index), 
                    L(best_match_id), 
                    gtsam.Rot2(obs_bearing), 
                    obs_r, 
                    self.measurement_noise
                ))
                # Update radius average (optional simple moving average)
                # self.landmark_registry[best_match_id]['radius'] = (self.landmark_registry[best_match_id]['radius'] + obs_radius) / 2.0

            elif len(self.landmark_registry) < self.MAX_LANDMARKS:
                # --- NEW LANDMARK: Initialize ---
                new_id = self.next_landmark_id
                self.next_landmark_id += 1
                
                # Calculate global position for initialization
                # TransformFrom converts point from Body frame to Global frame
                # Point in body frame: (x, y) = (r*cos(b), r*sin(b))
                point_in_body = gtsam.Point2(landmark_obs.x, landmark_obs.y)
                point_global = robot_pose.transformFrom(point_in_body)
                
                # Add Initial Estimate
                self.initial_estimates.insert(L(new_id), point_global)
                
                # Add Factor
                self.graph.add(gtsam.BearingRangeFactor2D(
                    X(self.pose_index), 
                    L(new_id), 
                    gtsam.Rot2(obs_bearing), 
                    obs_r, 
                    self.measurement_noise
                ))
                
                # Register metadata
                self.landmark_registry[new_id] = {'radius': obs_radius}
                rospy.loginfo(f"Initialized Landmark L{new_id}")

        # --- OPTIMIZATION STEP ---
        # This is the "magic" replacement for the EKF Update equations.
        # We push the new factors (graph) and new estimates (initial_estimates) into ISAM2.
        self.isam.update(self.graph, self.initial_estimates)
        
        # Update our current best estimate with the result
        self.current_estimate = self.isam.calculateEstimate()
        
        # Clear the containers (ISAM2 stores the history internally, we only feed it new stuff)
        self.graph.resize(0)
        self.initial_estimates.clear()

        self.publish_map_markers()

    def data_association(self, robot_pose, r, bearing, obs_radius):
        """ 
        Associates observation with existing landmarks in the Graph.
        Uses Euclidean distance in measurement space + Radius check.
        """
        best_id = -1
        min_dist = self.GATE_THRESHOLD

        for lm_id in self.landmark_registry.keys():
            # Check if this landmark exists in our current estimate
            if self.current_estimate.exists(L(lm_id)):
                lm_pos = self.current_estimate.atPoint2(L(lm_id))
            else:
                continue

            # 1. Predict measurement (What should I see?)
            # TransformTo converts Global Point to Body Frame Point
            local_point = robot_pose.transformTo(lm_pos)
            pred_r = np.linalg.norm(local_point)
            pred_bearing = atan2(local_point[1], local_point[0])

            # 2. Calculate Errors
            r_err = r - pred_r
            bearing_err = normalize_angle(bearing - pred_bearing)
            
            # Simple Euclidean distance in (r, theta) space
            # You could weigh this by covariance if you wanted full Mahalanobis parity
            dist_metric = sqrt(r_err**2 + (bearing_err)**2)

            # 3. Check Radius
            map_radius = self.landmark_registry[lm_id]['radius']
            radius_diff = abs(obs_radius - map_radius)

            # 4. Gate
            if dist_metric < min_dist and radius_diff < self.RADIUS_TOLERANCE:
                min_dist = dist_metric
                best_id = lm_id

        return best_id, min_dist

    def publish_tf(self, event):
        # Check if we have a valid estimate for the current pose
        if not self.current_estimate.exists(X(self.pose_index)):
            return

        current_pose = self.current_estimate.atPose2(X(self.pose_index))
        x = current_pose.x()
        y = current_pose.y()
        theta = current_pose.theta()

        # Calculate Map -> Odom transform
        # (Exactly as you did in EKF: T_map_odom = T_map_base * inv(T_odom_base))
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
        
        # Iterate over all registered landmarks
        for lm_id, props in self.landmark_registry.items():
            if self.current_estimate.exists(L(lm_id)):
                pos = self.current_estimate.atPoint2(L(lm_id))
                
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = rospy.Time.now()
                marker.ns = "graph_slam"
                marker.id = lm_id
                marker.type = Marker.CYLINDER
                marker.action = Marker.ADD
                marker.pose.position.x = pos[0]
                marker.pose.position.y = pos[1]
                marker.pose.position.z = 0.1
                marker.pose.orientation.w = 1.0
                marker.scale.x = props['radius'] * 2
                marker.scale.y = props['radius'] * 2
                marker.scale.z = 0.2
                marker.color.r, marker.color.g, marker.color.b, marker.color.a = (1.0, 0.0, 1.0, 0.8) # Purple for GraphSLAM
                marker.lifetime = rospy.Duration(0)
                marker_array.markers.append(marker)
                
        self.map_pub.publish(marker_array)

if __name__ == '__main__':
    try:
        slam = GraphSLAM()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
