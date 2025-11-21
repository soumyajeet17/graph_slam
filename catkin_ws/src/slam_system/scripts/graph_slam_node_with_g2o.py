#!/usr/bin/env python3

"""
graph_slam_backend.py

This node is the "backend" for your Graph SLAM system, written in Python.
It runs on your PC (not the ESP32) and does the following:
1. Subscribes to /robot_ticks and /landmarks from the ESP32.
2. Builds a pose graph using the g2opy library.
3. Performs data association to match new landmarks to existing ones.
4. Periodically optimizes the graph to correct for drift.
5. (Future Step) Publishes the corrected path (/map -> /odom transform) and landmark positions.
"""

import rospy
import numpy as np
import g2opy
import tf
from math import sqrt, atan2

# Import ROS Messages
from my_slam_interfaces.msg import Ticks, LandmarkArray
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray

class GraphSlamBackend:
    def __init__(self):
        rospy.loginfo("Initializing Graph SLAM Backend (Python)...")

        # === Robot Physical Parameters ===
        # Get parameters from ROS param server, with defaults
        self.wheel_base_m = rospy.get_param('~wheel_base_m', 0.3)
        self.ticks_per_meter = rospy.get_param('~ticks_per_meter', 1000.0)
        
        # === Data Association Parameters ===
        self.data_association_dist_m = rospy.get_param('~data_association_dist_m', 0.5)
        self.data_association_dist_sq_ = self.data_association_dist_m ** 2

        # === G2O Information (Uncertainty) Matrices ===
        # These define how "confident" we are in our measurements.
        # Higher numbers = more confidence.
        # Odometry (Pose-Pose)
        self.info_matrix_odom = np.identity(3)
        self.info_matrix_odom[0, 0] = rospy.get_param('~odom_info/x', 100.0)
        self.info_matrix_odom[1, 1] = rospy.get_param('~odom_info/y', 100.0)
        self.info_matrix_odom[2, 2] = rospy.get_param('~odom_info/theta', 200.0)
        
        # Landmark (Pose-Landmark)
        self.info_matrix_landmark = np.identity(2)
        self.info_matrix_landmark[0, 0] = rospy.get_param('~lm_info/range', 500.0)
        self.info_matrix_landmark[1, 1] = rospy.get_param('~lm_info/bearing', 500.0)

        # === Initialize G2O Optimizer ===
        self.optimizer = g2opy.SparseOptimizer()
        
        # Use CHOLMOD which is faster, fallback to CSPARSE
        try:
            solver_name = "AUTO" # g2opy will pick the best available
            linear_solver = g2opy.BlockSolverX.LinearSolverType()
            
            # Check for CHOLMOD
            try:
                linear_solver = g2opy.LinearSolverCholmodX()
                solver_name = "CHOLMOD"
            except AttributeError:
                rospy.logwarn("CHOLMOD not available, falling back to CSPARSE.")
                # Check for CSPARSE
                try:
                    linear_solver = g2opy.LinearSolverCSparseX()
                    solver_name = "CSPARSE"
                except AttributeError:
                    rospy.logerr("Fatal Error: Neither CHOLMOD nor CSPARSE are available in g2opy.")
                    rospy.logerr("Please install 'scikit-sparse' (for CHOLMOD) or 'scipy' (for CSPARSE)")
                    return

            rospy.loginfo(f"Using {solver_name} linear solver.")
            solver = g2opy.BlockSolverX(linear_solver)

        except Exception as e:
            rospy.logerr(f"Failed to initialize g2o solver: {e}")
            rospy.logerr("Make sure g2opy is installed correctly.")
            return
            
        algorithm = g2opy.OptimizationAlgorithmLevenberg(solver)
        self.optimizer.set_algorithm(algorithm)
        self.optimizer.set_verbose(False)
        
        self.next_vertex_id = 0

        # === Add the very first pose vertex ===
        first_pose = g2opy.VertexSE2()
        first_pose.set_id(self.next_vertex_id)
        first_pose.set_estimate(g2opy.SE2())  # (0, 0, 0)
        first_pose.set_fixed(True)
        self.optimizer.add_vertex(first_pose)
        
        self.pose_vertices = [first_pose]
        self.next_vertex_id += 1

        # === State Tracking ===
        self.first_ticks_received = False
        self.last_left_ticks = 0
        self.last_right_ticks = 0
        self.current_pose_estimate = g2opy.SE2()  # Odometry-only estimate

        # === Landmark Tracking ===
        self.next_landmark_id = 0  # Our internal ID
        # Map<Our_Landmark_ID, (G2O_Vertex, radius)>
        self.landmark_map = {}

        # === ROS Publishers ===
        self.path_pub = rospy.Publisher("/slam/path", Path, queue_size=10)
        self.map_pub = rospy.Publisher("/slam/map_markers", MarkerArray, queue_size=10)
        self.odom_path_pub = rospy.Publisher("/slam/odom_path", Path, queue_size=10) # Odometry-only
        self.tf_broadcaster = tf.TransformBroadcaster()

        # === ROS Subscribers ===
        self.sub_ticks = rospy.Subscriber("/robot_ticks", Ticks, self.ticks_callback, queue_size=100)
        self.sub_landmarks = rospy.Subscriber("/landmarks", LandmarkArray, self.landmarks_callback, queue_size=50)

        # === Timer for periodic optimization ===
        self.optimize_timer = rospy.Timer(rospy.Duration(5.0), self.optimize_graph)
        # === Timer for publishing results ===
        self.publish_timer = rospy.Timer(rospy.Duration(0.1), self.publish_results)

        # Paths for visualization
        self.slam_path_msg = Path()
        self.slam_path_msg.header.frame_id = "map"
        self.odom_path_msg = Path()
        self.odom_path_msg.header.frame_id = "map"


        rospy.loginfo("Graph SLAM Backend (Python) initialized.")

    def ticks_callback(self, msg):
        if not self.first_ticks_received:
            self.last_left_ticks = msg.left_ticks
            self.last_right_ticks = msg.right_ticks
            self.first_ticks_received = True
            return

        # 1. Calculate odometry from ticks
        delta_left_ticks = msg.left_ticks - self.last_left_ticks
        delta_right_ticks = msg.right_ticks - self.last_right_ticks

        dist_left = float(delta_left_ticks) / self.ticks_per_meter
        dist_right = float(delta_right_ticks) / self.ticks_per_meter

        self.last_left_ticks = msg.left_ticks
        self.last_right_ticks = msg.right_ticks

        d_dist = (dist_left + dist_right) / 2.0
        d_theta = (dist_right - dist_left) / self.wheel_base_m

        # 2. Create odometry-only estimate for data association
        odometry_delta = g2opy.SE2(d_dist, 0, d_theta)
        self.current_pose_estimate = self.current_pose_estimate * odometry_delta

        # 3. Create a new POSE VERTEX (g2opy.VertexSE2)
        new_pose_vertex = g2opy.VertexSE2()
        new_pose_vertex.set_id(self.next_vertex_id)
        new_pose_vertex.set_estimate(self.current_pose_estimate)
        self.optimizer.add_vertex(new_pose_vertex)
        self.pose_vertices.append(new_pose_vertex)
        self.next_vertex_id += 1

        # 4. Create a new ODOMETRY EDGE (g2opy.EdgeSE2)
        odometry_edge = g2opy.EdgeSE2()
        odometry_edge.set_vertex(0, self.pose_vertices[-2])  # From
        odometry_edge.set_vertex(1, new_pose_vertex)         # To
        odometry_edge.set_measurement(odometry_delta)
        odometry_edge.set_information(self.info_matrix_odom)
        self.optimizer.add_edge(odometry_edge)

        # 5. Add to odom_path for visualization
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = msg.header.stamp
        pose_stamped.header.frame_id = "map"
        est = self.current_pose_estimate.to_vector()
        pose_stamped.pose.position.x = est[0]
        pose_stamped.pose.position.y = est[1]
        quat = tf.transformations.quaternion_from_euler(0, 0, est[2])
        pose_stamped.pose.orientation.x = quat[0]
        pose_stamped.pose.orientation.y = quat[1]
        pose_stamped.pose.orientation.z = quat[2]
        pose_stamped.pose.orientation.w = quat[3]
        self.odom_path_msg.poses.append(pose_stamped)
        self.odom_path_pub.publish(self.odom_path_msg)


    def landmarks_callback(self, msg):
        if not self.pose_vertices:
            return

        current_pose_vertex = self.pose_vertices[-1]
        current_pose_tf = current_pose_vertex.estimate() # Use the *optimized* estimate

        for lm in msg.landmarks:
            # The landmark (x, y) is in the robot's "base_scan" frame.
            # We must convert this to (range, bearing) for the g2o edge.
            range_val = sqrt(lm.x**2 + lm.y**2)
            bearing_val = atan2(lm.y, lm.x)
            measurement = np.array([range_val, bearing_val])

            # Also, convert to global map frame for data association
            global_lm_pos = current_pose_tf * np.array([lm.x, lm.y])
            
            # --- DATA ASSOCIATION ---
            # Find the closest known landmark
            closest_landmark_id = -1
            min_dist_sq = self.data_association_dist_sq_

            for lm_id, (lm_vertex, lm_radius) in self.landmark_map.items():
                dist_sq = np.sum((lm_vertex.estimate() - global_lm_pos)**2)
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    closest_landmark_id = lm_id

            landmark_vertex = None

            if closest_landmark_id == -1:
                # --- NEW LANDMARK ---
                new_landmark_id = self.next_landmark_id
                self.next_landmark_id += 1
                
                landmark_vertex = g2opy.VertexPointXY()
                landmark_vertex.set_id(self.next_vertex_id) # G2O ID
                self.next_vertex_id += 1
                
                landmark_vertex.set_estimate(global_lm_pos)
                self.optimizer.add_vertex(landmark_vertex)

                self.landmark_map[new_landmark_id] = (landmark_vertex, lm.radius) # Store with radius
            
            else:
                # --- EXISTING LANDMARK ---
                landmark_vertex, _ = self.landmark_map[closest_landmark_id]
                # Optional: you could average the radius here

            # --- ADD LANDMARK EDGE ---
            landmark_edge = g2opy.EdgeSE2PointXY()
            landmark_edge.set_vertex(0, current_pose_vertex) # From Pose
            landmark_edge.set_vertex(1, landmark_vertex)     # To Landmark
            landmark_edge.set_measurement(measurement)
            landmark_edge.set_information(self.info_matrix_landmark)
            
            try:
                self.optimizer.add_edge(landmark_edge)
            except Exception as e:
                rospy.logwarn(f"Failed to add landmark edge: {e}")


    def optimize_graph(self, event):
        if not self.pose_vertices:
            return

        rospy.loginfo("=== Running Graph Optimization (Python) ===")
        rospy.loginfo(f"Graph has {len(self.optimizer.vertices())} vertices and {len(self.optimizer.edges())} edges.")
        
        self.optimizer.initialize_optimization()
        self.optimizer.optimize(10) # Run 10 iterations
        
        rospy.loginfo("Optimization complete.")

        # IMPORTANT: Update our "current_pose_estimate" with the optimized one
        # This stops the odometry-only estimate from drifting away from the
        # optimized graph, which would break data association.
        self.current_pose_estimate = self.pose_vertices[-1].estimate()

    def publish_results(self, event):
        """
        Publishes the optimized path, map (markers), and TF transform.
        """
        if not self.pose_vertices:
            return

        marker_array = MarkerArray()
        
        # 1. Publish Map Markers
        for lm_id, (lm_vertex, lm_radius) in self.landmark_map.items():
            pos = lm_vertex.estimate()
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = event.current_real
            marker.ns = "slam_map"
            marker.id = lm_id
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = pos[0]
            marker.pose.position.y = pos[1]
            marker.pose.position.z = 0.1
            marker.pose.orientation.w = 1.0
            marker.scale.x = lm_radius * 2
            marker.scale.y = lm_radius * 2
            marker.scale.z = 0.2
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = (0.0, 1.0, 1.0, 0.8)
            marker.lifetime = rospy.Duration(0) # Never expire
            marker_array.markers.append(marker)
        
        self.map_pub.publish(marker_array)

        # 2. Publish Optimized Path
        self.slam_path_msg.poses.clear()
        self.slam_path_msg.header.stamp = event.current_real
        for vertex in self.pose_vertices:
            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = event.current_real # Note: We lose individual timestamps
            pose_stamped.header.frame_id = "map"
            est = vertex.estimate().to_vector()
            pose_stamped.pose.position.x = est[0]
            pose_stamped.pose.position.y = est[1]
            quat = tf.transformations.quaternion_from_euler(0, 0, est[2])
            pose_stamped.pose.orientation.x = quat[0]
            pose_stamped.pose.orientation.y = quat[1]
            pose_stamped.pose.orientation.z = quat[2]
            pose_stamped.pose.orientation.w = quat[3]
            self.slam_path_msg.poses.append(pose_stamped)
        
        self.path_pub.publish(self.slam_path_msg)

        # 3. Publish TF (map -> odom)
        # This is the "correction" transform.
        # We find the *last* optimized pose and compare it to the *last* odom-only pose.
        # This part is a bit simplified; a full implementation would use the tf_listener
        # like in your EKF code, but that requires a separate /odom publisher.
        # For now, let's just publish the robot's pose in the map frame.
        
        last_optimized_pose = self.pose_vertices[-1].estimate().to_vector()
        trans = (last_optimized_pose[0], last_optimized_pose[1], 0)
        rot = tf.transformations.quaternion_from_euler(0, 0, last_optimized_pose[2])
        
        self.tf_broadcaster.sendTransform(
            trans, rot,
            event.current_real,
            "base_link_slam",  # This is the SLAM-corrected pose
            "map"
        )
        
        # Also broadcast the odom-only pose for comparison
        odom_pose = self.current_pose_estimate.to_vector()
        odom_trans = (odom_pose[0], odom_pose[1], 0)
        odom_rot = tf.transformations.quaternion_from_euler(0, 0, odom_pose[2])
        self.tf_broadcaster.sendTransform(
            odom_trans, odom_rot,
            event.current_real,
            "base_link_odom", # This is the odometry-only pose
            "map"
        )


if __name__ == '__main__':
    try:
        rospy.init_node('graph_slam_backend_py', anonymous=True)
        backend = GraphSlamBackend()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

