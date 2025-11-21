#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3, PoseStamped
import tf
from math import sin, cos, pi

# This is your custom message type
from my_slam_interfaces.msg import Ticks 

class OdometryCalculator:
    def __init__(self):
        rospy.init_node('odom_calculator_node')

        # --- Your Robot's Parameters ---
        self.wheel_radius = 0.025  # meters (2.5 cm)
        self.wheel_base = 0.20     # meters (distance between wheels)
        self.ticks_per_revolution = 191.0 # Average ticks for one wheel rotation
        
        self.dist_per_tick = (2 * pi * self.wheel_radius) / self.ticks_per_revolution

        # --- ROS Publishers and Subscribers ---
        self.odom_pub = rospy.Publisher('/odom11', Odometry, queue_size=10)
        self.path_pub = rospy.Publisher('/path', Path, queue_size=10) # <-- NEW PUBLISHER
        self.odom_broadcaster = tf.TransformBroadcaster()
        rospy.Subscriber('/robot_ticks', Ticks, self.ticks_callback)

        # --- State and Path Variables ---
        self.last_time = None
        self.last_left_ticks = 0
        self.last_right_ticks = 0
        
        self.x = 0.0
        self.y = 0.0
        self.th = 0.0
        
        # Initialize the path message
        self.path = Path()
        self.path.header.frame_id = "odom"

        rospy.loginfo("Odometry Calculator Node Started.")

    def ticks_callback(self, msg):
        current_time = msg.header.stamp
        if self.last_time is None:
            self.last_time = current_time
            self.last_left_ticks = msg.left_ticks
            self.last_right_ticks = msg.right_ticks
            return

        dt = (current_time - self.last_time).to_sec()
        
        delta_left = msg.left_ticks - self.last_left_ticks
        delta_right = msg.right_ticks - self.last_right_ticks

        self.last_time = current_time
        self.last_left_ticks = msg.left_ticks
        self.last_right_ticks = msg.right_ticks

        dist_left = delta_left * self.dist_per_tick
        dist_right = delta_right * self.dist_per_tick

        dist_center = (dist_left + dist_right) / 2.0
        delta_th = (dist_right - dist_left) / self.wheel_base
        
        if dt > 0:
            vx = dist_center / dt
            vth = delta_th / dt
        else:
            vx = 0.0
            vth = 0.0

        self.x += dist_center * cos(self.th)
        self.y += dist_center * sin(self.th)
        self.th += delta_th

        odom_quat = tf.transformations.quaternion_from_euler(0, 0, self.th)

        # Publish the TF transform
        self.odom_broadcaster.sendTransform(
            (self.x, self.y, 0.), odom_quat, current_time, "base_link", "odom"
        )

        # Publish the Odometry message
        odom = Odometry()
        odom.header.stamp = current_time
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose = Pose(Point(self.x, self.y, 0.), Quaternion(*odom_quat))
        odom.twist.twist = Twist(Vector3(vx, 0, 0), Vector3(0, 0, vth))
        self.odom_pub.publish(odom)

        # --- NEW: Update and Publish the Path ---
        # 1. Create a PoseStamped message for the current pose
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = current_time
        pose_stamped.header.frame_id = "odom"
        pose_stamped.pose = odom.pose.pose
        
        # 2. Append the new pose to our path
        self.path.poses.append(pose_stamped)
        
        # 3. Publish the updated path
        self.path.header.stamp = current_time
        self.path_pub.publish(self.path)


if __name__ == '__main__':
    try:
        odometry_calculator = OdometryCalculator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
