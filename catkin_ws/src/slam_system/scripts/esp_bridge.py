#!/usr/bin/env python3

import rospy
import tf
from math import sin, cos, pi

from my_slam_interfaces.msg import Ticks
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3, PoseStamped

class RobotBridge:
    def __init__(self):
        rospy.init_node('esp32_bridge_node', anonymous=True)

        # --- Get Robot Parameters ---
        self.wheel_base = rospy.get_param('~wheel_base', 0.165)
        self.ticks_per_meter = rospy.get_param('~ticks_per_meter', 191)
        self.odom_frame = rospy.get_param('~odom_frame', 'odom')
        self.base_frame = rospy.get_param('~base_frame', 'base_link')

        # --- NEW: Path visualization parameters ---
        self.max_path_poses = rospy.get_param('~max_path_poses', 4000) # Store up to 1000 poses

        # --- Internal State ---
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.vx = 0.0      # Velocity in x
        self.vth = 0.0     # Angular velocity in z
        self.last_left_ticks = None
        self.last_right_ticks = None
        self.last_time = rospy.Time.now()

        # --- ROS Subscribers ---
        rospy.Subscriber("robot_ticks", Ticks, self.ticks_callback)
        # REMOVED: Landmark subscriber is no longer needed in this node.

        # --- ROS Publishers ---
        self.odom_pub = rospy.Publisher("odom", Odometry, queue_size=10)
        # REMOVED: Marker publisher is no longer needed. The EKF node handles this.
        
        # --- NEW: Publisher for the robot's path ---
        self.path_pub = rospy.Publisher('/odom_path', Path, queue_size=10)
        self.odom_path = Path() # Path message object

        # --- TF Broadcaster ---
        self.odom_broadcaster = tf.TransformBroadcaster()

        rospy.loginfo("ESP32 Bridge Node Started.")

    def ticks_callback(self, msg):
        current_time = msg.header.stamp

        if self.last_left_ticks is None:
            self.last_left_ticks = msg.left_ticks
            self.last_right_ticks = msg.right_ticks
            self.last_time = current_time
            return

        dt = (current_time - self.last_time).to_sec()
        if dt <= 0:
            return

        delta_left = msg.left_ticks - self.last_left_ticks
        delta_right = msg.right_ticks - self.last_right_ticks

        dist_left = delta_left / self.ticks_per_meter
        dist_right = delta_right / self.ticks_per_meter

        delta_s = (dist_right + dist_left) / 2.0
        delta_theta = (dist_right - dist_left) / self.wheel_base

        self.vx = delta_s / dt
        self.vth = delta_theta / dt

        self.x += delta_s * cos(self.theta + delta_theta / 2.0)
        self.y += delta_s * sin(self.theta + delta_theta / 2.0)
        self.theta += delta_theta

        self.publish_odometry(current_time)

        self.last_left_ticks = msg.left_ticks
        self.last_right_ticks = msg.right_ticks
        self.last_time = current_time

    def publish_odometry(self, current_time):
        odom_quat = tf.transformations.quaternion_from_euler(0, 0, self.theta)

        self.odom_broadcaster.sendTransform(
            (self.x, self.y, 0.),
            odom_quat,
            current_time,
            self.base_frame,
            self.odom_frame
        )

        odom = Odometry()
        odom.header.stamp = current_time
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose = Pose(Point(self.x, self.y, 0.), Quaternion(*odom_quat))
        odom.twist.twist = Twist(Vector3(self.vx, 0, 0), Vector3(0, 0, self.vth))
        self.odom_pub.publish(odom)

        # --- NEW: Update and publish the path ---
        # Create a PoseStamped message for the current pose
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = current_time
        pose_stamped.header.frame_id = self.odom_frame
        pose_stamped.pose = odom.pose.pose
        
        # Append the new pose to our path
        self.odom_path.poses.append(pose_stamped)
        
        # Trim the path if it becomes too long
        if len(self.odom_path.poses) > self.max_path_poses:
            self.odom_path.poses.pop(0)
            
        # Add a header to the Path message and publish
        self.odom_path.header.stamp = current_time
        self.odom_path.header.frame_id = self.odom_frame
        self.path_pub.publish(self.odom_path)

    # REMOVED: landmarks_callback is no longer needed.

if __name__ == '__main__':
    try:
        bridge = RobotBridge()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

