#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Joy

class XboxStringTeleop:
    def __init__(self):
        # Initialize the node
        rospy.init_node('xbox_string_teleop')
        
        # Publisher to the same topic your robot listens to
        self.pub = rospy.Publisher('/cmd_move', String, queue_size=10)
        
        # Subscriber to the Xbox controller
        self.sub = rospy.Subscriber('/joy', Joy, self.joy_callback, queue_size=1)
        
        # Variable to track the last command to avoid spamming the topic
        self.last_command = 'stop'
        
        # --- CONFIGURATION ---
        # Threshold: How far you have to push the stick to trigger a command (0.0 to 1.0)
        self.TRIGGER_THRESHOLD = 0.5 
        
        # Axis Indices (Standard Xbox 360/One/Series mapping)
        self.AXIS_LEFT_STICK_X = 0  # Left/Right
        self.AXIS_LEFT_STICK_Y = 1  # Up/Down

        rospy.loginfo("Xbox String Teleop Started.")
        rospy.loginfo(f"Publishing to: /cmd_move")
        rospy.loginfo("Control Scheme: Left Stick to Move")

    def joy_callback(self, data):
        """
        Reads the joystick axes and converts them into string commands.
        """
        # Read raw values
        # Usually: +1.0 is Left, -1.0 is Right (Standard ROS Joy)
        lr_val = data.axes[self.AXIS_LEFT_STICK_X]
        
        # Usually: +1.0 is Up, -1.0 is Down
        ud_val = data.axes[self.AXIS_LEFT_STICK_Y]

        current_command = 'stop'

        # --- LOGIC TREE ---
        # We prioritize Forward/Backward over turning to prevent mixed signals
        
        if ud_val > self.TRIGGER_THRESHOLD:
            current_command = 'forward'
        elif ud_val < -self.TRIGGER_THRESHOLD:
            current_command = 'backward'
        elif lr_val > self.TRIGGER_THRESHOLD:
            current_command = 'left'
        elif lr_val < -self.TRIGGER_THRESHOLD:
            current_command = 'right'
        else:
            current_command = 'stop'

        # --- PUBLISHING ---
        # Only publish if the command has changed (state machine)
        if current_command != self.last_command:
            self.pub.publish(current_command)
            rospy.loginfo(f"Published command: {current_command}")
            self.last_command = current_command

if __name__ == '__main__':
    try:
        # Instantiate the class
        node = XboxStringTeleop()
        # Keep the node alive
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        # Safety: Publish stop on exit
        pub = rospy.Publisher('/cmd_move', String, queue_size=10)
        pub.publish("stop")
