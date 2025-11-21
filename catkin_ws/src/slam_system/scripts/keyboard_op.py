#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import sys, os
import tty, termios
import select # Used for the non-blocking key read

# Instructions for the user
msg = """
Control Your Robot! (Hold-to-move)
---------------------------
Moving around:
        w
   a    s    d

(Release key to stop)

CTRL-C to quit
"""

# Dictionary to map keys to command strings
key_mappings = {
    'w': 'forward',
    's': 'backward',
    'a': 'left',
    'd': 'right'
}

def is_data():
    """Checks if there is data to be read on stdin"""
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

if __name__ == '__main__':
    # Get the original terminal settings
    settings = termios.tcgetattr(sys.stdin)
    
    rospy.init_node('keyboard_teleop')
    pub = rospy.Publisher('/cmd_move', String, queue_size=10)

    print(msg)
    
    # Variable to track the last published command to avoid spamming "stop"
    last_command = ''

    try:
        # Set the terminal to raw mode
        tty.setraw(sys.stdin.fileno())

        while not rospy.is_shutdown():
            # Wait for a key press for 0.1 seconds
            if is_data():
                key = sys.stdin.read(1)
                
                # If the key is CTRL-C, quit
                if key == '\x03':
                    break
                
                # Check if the key is a valid movement key
                if key in key_mappings:
                    command = key_mappings[key]
                    # Only publish if the command is new
                    if command != last_command:
                        pub.publish(command)
                        last_command = command
                        rospy.loginfo(f"Published command: {command}")
            else:
                # No key was pressed within the timeout, so stop the robot
                # Only publish if the robot wasn't already stopped
                if last_command != 'stop':
                    pub.publish('stop')
                    last_command = 'stop'
                    rospy.loginfo("Timeout: Published stop command.")

            rospy.sleep(0.05) # Loop at 20Hz

    except Exception as e:
        print(e)

    finally:
        # Before exiting, publish a final "stop" command and restore terminal settings
        pub.publish("stop")
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
