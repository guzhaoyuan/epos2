#!/usr/bin/env python

from epos2.srv import *
import rospy
import sys

def print_position_n_torque(x, y):
    print("wait for Service")
    rospy.wait_for_service('print_request')
    try:
        print("now request service")
        print_request = rospy.ServiceProxy('print_request', Torque)
        print("now request service")
        resp1 = print_request(x, y)
        print(resp1)
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

def usage():
    return "%s [position torque]"%sys.argv[0]

if __name__ == "__main__":
    if len(sys.argv) == 3:
        position = int(sys.argv[1])
        torque = int(sys.argv[2])
        print(position,torque)
    else:
        print usage()
        sys.exit(1)

    print_position_n_torque(1, 2)