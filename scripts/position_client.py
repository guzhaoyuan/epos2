#!/usr/bin/env python

from epos2.srv import *
import rospy
import sys

def request_position(x):
    # print("wait for Service")
    rospy.wait_for_service('moveToPosition')
    try:
        # print("now request service")
        moveToPosition = rospy.ServiceProxy('moveToPosition', Position)
        res = moveToPosition(x)
        return res
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

def usage():
    return "%s [position torque]"%sys.argv[0]

if __name__ == "__main__":
    # if len(sys.argv) == 3:
    #     position = int(sys.argv[1])
    #     torque = int(sys.argv[2])
    #     print(position,torque)
    # else:
    #     print usage()
    #     sys.exit(1)

    step = 0
    while(True):
        step += 1
        rospy.loginfo("request:%s",step)
        if step % 2:
            res = request_position(10)
        else:
            res = request_position(20)
        print res.position_new