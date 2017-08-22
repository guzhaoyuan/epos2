#!/usr/bin/env python

from epos2.srv import *
import rospy
import sys

def request_position(position, isAbsolute=0):
    # print("wait for Service")
    rospy.wait_for_service('moveToPosition')
    try:
        # print("now request service")
        moveToPosition = rospy.ServiceProxy('moveToPosition', Position)
        res = moveToPosition(position, isAbsolute)
        return res
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

def usage():
    return "%s [position isAbsolute(0/1)]"%sys.argv[0]

if __name__ == "__main__":
    if len(sys.argv) == 2:
        position = int(sys.argv[1])
        isAbsolute = 0
        res = request_position(position, isAbsolute)
        print res.position_new
    elif len(sys.argv) == 3:
        position = int(sys.argv[1])
        isAbsolute = int(sys.argv[2])
        res = request_position(position, isAbsolute)
        print res.position_new
    else:
        print usage()

        step = 0
        while(True):
            step += 1
            rospy.loginfo("request:%s",step)
            if step % 2:
                res = request_position(10)
            else:
                res = request_position(20)
            print res.position_new