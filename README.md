# epos2

maxon DC motor current control, communicate with rospy

## Protocal

Torque.srv

	float64 position(can be none)
	float64 torque
	---
	float64 position_new
	float64 reward
	bool done
	float64 velocity(more info)
	short current

## EPOS2 info

using USB mode, no need to set baudrate

for EPOS2 setup for linux, see docs/EPOS Command Library.pdf

for driver and library, find 'EPOS_Linux_Library' on maxon official site(https://www.maxonmotorusa.com/maxon/view/product/control/Positionierung/347717)

- g_usNodeId = 1;
- g_deviceName = "EPOS2";
- g_protocolStackName = "MAXON SERIAL V2";
- g_interfaceName = "USB";
- g_portName = "USB0";

## Framework

node1: request position or torque, calculate on return

node2: receive request and execute, wait until end of this step, return new state and info, then wait for new request

## Files

### src

controller.cpp: node for controlling torque, now backup, not for use
test_epos2.cpp: test for develop
wrap.cpp: wrap api to make code clean and easy to program
time_test.cpp: test time

position_server.cpp: create service to control postion
current_server.cpp: create service to control current

### scripts

request.py: node for requesting service and get info from controller, now backup, not for use
position_client.py: request position control service
current_client.py: request current control service


### include

headings

## Usages

use as a ros package, put the whole epos2 files under workspace/src and compile with catkin

	#under ${workspace}
	catkin_make
	rosrun epos2 current_server
	rosrun epos2 current_client.py

## Benchmark

time taken:

- write position: **9ms**
- write current: **5ms**
- get current: **5ms**
- get position: **5ms**
- get velocity: **5ms**

get all 3 parameter takes about **14ms** on average

- ros send request immediately after get response: **7ms**(from the perspective of server)