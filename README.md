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

controller.cpp: node for controlling torque
test_epos2.cpp: test for develop
wrap.cpp: wrap api to make code clean and easy to program

### scripts

request.py: node for requesting service and get info from controller

### include

headings

## Usages

use as a ros package, put the whole epos2 files under workspace/src and compile with catkin

## Benchmark

time taken:

- write position: **9ms**
- write current: **5ms**
- get current: **5ms**
- get position: **5ms**

- ros send request immediately after get response: **7ms**(from the perspective of server)