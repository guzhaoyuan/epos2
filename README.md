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