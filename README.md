#epos2
maxon DC motor current control, communicate with rospy

## Protocal

torque.srv
float64 position(can be none)
float64 torque
---
float64 position_new
float64 reward
bool done
float64 velocity(more info)
