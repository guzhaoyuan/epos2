#include "ros/ros.h"
#include "epos2/Torque.h"

#include "wrap.h"

ros::Time begin;
// ros::Duration interval(1.0); // 1s
ros::Duration interval(0,25000000); // 25ms
// ros::Duration interval(0,30000000); // 30ms
ros::Time next;

bool applyTorque(epos2::Torque::Request &req, epos2::Torque::Response &res)
{
	unsigned int ulErrorCode = 0;

	short current;
	int position_new, pVelocityIs;

	// ROS_INFO("now read position n current");
	get_position(g_pKeyHandle, g_usNodeId, &position_new, &ulErrorCode);
	res.position_new = position_new;
	// get_current(g_pKeyHandle, g_usNodeId, &current, &ulErrorCode);
	// get_velocity(g_pKeyHandle, g_usNodeId, &pVelocityIs, &ulErrorCode);
	// res.current = current;
	// res.velocity = pVelocityIs;
	
	// the force transform of the data type can cause problem
	while((next - ros::Time::now()).toSec()<0){
		next += interval;
		ROS_INFO("");
	}
	(next - ros::Time::now()).sleep();

	ROS_INFO("now write: position=%ld, torque=%ld", (long int)req.position, (long int)req.torque);
	SetCurrentMust(g_pKeyHandle, g_usNodeId, req.torque, &ulErrorCode);

	// ROS_INFO("now return");
	return true;
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "epos2");
	ros::NodeHandle n;

	begin = ros::Time::now();
	next = begin + interval;
	int lResult = MMC_FAILED;
	unsigned int ulErrorCode = 0;
	// Set parameter for usb IO operation
	SetDefaultParameters();
	// open device
	if((lResult = OpenDevice(&ulErrorCode))!=MMC_SUCCESS)
	{
		LogError("OpenDevice", lResult, ulErrorCode);
		return lResult;
	}
	
	SetEnableState(g_pKeyHandle, g_usNodeId, &ulErrorCode);
	ActivateProfileCurrentMode(g_pKeyHandle, g_usNodeId, &ulErrorCode);

	ros::ServiceServer service = n.advertiseService("applyTorque", applyTorque);

	ROS_INFO("Ready to move.");
	ros::spin();

	//disable epos
	SetDisableState(g_pKeyHandle, g_usNodeId, &ulErrorCode);
	//close device
	if((lResult = CloseDevice(&ulErrorCode))!=MMC_SUCCESS)
	{
		LogError("CloseDevice", lResult, ulErrorCode);
		return lResult;
	}
	return 0;
}