#include "ros/ros.h"
#include "epos2/Torque.h"

#include "wrap.h"

bool applyTorque(epos2::Torque::Request &req, epos2::Torque::Response &res)
{
	unsigned int ulErrorCode = 0;
	// the force transform of the data type can cause problem
	ROS_INFO("request: position=%ld, torque=%ld", (long int)req.position, (long int)req.torque);
	SetCurrentMust(g_pKeyHandle, g_usNodeId, req.torque, &ulErrorCode);

	short current;
	int position_new, pVelocityIs;

	ROS_INFO("now read position n velocity n current");

	get_position(g_pKeyHandle, g_usNodeId, &position_new, &ulErrorCode);
	get_current(g_pKeyHandle, g_usNodeId, &current, &ulErrorCode);
	get_velocity(g_pKeyHandle, g_usNodeId, &pVelocityIs, &ulErrorCode);
	res.current = current;
	res.position_new = position_new;
	res.velocity = pVelocityIs;

	ROS_INFO("now return");
	return true;
}

int main(int argc, char **argv)
{
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

	ros::init(argc, argv, "epos2");
	ros::NodeHandle n;

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