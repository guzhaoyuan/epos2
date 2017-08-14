#include "ros/ros.h"
#include "epos2/Torque.h"

#include "wrap.h"

bool moveToPosition(epos2::Torque::Request &req, epos2::Torque::Response &res)
{
	unsigned int ulErrorCode = 0;
	// the force transform of the data type can cause problem
	ROS_INFO("request: position=%ld, torque=%ld", (long int)req.position, (long int)req.torque);
	moveToPosition(g_pKeyHandle, g_usNodeId, (long)req.position, 0, &ulErrorCode);
	res.position_new = 1;
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
	ActivateProfilePositionMode(g_pKeyHandle, g_usNodeId, &ulErrorCode);

	ros::init(argc, argv, "epos2");
	ros::NodeHandle n;

	ros::ServiceServer service = n.advertiseService("moveToPosition", moveToPosition);

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