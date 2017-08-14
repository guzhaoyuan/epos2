#include "wrap.h"

const string g_programName = "TEST";

int main(int argc, char** argv)
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
	// read position
	// if((lResult = get_position(g_pKeyHandle, g_usNodeId, &ulErrorCode))!=MMC_SUCCESS)
	// {
	// 	LogError("CloseDevice", lResult, ulErrorCode);
	// 	return lResult;
	// }

	//enable epos
	SetEnableState(g_pKeyHandle, g_usNodeId, &ulErrorCode);
	//enable position mode
	ActivateProfileVelocityMode(g_pKeyHandle, g_usNodeId, &ulErrorCode);
	//position, absolute, error info
	// moveToPosition(g_pKeyHandle, g_usNodeId, 5000, 0, &ulErrorCode);
	MoveWithVelocity(g_pKeyHandle, g_usNodeId, 100, &ulErrorCode);
	sleep(3);

	//disable epos
	SetDisableState(g_pKeyHandle, g_usNodeId, &ulErrorCode);
	//close device
	if((lResult = CloseDevice(&ulErrorCode))!=MMC_SUCCESS)
	{
		LogError("CloseDevice", lResult, ulErrorCode);
		return lResult;
	}
	return lResult;
}



