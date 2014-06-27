#ifndef JKINECTFRAME_H_
#define JKINECTFRAME_H_

#ifndef __APPLE__
#include <libfreenect/libfreenect.h>
#else

#endif

#define KINECT_FRAME_PIX (640*480)

#define KINECT_XYZL_FRAME_PIX (KINECT_FRAME_PIX*4)

#define KINECT_XYZRGB_FRAME_PIX (KINECT_FRAME_PIX*6)

#define KINECT_SHMEM_KEY 6317 //arbitrary
// 04/09/2011 SKJ: define SHMEM_KEY for second Kinect
#define KINECT_SHMEM_KEY_2 12634

#define KINECT_SHMEM_XYZL_KEY 8346

#define KINECT_SHMEM_XYZL_KEY_2 8584

#define KINECT_SHMEM_XYZRGB_KEY 8337

#define KINECT_SHMEM_XYZRGB_KEY_2 8932


struct JKinectFrame
{
	unsigned char luma[KINECT_FRAME_PIX];
	float depth[KINECT_FRAME_PIX];
};

struct JKinectFrameXYZL
{
	int stamp;
	int count;
	float xyzl[KINECT_XYZL_FRAME_PIX];
};

struct JKinectFrameXYZRGB
{
	int stamp;
	int count;
	float xyzrgb[KINECT_XYZRGB_FRAME_PIX];
};

#endif
