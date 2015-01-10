/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>

#include <math_constants.h> //pi



// OpenGL Graphics includes
#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)


//show the Eclipse CDT a GLUT header it knows how to find (picky about headers inside frameworks).
#ifdef __CDT_PARSER__
#include <GL/glut.h>
#else
#include <GLUT/glut.h>
#endif



#else
#include <GL/freeglut.h>
#endif


// CUDA Runtime, Interop, and includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

//extra helpers for cuda-gl interop & error checking
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include <helper_functions.h> //for sdkTimer



#define INTERLEAVE_RGB 1

#if INTERLEAVE_RGB
#define ACCUM_SHADER_OUT_TYPE unsigned char

//GL format for texture where we copy output of interference kernel. GL_RGBA if kernel data is RGBA
#define ACCUM_TEXTURE_FORMAT GL_RGB

//Internal format of the texture for output of kernel. Was GL_RGB8
#define ACCUM_TEXTURE_SIZED_INTERNAL_FORMAT GL_RGB8

//bytes per pixel in Internal format of the texture for output of kernel. 1 for GL_RED, 4 for GL_RGBA
#define ACCUM_TEXTURE_VALUES_PER_PIX 3

#define ACCUM_TEXTURE_CHANS_PER_PIX 3 //rgba = 3, rgb = 3, r = 1
#else
//datatype for output of cuda interference kernel
#define ACCUM_SHADER_OUT_TYPE unsigned char

//GL format for texture where we copy output of interference kernel. GL_RGBA if kernel data is RGBA
#define ACCUM_TEXTURE_FORMAT GL_RED

//Internal format of the texture for output of kernel. Was GL_RGB8
#define ACCUM_TEXTURE_SIZED_INTERNAL_FORMAT GL_R8

//bytes per pixel in Internal format of the texture for output of kernel. 1 for GL_RED, 4 for GL_RGBA
#define ACCUM_TEXTURE_VALUES_PER_PIX 1

#define ACCUM_TEXTURE_CHANS_PER_PIX 1


#endif

//for debugging, linear might be nice, but in final output, we want fast & don't need interpolation (GL_NEAREST)
#define OUTPUT_TEXTURE_MIN_FILTER GL_NEAREST


//static const int WORK_SIZE = 256;
#define HOLO_PIXELS_COUNT 262144
//number of threads per cuda block
#define BLOCKSIZE 512 //hardware limit of 1024

//how many cuda blocks should we use to process hololine
#define PIXELS_PER_THREAD 1 //use a loop to compute multiple pixels in each CUDA thread?

static const long int HOLO_PIXELS = HOLO_PIXELS_COUNT;//65536; (2048*128 = 262144) //number of pixels per hololine
#define NBLOCKS (HOLO_PIXELS_COUNT/BLOCKSIZE/PIXELS_PER_THREAD)
long int nblocks = NBLOCKS; // break computation into this many blocks.
static const long int blocksize = BLOCKSIZE;

#define HOLOPOINT_STRIDE 1 //pad between holo points (1 is no padding)

static const long int HOLO_POINTS = 2;//256; //number of 3d points to simulate per line
#define SHOULD_PRECACHE_POINTS 0 //should we attempt to preload point data into shared memory?



//mark II-specific
#define ACTIVE_SCANS_PER_FRAME 8
#define DACS_PER_VGA_CABLE 3
#define HOLOLINES_PER_VGA_CABLE (ACTIVE_SCANS_PER_FRAME*DACS_PER_VGA_CABLE)
#define VGA_CABLES_PER_GPU 2
#define MARKII_VGA_WIDTH 2048
#define MARKII_VGA_HEIGHT 3514


#define HOLOLINES_PER_GPU (HOLOLINES_PER_VGA_CABLE*VGA_CABLES_PER_GPU) //if not rendering for mark II, can put any number here



#define HOLOLINE_TEX_WIDTH 2048 // Since texture width is limited, wrap hologram line when converting to a 2D texture.
int hololineTexWidth = HOLOLINE_TEX_WIDTH;
int textureLinesPerHololine = HOLO_PIXELS_COUNT/HOLOLINE_TEX_WIDTH;
int hololineTexHeight = HOLO_PIXELS_COUNT/HOLOLINE_TEX_WIDTH*HOLOLINES_PER_GPU/ACCUM_TEXTURE_CHANS_PER_PIX;

#define ALL_HOLOLINES_THIS_GPU_SIZE (HOLO_PIXELS_COUNT*HOLOLINES_PER_GPU*ACCUM_TEXTURE_VALUES_PER_PIX*sizeof(ACCUM_SHADER_OUT_TYPE))

static int fpsCount = 0;
static int fpsLimit = 1;
StopWatchInterface *timer = NULL;

bool stretchForDebug = true;
bool useMarkIIFormat = true;

int gui = 1;

// this mode is "old fashion" : use glTexSubImage2D() to update the final result
// commenting it will make the sample use the other way :
// map a texture in CUDA and blit the result into it
#define USE_TEXSUBIMAGE2D

int glWindowWidth = 1600;
int glWindowHeight = 1000;

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#ifdef DEBUG
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }
#else
#define CUDA_CHECK_RETURN(value) do{value;}while(0)
#endif


//__global__ void computeHologram(void* __restrict__ pixdata, void * __restrict__ pointdata, float gain, int totalpoints ) { // can tell compiler that pixdata & pointdata never overlap

__global__ void computeHologram(void*  pixdata, void *  pointdata, float gain, int totalpoints ) {


	////////constants
	const float pi = CUDART_PI_F;
	const float pi_x2 = pi*2.0f;

		//lambda=0.000633; //633nm in mm
	const float lambda=1.0f; //in hologram samples (hack)
	const float twopioverlamda = (pi_x2/lambda);



	const float screen_width = 2048.0f;
		//float screen_height = 1757.*2.;//1050.;1757.

	const float lay_data=128.0f; //data lines per hololine


	const float maxangle = pi/180.0f*15.0f; //clip any frequencies producing angles higher than this

	const float refangle = pi/180.0f*15.0f; //angle of the reference beam

	const float emitter_x_screenmidpoint = 0.5f*screen_width*lay_data; //halfway through emitter (in pixels)


	////////memory access

	//pixels is 1-d array of pixels representing a hololine
	ACCUM_SHADER_OUT_TYPE *pixels = (ACCUM_SHADER_OUT_TYPE*) pixdata;
	float4 *points = (float4*) pointdata; //points are xyzl with xyz in (-1 1) clip coordinates


	//TODO: investigate bank conlicts for 1.x devices (may want to adjust order of these writes)
	//TODO: support multiple points per thread if totalpoints > blocksize

#if SHOULD_PRECACHE_POINTS
	__shared__ float4 pointcache[HOLO_POINTS*HOLOPOINT_STRIDE];

#if HOLOPOINTS == BLOCKSIZE
	pointcache[threadIdx.x * HOLOPOINT_STRIDE] = points[threadIdx.x];
#else

	int idx = threadIdx.x;
	while(idx<HOLO_POINTS) {
		pointcache[idx * HOLOPOINT_STRIDE] = points[idx + HOLO_POINTS*treadIdx.y];
		idx += blocksize;
	}
#endif
	__syncthreads();
#endif

#if PIXELS_PER_THREAD > 1
	for (int px=0;px < PIXELS_PER_THREAD;px++){
		//////pixel-location dependent
		unsigned int pixel_in_hololine = PIXELS_PER_THREAD * blockIdx.x * blockDim.x + threadIdx.x*PIXELS_PER_THREAD + px;
#else
		unsigned int pixel_in_hololine = blockIdx.x * blockDim.x + threadIdx.x;

#endif
		float refOffset=twopioverlamda*sinf(refangle) * pixel_in_hololine;

		float out = 0.0f;

		for (int p=0;p<totalpoints;p++) {

#if SHOULD_PRECACHE_POINTS
			float4 pointpos = pointcache[p * HOLOPOINT_STRIDE];
#else
			float4 pointpos = points[p + HOLO_POINTS*threadIdx.y];
#endif

			float emitter_z = 500000.0f*pointpos.z; // emitter distance in hologram samples (converting model units to pixel units)

			float emitter_x = emitter_x_screenmidpoint*(pointpos.x + 1.0f);


			//full chirp

			float sample_emitter_offset = pixel_in_hololine - emitter_x;


			float wavedirection = atan2f(sample_emitter_offset,emitter_z);


			//float emitter_to_sample = sqrt(emitter_z*emitter_z + sample_emitter_offset*sample_emitter_offset);
			float emitter_to_sample = emitter_z/cosf(wavedirection);

			float arg = copysignf(emitter_to_sample*twopioverlamda,emitter_z);;

			float object_angle = atan2f(abs(sample_emitter_offset),abs(emitter_z));


			if(object_angle > maxangle)
			{
				//continue;
			} else {


				float a = 0.5f + 0.5f*cosf(object_angle/maxangle*pi); // amplitude falloff over angle using raised cosine window. Can use a sharper window for brighter high-angle views

				float lum = 0.5f+ 0.5f*cosf(arg + refOffset); //angled chirp

				//out += a * lum * 1.0;//gain;
				out += a * lum * pointpos.w * gain;
				//out += lum * pointpos.w;
			}
		}


//Write pixel data to memory (to be mapped to texture)
#if INTERLEAVE_RGB
		int texline = blockIdx.y/ACCUM_TEXTURE_VALUES_PER_PIX; //texline bumps once per VALUES hololines
		int chan = blockIdx.y%ACCUM_TEXTURE_VALUES_PER_PIX; //interleave VALUES output pixels into one texture pixel
		int pixstart = pixel_in_hololine*ACCUM_TEXTURE_CHANS_PER_PIX; //stretch pixel pointer by striding CHANS spaces per pixel
	((ACCUM_SHADER_OUT_TYPE*)pixels)[chan + pixstart + HOLO_PIXELS_COUNT*ACCUM_TEXTURE_CHANS_PER_PIX*texline] = out;
#else
	((ACCUM_SHADER_OUT_TYPE*)pixels)[pixel_in_hololine + HOLO_PIXELS_COUNT*blockIdx.y] = out;//out;//out;//rintf(out); //round to integer with rintf
#endif


#if PIXELS_PER_THREAD > 1
	}
#endif

}

//version of compute kernel that only touches data in/out without any actual computation (to get lower bound on runtime)
__global__ void computeHologramTouchonly(void*  pixdata, void *  pointdata, float gain, int totalpoints ) {
	ACCUM_SHADER_OUT_TYPE *pixels = (ACCUM_SHADER_OUT_TYPE*) pixdata;
	float4 *points = (float4*) pointdata; //points are xyzl with xyz in (-1 1) clip coordinates
#if PIXELS_PER_THREAD > 1
	for (int px=0;px < PIXELS_PER_THREAD;px++){
		//////pixel-location dependent
		unsigned int pixel_in_hololine = PIXELS_PER_THREAD * blockIdx.x * blockDim.x + threadIdx.x*PIXELS_PER_THREAD + px;
#else
		unsigned int pixel_in_hololine = blockIdx.x * blockDim.x + threadIdx.x;
		float out = 0;
		for (int p=0;p<totalpoints;p++) {
			float4 pointpos = points[p];
			out += pointpos.x + pointpos.y + pointpos.z + pointpos.w;
		}

#endif

#if INTERLEAVE_RGB
		int texline = blockIdx.y/ACCUM_TEXTURE_VALUES_PER_PIX; //texline bumps once per VALUES hololines
		int chan = blockIdx.y%ACCUM_TEXTURE_VALUES_PER_PIX; //interleave VALUES output pixels into one texture pixel
		int pixstart = pixel_in_hololine*ACCUM_TEXTURE_CHANS_PER_PIX; //stretch pixel pointer by striding CHANS spaces per pixel
	((ACCUM_SHADER_OUT_TYPE*)pixels)[chan + pixstart + HOLO_PIXELS_COUNT*texline] = out;
#else
	((ACCUM_SHADER_OUT_TYPE*)pixels)[pixel_in_hololine + HOLO_PIXELS_COUNT*blockIdx.y] = out;
#endif


#if PIXELS_PER_THREAD > 1
	}
#endif

}

GLuint pbo = 0;     // OpenGL pixel buffer object
GLuint tex = 0;     // OpenGL texture object


// pbo and fbo variables
#ifdef USE_TEXSUBIMAGE2D
GLuint pbo_dest;
struct cudaGraphicsResource *cuda_pbo_dest_resource;// CUDA Graphics Resource (to transfer PBO)
#else
unsigned int *cuda_dest_resource;
GLuint shDrawTex;  // draws a texture
struct cudaGraphicsResource *cuda_tex_result_resource;
#endif

//GLuint framebuffer;     // to bind the proper targets
//GLuint depth_buffer;    // for proper depth test while rendering the scene
//GLuint tex_screen;      // where we render the image
GLuint tex_cudaResult;  // where we will copy the CUDA result




//float hpixels[HOLO_PIXELS];
ACCUM_SHADER_OUT_TYPE* hpixels = NULL;
void* pixdata = NULL;
float4 hpoints[HOLO_POINTS * HOLOLINES_PER_GPU];
void* pointdata = NULL;

unsigned int size_tex_data;
unsigned int num_texels;
unsigned int num_values;



//float brightness = 1.0f;

//allocate texture that will get result from CUDA kernel
void createTextureDst(GLuint *tex_cudaResult, unsigned int size_x, unsigned int size_y);
void copyKernelResultToTexture();
void displayImage(GLuint texture);

#ifdef USE_TEXSUBIMAGE2D
///////////////////////////////////////////////////////////cuGL/////////////////////
//! Create PBO
////////////////////////////////////////////////////////////////////////////////
void
createPBO(GLuint *pbo, struct cudaGraphicsResource **pbo_resource)
{
    // set up vertex data parameter
    num_texels = hololineTexWidth * hololineTexHeight;
    num_values = num_texels * ACCUM_TEXTURE_CHANS_PER_PIX;
    size_tex_data = sizeof(ACCUM_SHADER_OUT_TYPE) * num_values;
    void *data = malloc(size_tex_data);

    // create buffer object
    glGenBuffers(1, pbo);
    glBindBuffer(GL_ARRAY_BUFFER, *pbo);
    glBufferData(GL_ARRAY_BUFFER, size_tex_data, data, GL_DYNAMIC_DRAW); //multi-write, for GL texture
//    glBufferData(GL_ARRAY_BUFFER, size_tex_data, data, GL_STREAM_DRAW); //single write for GL texture
    free(data);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(pbo_resource, *pbo, cudaGraphicsRegisterFlagsWriteDiscard));
    SDK_CHECK_ERROR_GL();
}

void
deletePBO(GLuint *pbo)
{
    glDeleteBuffers(1, pbo);
    SDK_CHECK_ERROR_GL();
    *pbo = 0;
}
#endif

void initGL(int * argc, char **argv)
{
    // initialize GLUT callback functions
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(glWindowWidth, glWindowHeight);
    glutCreateWindow("CUDA Hologram Rendering");

    glewInit();

    if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object"))
    {
        printf("Required OpenGL extensions missing.");
        exit(EXIT_SUCCESS);
    }
    // create pbo
#ifdef USE_TEXSUBIMAGE2D
    createPBO(&pbo_dest, &cuda_pbo_dest_resource);
#endif
    // create texture that will receive the result of CUDA
    createTextureDst(&tex_cudaResult, hololineTexWidth, hololineTexHeight);

    // create texture for blitting onto the screen
    //createTextureSrc(&tex_screen, image_width, image_height);
    //createRenderBuffer(&tex_screen, image_width, image_height); // Doesn't work

    /*
    // load shader programs
   shDrawPot = compileGLSLprogram(NULL, glsl_drawpot_fragshader_src);

#ifndef USE_TEXSUBIMAGE2D
   shDrawTex = compileGLSLprogram(glsl_drawtex_vertshader_src, glsl_drawtex_fragshader_src);
#endif
*/
   SDK_CHECK_ERROR_GL();

   sdkCreateTimer(&timer);


}

//allocate texture that will get result from CUDA kernel
void createTextureDst(GLuint *tex_cudaResult, unsigned int size_x, unsigned int size_y)
{
    // create a texture
    glGenTextures(1, tex_cudaResult);
    glBindTexture(GL_TEXTURE_2D, *tex_cudaResult);

    // set basic parameters0
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, OUTPUT_TEXTURE_MIN_FILTER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

#ifdef USE_TEXSUBIMAGE2D
    glTexImage2D(GL_TEXTURE_2D, 0, ACCUM_TEXTURE_SIZED_INTERNAL_FORMAT, size_x, size_y, 0, ACCUM_TEXTURE_FORMAT, GL_UNSIGNED_BYTE, NULL);
    SDK_CHECK_ERROR_GL();
#else
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI_EXT, size_x, size_y, 0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_BYTE, NULL);
    SDK_CHECK_ERROR_GL();
    // register this texture with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterImage(&cuda_tex_result_resource, *tex_cudaResult,
                                                GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));
#endif
}


void copyKernelResultToTexture() {
#ifdef USE_TEXSUBIMAGE2D
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo_dest);

    glBindTexture(GL_TEXTURE_2D, tex_cudaResult);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
    		hololineTexWidth, hololineTexHeight,
                    ACCUM_TEXTURE_FORMAT, GL_UNSIGNED_BYTE, NULL);
    SDK_CHECK_ERROR_GL();
    glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
#else
    // We want to copy cuda_dest_resource data to the texture
    // map buffer objects to get CUDA device pointers
    cudaArray *texture_ptr;
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_tex_result_resource, 0));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex_result_resource, 0, 0));

    int num_texels = hololineTexWidth * hololineTexHeight;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;
    checkCudaErrors(cudaMemcpyToArray(texture_ptr, 0, 0, cuda_dest_resource, size_tex_data, cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_tex_result_resource, 0));
#endif
}


void displayImageForMarkII(GLuint texture) {
	//input texture needs blanking added
    glBindTexture(GL_TEXTURE_2D, texture);
    glEnable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0.0, MARKII_VGA_WIDTH, MARKII_VGA_HEIGHT, 0.0, -1.0, 1.0); //(0,0) in top left

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    if(stretchForDebug) {
    	glViewport(0, 0, glWindowWidth, glWindowHeight);
    } else {
    	glViewport(0, 0, hololineTexWidth, hololineTexHeight);
    }

    float lineytx = 1.0/(HOLOLINES_PER_GPU/ACCUM_TEXTURE_VALUES_PER_PIX);

    float lineypx = textureLinesPerHololine;

    int direction = 1;

    int ypixel = 0;

    int topskip = 387;

    int stride = 128 + 48;


    ypixel = topskip;
    for(int i=0; i<HOLOLINES_PER_GPU/ACCUM_TEXTURE_VALUES_PER_PIX ;i++) {
		glBegin(GL_QUADS);

		float txmin = 0.0;
		float txmax = 1.0;

		float tymin =  i*lineytx;
		float tymax = (i+1)*lineytx;

		if(direction < 0) {
			float tmp = txmin;
			txmin = txmax;
			txmax = tmp;

			tmp = tymin;
			tymin = tymax;
			tymax = tmp;

		}

		glTexCoord2f(txmin, tymin);
		glVertex3f(0, ypixel, 0.5);

		glTexCoord2f(txmax, tymin);
		glVertex3f(hololineTexWidth-1, ypixel, 0.5);

		glTexCoord2f(txmax, tymax);
		glVertex3f(hololineTexWidth-1, ypixel + lineypx, 0.5);

		glTexCoord2f(txmin, tymax);
		glVertex3f(0, ypixel + lineypx, 0.5);
		glEnd();
		if (i == ACTIVE_SCANS_PER_FRAME-1) {
			ypixel = MARKII_VGA_HEIGHT/2 + topskip;
		} else {
			ypixel += stride;
		}
		direction = -direction;
    }
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glDisable(GL_TEXTURE_2D);

}

// display image to the screen as textured quad
void displayImage(GLuint texture)
{
    glBindTexture(GL_TEXTURE_2D, texture);
    glEnable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    if(stretchForDebug) {
    	glViewport(0, 0, glWindowWidth, glWindowHeight);
    } else {
    	glViewport(0, 0, hololineTexWidth, hololineTexHeight);
    }

    // if the texture is a 8 bits UI, scale the fetch with a GLSL shader

#ifndef USE_TEXSUBIMAGE2D
    glUseProgram(shDrawTex);
    GLint id = glGetUniformLocation(shDrawTex, "texImage");
    glUniform1i(id, 0); // texture unit 0 to "texImage"
    SDK_CHECK_ERROR_GL();
#endif


    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0);
    glVertex3f(-1.0, -1.0, 0.5);
    glTexCoord2f(1.0, 0.0);
    glVertex3f(1.0, -1.0, 0.5);
    glTexCoord2f(1.0, 1.0);
    glVertex3f(1.0, 1.0, 0.5);
    glTexCoord2f(0.0, 1.0);
    glVertex3f(-1.0, 1.0, 0.5);
    glEnd();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glDisable(GL_TEXTURE_2D);


#ifndef USE_TEXSUBIMAGE2D
    glUseProgram(0);
#endif

    SDK_CHECK_ERROR_GL();
}



// render image using CUDA
void render()
{
    //copyInvViewMatrix(invViewMatrix, sizeof(float4)*3);

    // map PBO to get CUDA device pointer
    GLuint *d_output;
    // map PBO to get CUDA device pointer
    CUDA_CHECK_RETURN(cudaGraphicsMapResources(1, &cuda_pbo_dest_resource, 0));
    size_t num_bytes;
    CUDA_CHECK_RETURN(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes,
                                                         cuda_pbo_dest_resource));
    //printf("CUDA mapped PBO: May access %ld bytes\n", num_bytes);

    // clear image -- test without
    CUDA_CHECK_RETURN(cudaMemset(d_output, 0, hololineTexWidth*hololineTexHeight*ACCUM_TEXTURE_VALUES_PER_PIX));


    //copy 3d points to graphics memory
	CUDA_CHECK_RETURN(cudaMemcpy(pointdata, hpoints, sizeof(float4) * HOLO_POINTS * HOLOLINES_PER_GPU, cudaMemcpyHostToDevice));
						//computeHologram<<<nblocks,blocksize>>>(pixdata,pointdata,1.0,HOLO_POINTS);

	//dim3 computeBlockDims(nblocks,HOLOLINES_PER_GPU);
	dim3 computeBlockDims(nblocks,HOLOLINES_PER_GPU);
	computeHologram<<<computeBlockDims,blocksize>>>(d_output,pointdata,255.0/HOLO_POINTS,HOLO_POINTS);

	//computeHologramTouchonly<<<computeBlockDims,blocksize>>>(d_output,pointdata,255.0/HOLO_POINTS,HOLO_POINTS);

	CUDA_CHECK_RETURN(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
		CUDA_CHECK_RETURN(cudaGetLastError());


    CUDA_CHECK_RETURN(cudaGraphicsUnmapResources(1, &cuda_pbo_dest_resource, 0));
}

// display results using OpenGL (called by GLUT)
void display()
{
    sdkStartTimer(&timer);

    /*
    // use OpenGL to build view matrix
    GLfloat modelView[16];
    GLfloat invViewMatrix[16];
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    //glRotatef(-viewRotation.x, 1.0, 0.0, 0.0);
    //glRotatef(-viewRotation.y, 0.0, 1.0, 0.0);
    //glTranslatef(-viewTranslation.x, -viewTranslation.y, -viewTranslation.z);
    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
    glPopMatrix();

    invViewMatrix[0] = modelView[0];
    invViewMatrix[1] = modelView[4];
    invViewMatrix[2] = modelView[8];
    invViewMatrix[3] = modelView[12];
    invViewMatrix[4] = modelView[1];
    invViewMatrix[5] = modelView[5];
    invViewMatrix[6] = modelView[9];
    invViewMatrix[7] = modelView[13];
    invViewMatrix[8] = modelView[2];
    invViewMatrix[9] = modelView[6];
    invViewMatrix[10] = modelView[10];
    invViewMatrix[11] = modelView[14];
*/
    glClear(GL_COLOR_BUFFER_BIT);

    render();

    copyKernelResultToTexture();

    if(useMarkIIFormat) {
    	displayImageForMarkII(tex_cudaResult);
    } else {
    	displayImage(tex_cudaResult);

    }
    sdkStopTimer(&timer);


    glutSwapBuffers();
    glutPostRedisplay();

    glutReportErrors();

    // Update fps counter, fps/title display and log
    if (++fpsCount == fpsLimit)
    {
        char cTitle[256];
        float fps = 1000.0f / sdkGetAverageTimerValue(&timer);
#ifdef DEBUG
        sprintf(cTitle, "CUDA Shader render of %d points (%d blocks of %d): %.2f fps (DEBUG)", (int)HOLO_POINTS, (int)NBLOCKS, (int)BLOCKSIZE, fps);
#else
        sprintf(cTitle, "CUDA Shader render of %d points (%d blocks of %d): %.2f fps", (int)HOLO_POINTS, (int)NBLOCKS, (int)BLOCKSIZE, fps);

#endif
        glutSetWindowTitle(cTitle);
        //printf("%s\n", cTitle);
        fpsCount = 0;
        fpsLimit = (int)((fps > 1.0f) ? fps : 1.0f);
        sdkResetTimer(&timer);
    }

}




/************************Base code for single line render*******************/
extern "C"  void renderOneLine() {

	//could comment back in to do validation on single scanline.
	/*
	CUDA_CHECK_RETURN(
				cudaMemcpy(pointdata, hpoints, sizeof(float4) * HOLO_POINTS, cudaMemcpyHostToDevice));

	computeHologram<<<nblocks,blocksize>>>(pixdata,pointdata,1.0,HOLO_POINTS);

	CUDA_CHECK_RETURN(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
		CUDA_CHECK_RETURN(cudaGetLastError());
		CUDA_CHECK_RETURN(cudaMemcpy(hpixels, pixdata, sizeof(ACCUM_SHADER_OUT_TYPE) * HOLO_PIXELS, cudaMemcpyDeviceToHost));

	if(!gui) {
		for (long int i = 0; i < HOLO_PIXELS; i++) {
			printf("%g ", (double)hpixels[i]);
				//printf("Input value: %u, device output: %u\n", idata[i], odata[i]);
		}
	}
	*/
}


void cleanup() {
	if(pixdata) {
		CUDA_CHECK_RETURN(cudaFree((void*) pixdata));
		pixdata = NULL;
	}
	if(pointdata) {
		CUDA_CHECK_RETURN(cudaFree((void*) pointdata));
		pointdata = NULL;
	}
	if(hpixels) {
		free(hpixels);
		hpixels = NULL;
	}

	CUDA_CHECK_RETURN(cudaDeviceReset());
	exit(EXIT_SUCCESS);

}

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void
keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
        case (27) :
            cleanup();
            break;

        case ' ':
            stretchForDebug = !stretchForDebug;
            break;
        case 'm':
            useMarkIIFormat = !useMarkIIFormat;
            break;


    }
}

void reshape(int w, int h)
{
    glWindowWidth = w;
    glWindowHeight = h;
}


/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(int argc, char** argv) {


	//***************************

	float xpos = 0.0;
	float zpos = -1.6;
	float ypos = 0.16;
	if (argc > 1) {
		nblocks = atoi(argv[1]);
	}

	if (argc > 2) {
		xpos = atof(argv[2]);
	}

	if (argc > 3) {
		zpos = atof(argv[3]);
	}


	//init random points
	for(int i=0;i<HOLO_POINTS * HOLOLINES_PER_GPU;i++) {
		hpoints[i].x = xpos;//00.9;//0.6;
		hpoints[i].y = ypos;
		hpoints[i].z = zpos;
		hpoints[i].w = 1.0;
	}

	//loadpcd("test.pcd",(float*) hpoints, 1);

	/*
	for(long int i=0;i<HOLO_PIXELS;i++) {
		hpixels[i]= 0;
	}*/

	CUDA_CHECK_RETURN(cudaMalloc((void**) &pointdata, sizeof(float4) * HOLO_POINTS * HOLOLINES_PER_GPU));


if(gui)
    {


		initGL(&argc, argv);
        // This is the normal rendering path for VolumeRender
        glutDisplayFunc(display);
        glutKeyboardFunc(keyboard);
        //glutMouseFunc(mouse);
        //glutMotionFunc(motion);
        glutReshapeFunc(reshape);
        //glutIdleFunc(idle);

//        initPixelBuffer();

        atexit(cleanup);

        glutMainLoop();


} else {

	//float hpixels[HOLO_PIXELS];
	hpixels = (ACCUM_SHADER_OUT_TYPE*)malloc(sizeof(ACCUM_SHADER_OUT_TYPE)*HOLO_PIXELS);

	CUDA_CHECK_RETURN(cudaMalloc((void**) &pixdata, sizeof(float) * HOLO_PIXELS)); //uninitialized
	//force pixels to zero before running. This is not needed
	//CUDA_CHECK_RETURN(cudaMemcpy(pixdata, hpixels, sizeof(float) * HOLO_POINTS, cudaMemcpyHostToDevice));

	renderOneLine();
}
cleanup();

}
