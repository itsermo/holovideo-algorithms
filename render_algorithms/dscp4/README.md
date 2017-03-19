# DSCP4 Holovideo Algorithm

Diffraction-specific panoramagram for full-color MIT/BYU Mark IV holovideo display

## Synopsis

dscp4 is a real-time computer generated hologram (CGH) algorithm for full-color holovideo displays based on surface acoustic wave acousto-optic modulators.

 dscp4 is a CMAKE package, that is composed of three C++11 projects:
 
- libdscp4 - the render library that implements DSCP4 and computes fringe
               patterns. libdscp4 is built as a shared library with C-linkage.
               It generates libdscp4.so and dscp4.h files, which are installed
               to /usr/local/lib and /usr/local/include/dscp4 respectively.
               The dscp4.h file includes functions to initialize the renderer,
               change renderer settings, add 3d model vertices and color, and
               manipulate 3d models in the renderer.
	       
- dscp4    - the "test" program that loads 3D model files and feeds 3D model 
               data to libdscp4 and controls the rendering through libdscp4.
               This program builds as an executable and is installed to
               /usr/local/bin, and can be run anywhere from the command line.
               This program can be built optionally, but if you want to see
               stuff on the holovideo display, you'll have to use this
			   or dscp4-qt, the GUI version.

- dscp4-qt - the GUI version of "dscp4" test program. This is built with Qt
               and allows one to control all aspects of libdscp4 with a GUI,
               optional, but highly recommended install.

For more information on how this algorithm works, refer to the following publication:

- S. Jolly, E. Dreshaj, and V. M. Bove, Jr., “Computation of Fresnel holograms and diffraction-specific coherent panoramagrams for full-color holographic displays based on anisotropic leaky-mode modulators,” Proc. SPIE Practical Holography XXIX, 9386, 2015. [PDF](http://obm.media.mit.edu/wp-content/uploads/sites/10/2012/09/PW2015.pdf)

## Dependencies

- Cross-platform Make (CMake) v2.8.2+
- GNU Make or equivalent.
- GCC-4.8.2 C++11 capable compiler.
	(CUDA 6.5 is only compatible with GCC <= 4.8)
	(CUDA 7.0 supports GCC 4.9 but has removed kepler support)
- Boost C++ Libraries v1.54+ [HEADERS and LIBRARIES]
    (system, filesystem, program_options)
- OpenGL 3.1 (must have GPU that has OpenGL 3.1 support)
- GLEW 1.11.0+ (OpenGL Extension Wrangler, required)
- GLM 0.9.6.1+ (header-only library for common OpenGL math functions,
	such as matrix multiplication)
- SDL 2.0.1+ (takes care of window management, OpenGL render contexts)
- CUDA 6.5 (optional, fringe computation is done in CUDA)
- OpenCL 1.1 (optional, fringe computation done in OpenCL)
- Log4Cxx v0.10.0+ (optional, for logging errors and info)
- ASSIMP 3.1.1+ (optional, for importing 3D object files, like .ply files)
- zLib 1.2.2+ (optional, for PNG screenshots)
- libpng 1.2 (optional, for PNG screenshots)
- Qt 5.4 (optional, for building dscp4-qt, a GUI controller for libdscp4)
  	
 CMAKE is a cross-platform make tool that allows one to generate make projects
 for specific compilers.  For example, we can use CMAKE to generate a
 Visual Studio, Eclipse, XCode or standard GNU Make project which then can be
 compiled, debugged or installed from the respective OS and toolchain.

 This readme will show you how to generate a GNU Make project, that can be
 compiled and installed from the command line on Linux.

## Dependencies

 Installing the above dependencies can be done mostly via your package manager,
 for example, in Ubuntu, you would type "sudo apt-get install <package-name>".
 Here is how to do this in Ubuntu:
 
 a) Sync your package manager with the latest packages:
	
Ubuntu/Debian:
	
	sudo apt-get update
	sudo apt-get upgrade

Fedora/RedHat:

	sudo yum update
 
 a) Install the dependenices using the command below:

Ubuntu/Debian:
 
	sudo apt-get install cmake g++ libboost-all-dev libglew-dev libpng-dev \
	zlib1g-dev libglm-dev libsdl2-dev libassimp-dev liblog4cxx10-dev \
	qt5base-dev

Fedora/RedHat:

	sudo yum install gcc gcc-c++ automake autoconf cmake boost-devel \
	glew-devel libpng-devel glm-devel SDL2-devel assimp-devel \
	log4cxx-devel qt5-qtbase-devel
 
 c) Install CUDA 6.5 by navigating to 
    https://developer.nvidia.com/cuda-downloads
    download the 64-bit "RUN" file for x86 Linux, Ubuntu 14.04.i This can also
    be done with the wget command:

	cd ~/Downloads
	wget http://developer.download.nvidia.com/compute/cuda/6_5/rel/installers/cuda_6.5.14_linux_64.run

Open up the terminal, navigate to the download folder and run the .RUN file:
	
	chmod +x cuda_6.5.14_linux_64.run
	sudo ./cuda_6.5.14_linux_64.run

Say "NO" to installing accelerated graphics drivers, otherwise it may break
your current setup.
	
The "chmod" command gives the file executable permission so it can be run,
running the file with "./" will execute and install the cuda libraries
  
## Building
 
This project uses the Cross-platform Make (CMake) build system. However, we
have conveniently provided a wrapper configure script and Makefile so that
the typical build invocation of "./configure" followed by "make" will work.
For a list of all possible build targets, use the command "make help".
 
If you're comfortable with CMAKE, feel free to generate and compile like so:
 
	cd build
	cmake .. -DCMAKE_BUILD_TYPE=Release
	make
 
 NOTE: Users of CMake may believe that the top-level Makefile has been
 generated by CMake; it hasn't, so please do not delete that file.
 
 Here are some available options that can be used:
 
 	-DBUILD_DSCP4_APP={YES|NO} (If this is NO, only the DSCP4 library will be
			    built, and libdscp4.so dscp4.h will be installed.
			    This is probably not what you want.)

	-DWITH_LIBLOG4CXX={YES|NO} (If, for whatever reason, you despise console output
			    or can't install liblog4cxx, this can turn off all 
			    dependencies to log4cxx and turn off logging)

	-DWITH_TRACE_LOG={YES|NO}  (Toggles performance logging. Use with "-v 6" 
					command option)

 	-DWITH_OPENCL={YES|NO}	    (Toggles OpenCL fringe computation capability)

 	-DWITH_CUDA={YES|NO}	    (Toggles NVIDIA CUDA fringe computation capability)

 	-DWITH_PNG={YES|NO}	    (Toggles screenshot capability)

 	-DBUILD_DSCP4_QT_APP=NO    (Turns off dscp4-qt app building)
 
 	-DBUILD_COPY_MODELS=NO		(Turns off copying of 3D models on install)

 These can be used like so:

	./configure -DWITH_LIBLOG4CXX=NO
	make
 
 or
 
	cd build
	cmake -DWITH_LIBLOG4CXX=NO ..
	make
							 
## Installing

 Once the project has been built (see "BUILDING"), enter the following command:

	sudo make install
 
 The install function will build (if necessary) and copy (if files are newer):
  * dscp4.h and dscp4_defs.h to /usr/local/include/dscp4
  * libdscp4.so to /usr/local/lib
  * dscp4 and dscp4-qt executables to /usr/local/bin
  * 3d model files from ./models to /usr/local/share/dscp4/models
  * shader files from libdscp4/shaders to /usr/local/share/dscp4/shaders
  * OpenCL kernel files from libdscp4/kernels to /usr/local/share/dscp4/kernels
  * it will generate dscp4.conf from ./dscp4.conf.in and copy to /etc/dscp4
  * it will generate dscp4.desktop and dscp4-qt.desktop and copy to
    /usr/local/share/applications. This creates shortcuts for X/Linux desktop
	shells. Look for dscp4 and dscp4-qt programs under "Scientific" category
	in your programs menu.  You may need to log out and log in to see it
	appear when you install dscp4 the first time.
 
 ## Configuring
 
 You can set many program, algorithm and display options by editing the file
 '/etc/dscp4/dscp4.conf'. Please have a look at this file before you do any
 editing of code.
 
 ## Running
 
 To run the program, just type 'dscp4' anywhere in the command line.
 For example:
 
 	dscp4 --help
 
 Will print out all of the command options.
 
	dscp4 -i dragon.ply
 
 Will open dscp4 with the dragon.ply model file.  First it will search for
 "dragon.ply" locally, if it can't be found, it will navigate to the
 model_path, as defined by /etc/dscp4.conf and look for the file there.

 From the render window, there are many useful keyboard shortcuts:

  - Use the arrow keys to rotate the view
  - Hold LEFT SHIFT and arrow keys, {,} to translate the object
  - Hold LEFT SHIFT and W,A,S,D,Z,X to change light position
  - Hold LEFT SHIFT and =,- to change the far clipping plane
  - Press =,- to change the near clipping plane  
  - Press "R" key to toggle model spinning, arrow keys will set spin rate
  - Press "F" key to toggle fullscreen mode
  - CTRL + S will dump the current color and depth buffer to PNG files
    in modelview, stereogram, and aerial modes.  In "holovideo" mode, it
	will write the fringe pattern data onto PNG files.
  - U will force the renderer to redraw
  - SPACEBAR toggles the render window view mode from color to depth buffer
  - Q will close the render window

## Uninstalling
 
 Simply running the following command from the CMAKE build path:
	
	sudo make uninstall

 Will delete all traces of dscp4 from your computer
 
 ## Using
 
 libdscp4 has 4 modes of operation:
- "model viewing" - Opens an OpenGL window and shows the
  models with normal, perspective projection.
- "stereogram view" - Opens an OpenGL window and shows models
  with orthographic shearing projection (aka panoramagrams).
  Use this mode to debug and see what panoramagram is
  generated before fringe computation.
- "aerial" - Opens a fullscreen window for every monitor,
  with each window displaying a horizontal parallax shift.
  This mode is for projector setups where discrete views are
  shown for each parallax shift.
- "holovideo" - Computes the fringe pattern from the
  panoramagram and writes the fringe pattern to a textured
  fullscreen OpenGL window for every GPU.  Each window spans
  multiple heads per GPU (one window per GPU).  This requires
  Xinerama mode to be enabled in the X11 configuration.
