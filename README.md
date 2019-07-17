# DDPSC PhenotyperCV
Pipeline for analyzing images from the Bellwether Foundation Phenotyping Facility at Donald Danforth Plant Science Center


###Installing
PhenotyperCV is dependent on two packages: OpenCV and Eigen3. Additionally, the OpenCV installation must have been with the extra modules enabled, namely: aruco, ml, and ximgproc. This program must be compiled from source and is made easier with `cmake`  
First clone the repository
```bash
git clone https://github.com/jberry47/ddpsc_phenotypercv
```
Create a build directory
```bash
mkdir build && cd build
```
Build the program
```bash
cmake ..
```

If there were no errors or warnings (mainly about finding Eigen) then you can build the program
```bash
make
```
If you have an unconventional installation of Eigen you'll need to comment out the find_packages for Eigen in the CMakeList.txt and manually add the path to your installation with `-DCMAKE_MODULE_PATH=/path/to/insall/eigen/Eigen` after a successful build, you can read the help page with `./PhenotyperCV -h`
