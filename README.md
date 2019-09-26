<img src="www/raw.png" width="33%"></img><img src="www/mask.png" width="33%"></img><img src="www/classified.png" width="33%"></img>

# DDPSC PhenotyperCV
This is a header-only library that has a large range of functionality for image-based plant phenotyping. Included is a single source file that, when built, has workflows for many pratical applications. The created executable has multiple features that are selected for using `-m` flag indicating the "mode" you'd like execute. As development continues, new modes will become available to expand the use of this program to problems outside the framework of Bellweather platform. Futher information can be found using the `-h` flag. 

### Workflows
Visit [the wiki page](https://github.com/jberry47/ddpsc_phenotypercv/wiki) for more info on how to use this program

### Building the program
PhenotyperCV is dependent on two packages: OpenCV and Eigen3. Additionally, the OpenCV installation must have been with the extra modules enabled, namely: aruco, ml, and ximgproc. This program must be compiled from source and is made easier with `cmake`

##### Making a directory for all builds
```bash
mkdir big_build && cd big_build
export my_path=$(pwd)
```

##### Getting Eigen3 and installing
```bash
wget "http://bitbucket.org/eigen/eigen/get/3.3.7.tar.bz2" -O"eigen-3.3.7.tar"
tar -xf eigen-3.3.7.tar
cd eigen-eigen-323c052e1731
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr ..
sudo make install
sudo cp -r /usr/include/eigen3/Eigen /usr/include/
```

##### Getting OpenCV extra modules
```bash
wget https://github.com/opencv/opencv_contrib/archive/4.1.1.tar.gz -O"opencv_contrib-4.1.1.tar.gz"
tar -xzf opencv_contrib-4.1.1.tar.gz
cd opencv_contrib-4.1.1/modules
rm -rv !("aruco"|"ximgproc") 
```

##### Getting OpenCV and installing
```bash
wget https://github.com/opencv/opencv/archive/4.1.1.tar.gz -O"opencv-4.1.1.tar.gz"
tar -xzf opencv-4.1.1.tar.gz
cd opencv-4.1.1
mkdir build && cd build
cmake .. -DOPENCV_EXTRA_MODULES_PATH=$my_path/opencv_contrib-4.1.1/modules
make -j8
sudo make install
```

##### Getting PhenotyperCV and building
```bash
git clone https://github.com/jberry47/ddpsc_phenotypercv
cd ddpsc_phenotypercv && mkdir build && cd build
cmake ..
make
export PATH=$PATH:$my_path/ddpsc_phenotypercv/build
```

If you have an unconventional installation of Eigen you'll need to comment out the find_packages for Eigen in the CMakeList.txt and manually add the path to your installation with `-DCMAKE_MODULE_PATH=/path/to/install/eigen/Eigen` 

### Building the program on DDPSC Infrastructure
Both OpenCV and Eigen3 depedencies are in unconventional locations that are not found with cmake. To build the program on the infrastructure, `misc/pull_compile_phenocv.sh` is a bash script that MUST BE EDITED to your file paths and will first pull the repository, and execute a series of g++ commands that will create the executable and clean up all temporary files during the build. Alternatively, a pre-built executable exists already in `/home/jberry/programs/PhenotyperCV`.

After a successful build, you can read the help page with `./PhenotyperCV -h`

### DDPSC Datasci Cluster Usage
Example condor job file and accompanying executable for processing images on the DDPSC infrastructure can be found in the `misc` directory
* **phenotypercv.submit** - condor submit file that MUST BE EDITED to your file paths and image location
* **run_phenocv.sh** - the executable that the job file calls and MUST BE EDITED to your file paths and image location

