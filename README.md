<img src="www/raw.png" width="33%"></img><img src="www/mask.png" width="33%"></img><img src="www/classified.png" width="33%"></img>

# DDPSC PhenotyperCV
PhenotyperCV is a header-only C++11 library that has a large range of functionality for image-based plant phenotyping. 

Included is a single source file that, when built, has workflows for many pratical applications. The created executable has multiple features that are selected for using `-m` flag indicating the "mode" you'd like execute. As development continues, new modes will become available to expand the use of this program to problems outside the framework of Bellwether platform. Futher information can be found using the `-h` flag. 

Custom workflows, with the aid of functions in this library, are also made possible and are intended to ease the difficulty in processing images from any source. A common bottleneck of image processing is not knowing how to perform specific tasks and this library is designed with that in mind.

### Usage
Visit [the wiki page](https://github.com/jberry47/ddpsc_phenotypercv/wiki) for more info on how to use included source file and how to create your own custom workflows. 

### Building the program on a local computer
PhenotyperCV is dependent on two packages: OpenCV and Eigen3. Additionally, the OpenCV installation must have been with the extra modules enabled, namely: aruco, ml, and ximgproc. This program must be compiled from source and is made easier with `cmake`

##### (Mac Only) Installing Homebew
```bash 
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

##### Installing CMake
For Ubuntu users:
```bash
sudo apt-get install cmake
```
For MacOSX users:
```bash
brew install cmake
```

##### Making a directory for all builds
```bash
mkdir big_build && cd big_build
export my_path=$(pwd)
```

##### Getting Eigen3 and installing
```bash
wget "https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.bz2" -O"eigen-3.3.7.tar"
tar -xf eigen-3.3.7.tar && cd eigen-3.3.7
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
sudo make install
sudo cp -r /usr/local/include/eigen3/Eigen /usr/local/include/
cd $my_path
```

##### Getting OpenCV extra modules
```bash
wget https://github.com/opencv/opencv_contrib/archive/4.5.2.tar.gz -O"opencv_contrib-4.5.2.tar.gz"
tar -xzf opencv_contrib-4.5.2.tar.gz
cd opencv_contrib-4.5.2/modules
rm -rv !("aruco"|"ximgproc") 
cd $my_path
```

##### Getting OpenCV and installing
```bash
wget https://github.com/opencv/opencv/archive/4.5.2.tar.gz -O"opencv-4.5.2.tar.gz"
tar -xzf opencv-4.5.2.tar.gz
cd opencv-4.5.2
mkdir build && cd build
cmake .. -DOPENCV_EXTRA_MODULES_PATH=$my_path/opencv_contrib-4.5.2/modules
make -j8
sudo make install
cd $my_path
```
##### Getting ZBar and installing
For Ubuntu users:
```bash
sudo apt-get install libzbar-dev libzbar0
```
For MacOSX users:
```bash
brew install zbar
```

##### Getting PhenotyperCV and building
```bash
git clone https://github.com/jberry47/ddpsc_phenotypercv
cd ddpsc_phenotypercv && mkdir build && cd build
cmake ..
make
export PATH=$PATH:$my_path/ddpsc_phenotypercv/build
```

### Building the program on DDPSC Infrastructure
Both OpenCV and Eigen3 depedencies are in unconventional locations that are not found with cmake. To build the program on the infrastructure, there is a bash script in the misc directory of this repo that links to everything needed, and can be run with: 
```bash 
cd ddpsc_phenotypercv/misc
./pull_compile_phenocv.sh
cd ../../
```
Alternatively, a pre-built executable exists already in `/home/jberry/programs/PhenotyperCV`.

After a successful build, you need to include the libraries to OpenCV and ZBar in your LD_LIBRARY_PATH
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/shares/bioinfo/installs/opencv-3.3.0/install/lib:/bioinfo/lib
```
It is recommended that you add this line to your bash_profile or bash_rc so you don't have to add the libraries every time you want to use PhenotyperCV.

### Testing the build
After you create the executable, test to make sure it can be used by running `PhenotyperCV -h` which should display a help page. 

### DDPSC Datasci Cluster Usage
Example condor job file and accompanying executable for processing images on the DDPSC infrastructure can be found in the `misc` directory
* **phenotypercv.submit** - condor submit file that MUST BE EDITED to your file paths and image location
* **run_phenocv.sh** - the executable that the job file calls and MUST BE EDITED to your file paths and image location

