<img src="www/raw.png" width="33%"></img><img src="www/mask.png" width="33%"></img><img src="www/classified.png" width="33%"></img>

# DDPSC PhenotyperCV
This program has multiple features that are selected for using `-m` flag indicating the "mode" you'd like execute. As development continues, new modes will become available to expand the use of this program to problems outside the framework of Bellweather platform. Current modes are listed here and futher information can be found using the `-h` flag. 

### Modes
* **VIS** - Segment and measure plant in RGB images
* **VIS_CH** - Color correct, segment, and measure plant in RGB images
* **VIS_CH_CHECK** - Color correct, and output image for viewing
* **SET_TARGET** - Obtain and print to stdout the RGB information for each of the chips in the image
* **DRAW_ROIS** - GUI for making card_masks/ images to be used by VIS_CH, VIS_CH_CHECK, and SET_TARGET
* **AVG_IMGS** - Pipe in list of input images to be averaged and outputs average_images.png
* **NIR** - Segment and measure plant in near-infrared images
* **CHARUCO_CREATE** - Creates a ChArUco board with specified sizes and dictionary
* **CHARUCO_CALIB** - Camera calibration using multiple viewpoints of a ChArUco board
* **CHARUCO_EST** - Warps the image to the orthogonal plane projection using calibration file from CHARUCO_CALIB
* **SVM_CREATE** - Creates a SVM classifier from input image and respective labeled image
* **SVM_PRED** - Uses provided SVM classifier to predict features in input image
* **BC_CREATE** - Creates a Bayesian classifer from input image and respective labeled image
* **BC_PRED** - Uses provided Bayes classifier to predict features in input image

### VIS_CH Workflow
1. Average together all the empty pots using AVG_IMGS
2. Create card_masks directory and use DRAW_ROIS to create the chip masks
3. Set the averaged image to target for color correction using SET_TARGET
4. Run the analysis with the averaged image as background using VIS_CH

### VIS Workflow
1. Average together all the empty pots using AVG_IMGS
2. Run the analysis with the averaged image as background using VIS

In both the VIS and VIS_CH, the plant is segmented using the following protocol: 
<img src="www/pheno3_segmentation.png"></img>

### NIR Workflow
1. Average together all the empty pots using AVG_IMGS
2. Run the analysis with the averaged image as background using NIR


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

