/*
 * spatial_calibration.h
 *
 *  Created on: Jun 17, 2019
 *      Author: jberry
 */

#ifndef SPATIAL_CALIBRATION_H_
#define SPATIAL_CALIBRATION_H_

#include <opencv2/opencv.hpp>
#include "opencv2/ximgproc.hpp"
#include <opencv2/aruco/charuco.hpp>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <math.h>
#include <Eigen/Dense>

using namespace cv;
using namespace std;
using namespace Eigen;
using namespace ximgproc;

void saveCameraParams(const string &filename, Size imageSize,
                             const Mat &cameraMatrix, const Mat &distCoeffs);
void charuco_calibrate(string outfile, string calib_imgs, int dict_id, int nx, int ny, float mw, float aw);



#endif /* SPATIAL_CALIBRATION_H_ */
