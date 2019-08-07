/*
 * color_calibration.h
 *
 *  Created on: Jun 17, 2019
 *      Author: jberry
 */

#ifndef COLOR_CALIBRATION_H_
#define COLOR_CALIBRATION_H_

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

float extractRGB_chips(Mat img,Mat &mask);
MatrixXd getRGBarray(Mat img);
void get_standardizations(Mat img, float &det, MatrixXd &rh,MatrixXd &gh,MatrixXd &bh);
Mat color_homography(Mat img, MatrixXd r_coef,MatrixXd g_coef,MatrixXd b_coef);
Mat CLAHE_correct_rgb(Mat img);
Mat CLAHE_correct_gray(Mat img);
Mat nonUniformCorrect(Mat img, int kernel_siz);


#endif /* COLOR_CALIBRATION_H_ */
