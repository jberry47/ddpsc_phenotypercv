/*
 * machine_learning.h
 *
 *  Created on: Jul 16, 2019
 *      Author: jberry
 */

#ifndef MACHINE_LEARNING_H_
#define MACHINE_LEARNING_H_

#include <opencv2/opencv.hpp>
#include "opencv2/ximgproc.hpp"
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <math.h>
#include <Eigen/Dense>

using namespace cv;
using namespace cv::ml;
using namespace std;
using namespace Eigen;

void trainSVM(Mat img, Mat mask, string fname);
Mat predictSVM(Mat img, string fname);
void trainBC(Mat img, Mat mask, string fname);
Mat predictBC(Mat img, string fname);
void trainBoost(Mat img, Mat mask, string fname);
Mat predictBoost(Mat img, string fname);

#endif /* MACHINE_LEARNING_H_ */
