/*
 * feature_extraction.h
 *
 *  Created on: Jun 17, 2019
 *      Author: jberry
 */

#ifndef FEATURE_EXTRACTION_H_
#define FEATURE_EXTRACTION_H_

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

float get_fd(Mat mask);
vector<double> get_shapes(vector<Point> cc,Mat mask);
Mat get_color(Mat img,Mat mask);
Mat get_nir(Mat img,Mat mask);
int is_oof(Mat img);
vector<Point> keep_roi(Mat img,Point tl, Point br, Mat &mask);



#endif /* FEATURE_EXTRACTION_H_ */
