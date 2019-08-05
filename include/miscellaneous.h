/*
 * miscellaneous.h
 *
 *  Created on: Jun 18, 2019
 *      Author: jberry
 */

#ifndef MISCELLANEOUS_H_
#define MISCELLANEOUS_H_

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

extern int roi_size;
extern Mat src;
extern Mat selMat;
extern int counter;

void split(const string& s, char c, vector<string>& v);
void onMouse( int event, int x, int y, int f, void* );
void kMouse( int event, int x, int y, int f, void* );
void showImage(Mat img);
void selectionGUI(Mat orig, string orig_fname,Mat mask, int size, string shape_fname, string color_fname);

#endif /* MISCELLANEOUS_H_ */
