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
extern Mat src, src1, dst, selMat,gray;
extern int counter;

extern int threshold_value;
extern int threshold_type;
extern int max_value;
extern int max_type;
extern int max_BINARY_value;

void split(const string& s, char c, vector<string>& v);
void onMouse( int event, int x, int y, int f, void* );
void kMouse( int event, int x, int y, int f, void* );
void showImage(Mat img, string title);
void find_and_measure_selection(vector<vector<Point> > sel_contours,vector<vector<Point> > pred_contours, string color, Mat &map, int threshold_value,string shape_fname, string color_fname, string orig_fname,vector<Mat> split_lab);
void selectionGUI(Mat orig, string orig_fname,Mat mask, int size, string shape_fname, string color_fname, int threshold_value);
void confusionGUI(Mat orig, Mat predicted, Mat labeled, int size);
void thresholdGUI( int, void* );

#endif /* MISCELLANEOUS_H_ */
