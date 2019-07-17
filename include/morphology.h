/*
 * morphology.h
 *
 *  Created on: Jun 17, 2019
 *      Author: jberry
 */

#ifndef MORPHOLOGY_H_
#define MORPHOLOGY_H_

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

Mat find_endpoints(Mat input);
Mat find_branchpoints(Mat input);
Mat length_filter(Mat input, int size);
Mat prune(Mat input, int size);
Mat segment_skeleton(Mat input, bool colored=false);
Mat find_leaves(Mat skel, Mat tips);
Mat add_stem(Mat classified_skel, Mat full_skel);
Mat fill_mask(Mat mask, Mat classified_in);
vector<vector<double> > get_leaf_info(Mat classified_skel, Mat filled_mask);


#endif /* MORPHOLOGY_H_ */
