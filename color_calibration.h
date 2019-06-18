/*
 * color_calibration.h
 *
 *  Created on: Jun 17, 2019
 *      Author: jberry
 */

#ifndef COLOR_CALIBRATION_H_
#define COLOR_CALIBRATION_H_


float extractRGB_chips(Mat img,Mat &mask);
MatrixXd getRGBarray(Mat img);
void get_standardizations(Mat img, float &det, MatrixXd &rh,MatrixXd &gh,MatrixXd &bh);
Mat color_homography(Mat img, MatrixXd r_coef,MatrixXd g_coef,MatrixXd b_coef);


#endif /* COLOR_CALIBRATION_H_ */
