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

static bool saveCameraParams(const string &filename, Size imageSize,
                             const Mat &cameraMatrix, const Mat &distCoeffs) {
    FileStorage fs(filename, FileStorage::WRITE);
    if(!fs.isOpened())
        return false;

    time_t tt;
    time(&tt);
    struct tm *t2 = localtime(&tt);
    char buf[1024];
    strftime(buf, sizeof(buf) - 1, "%c", t2);

    fs << "calibration_time" << buf;

    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;

    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;

    return true;
}

bool charuco_calibrate(string outfile, string calib_imgs, int dict_id, int nx, int ny, float mw, float aw){
	Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dict_id));
		cv::Ptr<cv::aruco::CharucoBoard> board = cv::aruco::CharucoBoard::create(nx, ny, mw, aw, dictionary);
		vector< vector< vector< Point2f > > > allCorners;
		vector< vector< int > > allIds;
		vector< Mat > allImgs;
		Size imgSize;

		cout << "Gathering images..." << endl;
		int number_of_lines = 0;
		string line;
		ifstream myfile(calib_imgs);
		while (std::getline(myfile, line)){
			++number_of_lines;
			Mat image, imageCopy;
			image = imread(line);
			vector< int > ids;
			vector< vector< Point2f > > corners, rejected;

			// detect markers
			cv::aruco::detectMarkers(image, dictionary, corners, ids);
			Mat currentCharucoCorners, currentCharucoIds;
			if(ids.size() > 0){
				aruco::interpolateCornersCharuco(corners, ids, image, board, currentCharucoCorners,currentCharucoIds);
			}
	        allCorners.push_back(corners);
	        allIds.push_back(ids);
	        allImgs.push_back(image);
	        imgSize = image.size();
		}

		double repError;
		Mat cameraMatrix, distCoeffs;

		vector< vector< Point2f > > allCornersConcatenated;
		vector< int > allIdsConcatenated;
		vector< int > markerCounterPerFrame;
		markerCounterPerFrame.reserve(allCorners.size());
		for(unsigned int i = 0; i < allCorners.size(); i++) {
		    markerCounterPerFrame.push_back((int)allCorners[i].size());
		        for(unsigned int j = 0; j < allCorners[i].size(); j++) {
		            allCornersConcatenated.push_back(allCorners[i][j]);
		            allIdsConcatenated.push_back(allIds[i][j]);
		        }
		    }

		cout << "calibrating camera - aruco" << endl;
	    double arucoRepErr;
	    arucoRepErr = aruco::calibrateCameraAruco(allCornersConcatenated, allIdsConcatenated,
	                                              markerCounterPerFrame, board, imgSize, cameraMatrix,
	                                              distCoeffs, noArray(), noArray());

	    int nFrames = (int)allCorners.size();
	    vector< Mat > allCharucoCorners;
	    vector< Mat > allCharucoIds;
	    vector< Mat > filteredImages;
	    allCharucoCorners.reserve(nFrames);
	    allCharucoIds.reserve(nFrames);
	    for(int i = 0; i < nFrames; i++) {
	            // interpolate using camera parameters
	            Mat currentCharucoCorners, currentCharucoIds;
	            aruco::interpolateCornersCharuco(allCorners[i], allIds[i], allImgs[i], board,
	                                             currentCharucoCorners, currentCharucoIds, cameraMatrix,
	                                             distCoeffs);

	            allCharucoCorners.push_back(currentCharucoCorners);
	            allCharucoIds.push_back(currentCharucoIds);
	            filteredImages.push_back(allImgs[i]);
	    }

		cout << "calibrating camera - charuco" << endl;
	    repError =
	            aruco::calibrateCameraCharuco(allCharucoCorners, allCharucoIds, board, imgSize,
	                                          cameraMatrix, distCoeffs, noArray(), noArray());

	    bool saveOk =  saveCameraParams(outfile, imgSize, cameraMatrix, distCoeffs);
	    cout << "Rep Error ChArUco: " << repError << endl;
	    cout << "Rep Error ArUco: " << arucoRepErr << endl;
	    cout << "Calibration saved to " << outfile << endl;
	    return(saveOk);
}
