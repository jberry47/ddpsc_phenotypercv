/* Title: PhenotyperCV
 * Author: Jeffrey Berry (Bart Lab)
 *
 * Description: This program is for segmenting and measuring plants from the Bellweather Phenotyping
 * Facility. Segmentation is achieved by supplying a background image that does not contain a plant
 * and using the difference between that and a supplied image to threshold on. Further processing is
 * done to remove artifacts that arise. After segmentation is complete, shapes and color profile are
 * reported in corresponding user-specified files.
 *
 * Usage: ./PhenotyperCV --help  for more information
 *
 * Compiling on Danforth Center Bioinformatics Infrastructure:
 * g++ -I/shares/bioinfo/installs/opencv-3.3.0/install/include/ -I/shares/bioinfo/installs/eigen/Eigen -I/home/jberry/programs/PhenotyperCV/ddpsc_phenotypercv -O0 -g3 -Wall -c -std=c++11 -MMD -MP -MF"./ddpsc_phenotypercv/phenotypercv.d" -MT"./ddpsc_phenotypercv/phenotypercv.d" -o "./ddpsc_phenotypercv/phenotypercv.o" "./ddpsc_phenotypercv/phenotypercv.cpp"
 * g++ -I/shares/bioinfo/installs/opencv-3.3.0/install/include/ -I/shares/bioinfo/installs/eigen/Eigen -I/home/jberry/programs/PhenotyperCV/ddpsc_phenotypercv -O0 -g3 -Wall -c -std=c++11 -MMD -MP -MF"./ddpsc_phenotypercv/color_calibration.d" -MT"./ddpsc_phenotypercv/color_calibration.d" -o "./ddpsc_phenotypercv/color_calibration.o" "./ddpsc_phenotypercv/color_calibration.cpp"
 * g++ -I/shares/bioinfo/installs/opencv-3.3.0/install/include/ -I/shares/bioinfo/installs/eigen/Eigen -I/home/jberry/programs/PhenotyperCV/ddpsc_phenotypercv -O0 -g3 -Wall -c -std=c++11 -MMD -MP -MF"./ddpsc_phenotypercv/spatial_calibration.d" -MT"./ddpsc_phenotypercv/spatial_calibration.d" -o "./ddpsc_phenotypercv/spatial_calibration.o" "./ddpsc_phenotypercv/spatial_calibration.cpp"
 * g++ -I/shares/bioinfo/installs/opencv-3.3.0/install/include/ -I/shares/bioinfo/installs/eigen/Eigen -I/home/jberry/programs/PhenotyperCV/ddpsc_phenotypercv -O0 -g3 -Wall -c -std=c++11 -MMD -MP -MF"./ddpsc_phenotypercv/feature_extraction.d" -MT"./ddpsc_phenotypercv/feature_extraction.d" -o "./ddpsc_phenotypercv/feature_extraction.o" "./ddpsc_phenotypercv/feature_extraction.cpp"
 * g++ -I/shares/bioinfo/installs/opencv-3.3.0/install/include/ -I/shares/bioinfo/installs/eigen/Eigen -I/home/jberry/programs/PhenotyperCV/ddpsc_phenotypercv -O0 -g3 -Wall -c -std=c++11 -MMD -MP -MF"./ddpsc_phenotypercv/morphology.d" -MT"./ddpsc_phenotypercv/morphology.d" -o "./ddpsc_phenotypercv/morphology.o" "./ddpsc_phenotypercv/morphology.cpp"
 * g++ -L/shares/bioinfo/installs/opencv-3.3.0/install/lib ./ddpsc_phenotypercv/color_calibration.o ./ddpsc_phenotypercv/feature_extraction.o ./ddpsc_phenotypercv/morphology.o ./ddpsc_phenotypercv/phenotypercv.o ./ddpsc_phenotypercv/spatial_calibration.o -lopencv_core -lopencv_aruco -lopencv_calib3d -lopencv_highgui -lopencv_features2d -lopencv_imgproc -lopencv_imgcodecs -o "PhenotyperCV"
 *
 * To use the program in a parallelized fashion, pipe find into xargs with the -P flag followed by
 * number of cores like this:
 * 	Usage VIS or VIS_CH:
 * 		find Images/ -name 'VIS_SV*' | xargs -P8 -I{} ./PhenotyperCV -m=VIS_CH -i={} -b=background_image.png -s=shapes.txt -c=color.txt'
 * 	Usage NIR:
 * 		find Images/ -name 'NIR_SV*' | xargs -P8 -I{} ./PhenotyperCV -m=NIR -i={} -b=background_image.png -c=nir_color.txt
 */
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

#include <morphology.h>
#include <spatial_calibration.h>
#include <color_calibration.h>
#include <feature_extraction.h>
#include <miscellaneous.h>
#include <machine_learning.h>

using namespace cv;
using namespace cv::ml;
using namespace std;
using namespace Eigen;
using namespace ximgproc;

namespace {
const char* keys  =
        "{m        |       | Mode to run }"
        "{h        |       | Show help documentation }"
        "{i        |       | Input image }"
        "{b        |       | Background image }"
        "{size     |       | Square size (pixels) for DRAW_ROI mode}"
        "{class    |       | machine learning classifier}"
        "{prob     |       | Probability for WS thresh}"
        "{s        |       | Shape file to write to }"
        "{c        |       | Color file to write to }"
		"{ci       |       | ChArUco calibrate input file }"
		"{cc       |       | Camera calibration file name }"
		"{nx       |       | Number of board spaces - x }"
		"{ny       |       | Number of board spaces - y }"
		"{mw       |       | Marker width }"
		"{aw       |       | ArUco width }"
		"{method   |       | bayes or svm for mode=WS }"
		"{debug    |       | If used, write out final mask }"
		"{d        |       | dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,"
		        "DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, "
		        "DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12,"
		        "DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16}";
}

int main(int argc, char *argv[]){
    CommandLineParser parser(argc, argv, keys);
	string mode;
	if(!parser.has("m")){
		mode = "-h";
	}else{
		mode = parser.get<String>("m");
	}

	bool bool_nir = mode=="NIR";
	bool bool_vis = mode=="VIS";
	bool bool_vis_CH = mode=="VIS_CH";
	bool bool_vis_CH_check = mode=="VIS_CH_CHECK";
	bool bool_getCH = mode=="SET_TARGET";
	bool bool_avgImgs = mode=="AVG_IMGS";
	bool bool_drawROIS = mode == "DRAW_ROIS";
	bool bool_charucoCalib = mode == "CHARUCO_CALIB";
	bool bool_charuco_est = mode == "CHARUCO_EST";
	bool bool_charucoCreate = mode == "CHARUCO_CREATE";
	bool bool_svmCreate = mode == "SVM_CREATE";
	bool bool_svmPred = mode == "SVM_PRED";
	bool bool_svmStat = mode == "SVM_STAT";
	bool bool_bcCreate = mode == "BC_CREATE";
	bool bool_bcPred = mode == "BC_PRED";
	bool bool_bcStat = mode == "BC_STAT";
	bool bool_testing = mode == "TESTING";
	bool bool_ws = mode == "WS";

	if(bool_testing){
		Mat inputImage = imread(parser.get<string>("i"));
		Mat corrected = nonUniformCorrect(inputImage,15);
		imwrite("nu_corrected.png",corrected);
		/* for taylor
		Mat inputImage = imread(parser.get<string>("i"));
	    Mat lab;
   		cvtColor(inputImage, lab, cv::COLOR_BGR2Lab);
   		vector<Mat> split_lab;
   		split(lab, split_lab);

   		//-- necrotic_mask
   		Mat l_thresh;
   		threshold(split_lab[0],l_thresh,20,255,THRESH_BINARY);
		Mat a_thresh;
		threshold(split_lab[1],a_thresh,119,255,THRESH_BINARY_INV);
		Mat mask1 = l_thresh - (l_thresh & a_thresh);
		Mat b_thresh;
		threshold(split_lab[2],b_thresh,145,255,THRESH_BINARY);
		Mat mask2 = ((255-b_thresh) & l_thresh);
		Mat nec = mask1-(mask2 & mask1);
		Mat n_erode0;
		erode(nec,n_erode0, Mat(), Point(-1, -1), 2, 1, 1);
		Mat n_dilate;
		dilate(n_erode0, n_dilate, Mat(), Point(-1, -1), 6, 1, 1);
		Mat nec_mask;
		erode(n_dilate,nec_mask, Mat(), Point(-1, -1), 4, 1, 1);

		//-- watersoaking mask
		Mat w_thresh;
   		threshold(split_lab[0],w_thresh,65,255,THRESH_BINARY_INV);
		Mat w_dilate0;
		dilate(w_thresh, w_dilate0, Mat(), Point(-1, -1), 6, 1, 1);
		Mat w_erode1;
		erode(w_dilate0,w_erode1, Mat(), Point(-1, -1), 5, 1, 1);
		Mat w_mask = w_erode1 | nec_mask;
   		Mat l_thresh1;
   		threshold(split_lab[0],l_thresh1,20,255,THRESH_BINARY_INV);
   		Mat m_or = w_mask-l_thresh1;
   		Mat m_erode0;
		erode(m_or,m_erode0, Mat(), Point(-1, -1), 4, 1, 1);
		Mat m_dilate;
		dilate(m_erode0, m_dilate, Mat(), Point(-1, -1), 4, 1, 1);
		selectionGUI(inputImage,parser.get<string>("i"),m_dilate,parser.get<int>("size"), parser.get<string>("s"),parser.get<string>("c"));
		*/

		/*
		Mat inputImage = imread(parser.get<string>("i"));
		Mat msk = imread(parser.get<string>("b"),0);
		Mat mask;
		threshold(msk,mask,25,255,THRESH_BINARY);
		trainBoost(inputImage, mask, parser.get<string>("s"));
		*/

		/*
		Mat inputImage = imread(parser.get<string>("i"));
		Mat response = predictBoost(inputImage,parser.get<string>("s"))*255;
		imwrite("boost_pred.png",response);
		 */

		//Mat inputImage = imread(parser.get<string>("i"));
		//Mat CLAHE_corrected = CLAHE_correct_rgb(inputImage);
		//Mat nirImage = imread(parser.get<string>("i"),0);
    	//Mat CLAHE_corrected = CLAHE_correct_gray(nirImage);
		//imwrite("clahe_corrected.png",CLAHE_corrected);
	}
	else if(bool_ws){
		if(!(parser.has("i") && parser.has("class") && parser.has("size") && parser.has("s")  && parser.has("c") && parser.has("prob") && parser.has("method"))){
			cout << "Using mode WS requires input: -i=inputImage -class=input_svm_classifier.yaml -size=number(range 0-20) -s=shapes_output.txt -c=gray_output.txt -prob=decimal(range 0-1) -method=[bayes,svm]" << endl;
		}else{
			Mat inputImage = imread(parser.get<string>("i"));
			Mat response;
			string suffix;
			if(parser.get<string>("method") == "bayes"){
				suffix = "_bayes_pred.png";
				response = predictBC(inputImage,parser.get<string>("class"));
			}else if(parser.get<string>("method") == "svm"){
				suffix = "_svm_pred.png";
				response = predictSVM(inputImage,parser.get<string>("class"));
			}else{
				cout << "Unknown method: expecting either bayes or svm" << endl;
				return(1);
			}

			Mat filt;
			bilateralFilter(response.clone(),filt,30,50,50);

			Mat r_thresh;
			int val = parser.get<float>("prob")*255;
			threshold(filt,r_thresh,val,255,THRESH_BINARY);
			Mat r_dilate;
			dilate(r_thresh,r_dilate, Mat(), Point(-1, -1), 1, 1, 1);

			selectionGUI(inputImage,parser.get<string>("i"),r_dilate.clone(),parser.get<int>("size"), parser.get<string>("s"),parser.get<string>("c"));

			vector<string> sub_str;
			const string full_str = string(parser.get<string>("i"));
			char del = '.';
			split(full_str,del,sub_str);
			string new_name = sub_str[0]+suffix;
			imwrite(new_name,filt);
		}
	}
	else if(bool_bcCreate){
		if(!(parser.has("i") && parser.has("b") && parser.has("s"))){
			cout << "Using mode BC_CREATE requires input: -i=inputImage -b=labeledImage -s=output_classifier.yaml" << endl;
		}else{
			Mat inputImage = imread(parser.get<string>("i"));
			Mat labels = imread(parser.get<string>("b"),0);
			//Mat mask;
			//threshold(msk,mask,25,255,THRESH_BINARY);
			trainBC(inputImage, labels, parser.get<string>("s"));
		}
	}
	else if(bool_bcPred){
		if(!(parser.has("i") && parser.has("s"))){
			cout << "Using mode BC_PRED requires input: -i=inputImage -s=classifier.yaml" << endl;
		}else{
			Mat inputImage = imread(parser.get<string>("i"));
			//Mat corrected = nonUniformCorrect(inputImage,5);
			Mat response = predictBC(inputImage,parser.get<string>("s"));

			vector<string> sub_str;
			const string full_str = string(parser.get<string>("i"));
			char del = '.';
			split(full_str,del,sub_str);
			string new_name = sub_str[0]+"_bayes_pred.png";
			imwrite(new_name,response);
		}
	}
	else if(bool_bcStat){
		if(!(parser.has("i") && parser.has("s") && parser.has("b") && parser.has("prob") && parser.has("size"))){
			cout << "Using mode BC_STAT requires input: -i=inputImage -b=labeledImage -s=classifier.yaml -prob=decimal(range 0-1) -size=integer(range 0-20)" << endl;
		}else{
			Mat inputImage = imread(parser.get<string>("i"));
			Mat labels = imread(parser.get<string>("b"),0);
			Mat response = predictBC(inputImage,parser.get<string>("s"));
			Mat filt;
			bilateralFilter(response.clone(),filt,30,50,50);
			Mat r_thresh,l_thresh;
			int val = parser.get<float>("prob")*255;
			threshold(filt,r_thresh,val,255,THRESH_BINARY);
			threshold(labels,l_thresh,25,255,THRESH_BINARY);
			confusionGUI(inputImage, r_thresh, l_thresh, parser.get<int>("size"));
		}
	}
	else if(bool_svmCreate){
		if(!(parser.has("i") && parser.has("b") && parser.has("s"))){
			cout << "Using mode SVM_CREATE requires input: -i=inputImage -b=labeledImage -s=output_classifier.yaml -size=box_size -prob=decimal(range 0-1)" << endl;
		}else{
			Mat inputImage = imread(parser.get<string>("i"));
			Mat labels = imread(parser.get<string>("b"),0);
			//Mat mask;
			//threshold(msk,mask,25,255,THRESH_BINARY_INV);
			trainSVM(inputImage, labels, parser.get<string>("s"));
		}
	}
	else if(bool_svmPred){
		if(!(parser.has("i") && parser.has("s"))){
			cout << "Using mode SVM_PRED requires input: -i=inputImage -s=classifier.yaml" << endl;
		}else{
			Mat inputImage = imread(parser.get<string>("i"));
			Mat response = predictSVM(inputImage,parser.get<string>("s"));
			vector<string> sub_str;
			const string full_str = string(parser.get<string>("i"));
			char del = '.';
			split(full_str,del,sub_str);
			string new_name = sub_str[0]+"_svm_pred.png";
			imwrite(new_name,response);
		}
	}
	else if(bool_svmStat){
		if(!(parser.has("i") && parser.has("s") && parser.has("b") && parser.has("prob") && parser.has("size"))){
			cout << "Using mode SVM_STAT requires input: -i=inputImage -b=labeledImage -s=classifier.yaml -prob=decimal(range 0-1) -size=integer(range 0-20)" << endl;
		}else{
			Mat inputImage = imread(parser.get<string>("i"));
			Mat labels = imread(parser.get<string>("b"),0);
			Mat response = predictSVM(inputImage,parser.get<string>("s"));
			Mat filt;
			bilateralFilter(response.clone(),filt,30,50,50);
			Mat r_thresh,l_thresh;
			int val = parser.get<float>("prob")*255;
			threshold(filt,r_thresh,val,255,THRESH_BINARY);
			threshold(labels,l_thresh,25,255,THRESH_BINARY);
			confusionGUI(inputImage, 255-r_thresh, l_thresh, parser.get<int>("size"));
		}
	}
	else if(bool_charucoCreate){
		if(!(parser.has("d") && parser.has("nx") && parser.has("ny") && parser.has("aw") && parser.has("mw"))){
			cout << "Using mode CHARUCO_CREATE requires input: -d=dictionaryID -nx=num_board_spacesX -ny=num_board_spacesY -mw=marker_width -aw=aruco_width" << endl;
		}else{
			Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(parser.get<int>("d")));
			Ptr<cv::aruco::CharucoBoard> board = cv::aruco::CharucoBoard::create(parser.get<int>("nx"), parser.get<int>("ny"), parser.get<float>("mw"), parser.get<float>("aw"), dictionary);
			Mat boardImage;
			board->draw( Size(2160,2160), boardImage );
			string outname = parser.get<string>("d")+"-"+parser.get<string>("nx")+"x"+parser.get<string>("ny")+"_"+parser.get<string>("mw")+"_"+parser.get<string>("aw")+".png";
			imwrite(outname,boardImage);
		}
	}
	else if(bool_charucoCalib){
		if(!(parser.has("ci") && parser.has("d") && parser.has("cc") && parser.has("nx") && parser.has("ny") && parser.has("aw") && parser.has("mw"))){
			cout << "Using mode CHARUCO_CALIB requires input: -ci=calib_img_paths.txt -d=dictionaryID -cc=camera_calibration_outfile.yaml -nx=num_board_spacesX -ny=num_board_spacesX -mw=marker_width -aw=aruco_width" << endl;
		}else{
		    bool check = charuco_calibrate(parser.get<string>("cc"),parser.get<string>("ci"),parser.get<int>("d"),parser.get<int>("nx"),parser.get<int>("ny"),parser.get<float>("mw"),parser.get<float>("aw"));
		}
	}
	else if(bool_charuco_est){
		if(!(parser.has("i") && parser.has("d") &&parser.has("cc") && parser.has("nx") && parser.has("ny") && parser.has("aw") && parser.has("mw"))){
			cout << "Using mode CHARUCO_EST requires input: -i=inputImage -d=dictionaryID -cc=camera_calibration_infile.yaml -nx=num_board_spacesX -ny=num_board_spacesX -mw=marker_width -aw=aruco_width" << endl;
		}else{
			//-- Getting camera calibration details
			Mat cameraMatrix, distCoeffs;
			FileStorage fs;
			fs.open(parser.get<string>("cc"), FileStorage::READ);
			fs["camera_matrix"] >> cameraMatrix;
			fs["distortion_coefficients"] >> distCoeffs;

			//-- Getting input image and board image
			Mat inputImage = imread(parser.get<string>("i"));
			Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(parser.get<int>("d")));
			Ptr<cv::aruco::CharucoBoard> board = cv::aruco::CharucoBoard::create(parser.get<int>("nx"), parser.get<int>("ny"), parser.get<float>("mw"), parser.get<float>("aw"), dictionary);
			Mat boardImage;
			board->draw( inputImage.size(), boardImage );

			vector< cv::Point2f > charucoCorners, detectedCharucoCorners, matchedCharucoCorners;
			vector< int > charucoIds, detectedCharucoIds, matchedCharucoIds;

			//-- Detecting input image board
			vector< int > ids;
			vector< vector< Point2f > > corners;
			aruco::detectMarkers(inputImage, dictionary, corners, ids);
			if(ids.size() > 0){
				aruco::interpolateCornersCharuco(corners, ids, inputImage, board, detectedCharucoCorners, detectedCharucoIds);
			}

			//-- Detecting perfect board image
			vector< int > markerIds;
			vector< vector< Point2f > > markerCorners;
			aruco::detectMarkers(boardImage, dictionary, markerCorners, markerIds);
			aruco::interpolateCornersCharuco(markerCorners, markerIds, boardImage, board, charucoCorners, charucoIds);

			Vec3d rvec, tvec;

			if (detectedCharucoIds.size() > 0) {
			//-- Matching input board to perfect board
						for (unsigned int i = 0; i < charucoIds.size(); i++) {
							for (unsigned int j = 0; j < detectedCharucoIds.size(); j++) {
								if (charucoIds[i] == detectedCharucoIds[j]) {
									matchedCharucoIds.push_back(charucoIds[i]);
									matchedCharucoCorners.push_back(charucoCorners[i]);
								}
							}
						}
			//-- Computing spatial homography and warping
						Mat perspectiveTransform = findHomography(detectedCharucoCorners, matchedCharucoCorners, cv::RANSAC);
						Mat undistoredCharuco;
						warpPerspective(inputImage, undistoredCharuco, perspectiveTransform, inputImage.size());
						bool valid = aruco::estimatePoseCharucoBoard(detectedCharucoCorners, detectedCharucoIds, board, cameraMatrix, distCoeffs, rvec, tvec);

						if(valid){
							cv::aruco::drawAxis(inputImage, cameraMatrix, distCoeffs, rvec, tvec, 0.1);
						}

						vector<string> sub_str;
						const string full_str = parser.get<string>("i");
						char del = '.';

						split(full_str,del,sub_str);
						string new_name;
						new_name= sub_str[0]+"_posed.png";
						imwrite(new_name,inputImage);

						split(full_str,del,sub_str);
						new_name = sub_str[0]+"_opp.png";
						imwrite(new_name,undistoredCharuco);
			}
		}
	}
	else if(bool_vis | bool_vis_CH){
		if(bool_vis && !(parser.has("i") && parser.has("b") && parser.has("s") && parser.has("c") && parser.has("d"))){
			cout << "Using mode VIS requires input: -i=inputImage -b=backgroundImage -s=shapes_file.txt -c=color_file.txt -d=leaf_file.txt" << endl;
		}
		else if(bool_vis_CH && !(parser.has("i") && parser.has("b") && parser.has("s") && parser.has("c") && parser.has("d"))){
			cout << "Using mode VIS_CH requires input: -i=inputImage -b=backgroundImage -s=shapes_file.txt -c=color_file.txt -d=leaf_file.txt" << endl;
			cout << "In addition to this input, a directory called 'card_masks' must be present and contains binary images of each chip of the input image" << endl;
			cout << "and a CSV called 'target_homography.csv' must be present. This is obtained using the SET_TARGET mode of this program." << endl;
		}else{
			Mat inputImage = imread(parser.get<string>("i"));
			Mat adjBackground = imread(parser.get<string>("b"));

			//-- Processing the VIS image
			Mat adjImage;
			float det=0;
			float D;
			//-- Color homography
			if(bool_vis_CH){
				MatrixXd rh, gh, bh;
				get_standardizations(inputImage, det, rh, gh, bh);
				adjImage = color_homography(inputImage,rh,gh,bh);
				D = 1-det;
			}else{
				adjImage = inputImage;
			}

			//-- Difference in images
			Mat dest;
			absdiff(adjBackground,adjImage,dest);
			vector<Mat> channels(3);
			split(dest,channels);
			Mat dest_blur;
			blur(channels[1], dest_blur, Size( 2, 2 ) );
			Mat dest_thresh;
			threshold(dest_blur,dest_thresh,25,255,THRESH_BINARY);
			Mat dest_dilate;
			dilate(dest_thresh, dest_dilate, Mat(), Point(-1, -1), 5, 1, 1);
			Mat dest_erode;
			erode(dest_dilate,dest_erode, Mat(), Point(-1, -1), 4, 1, 1);

			//-- Removing barcode
			Mat lab;
			cvtColor(adjImage, lab, cv::COLOR_BGR2Lab);
			vector<Mat> split_lab;
			split(lab, split_lab);
			Mat b_thresh1;
			inRange(split_lab[2],90,139,b_thresh1);
			Mat invSrc =  cv::Scalar::all(255) - b_thresh1;
			Mat mask1;
			bitwise_and(dest_erode,invSrc,mask1);
			Mat barcode_roi;
			vector<Point> cc_barcode = keep_roi(mask1,Point(1146,1368),Point(1359,1479),barcode_roi);
			Mat mask2 = mask1-barcode_roi;

			//-- Remove edges of pot
			Mat dest_lab;
			cvtColor(dest, dest_lab, cv::COLOR_BGR2Lab);
			vector<Mat> channels_lab;
			split(dest_lab, channels_lab);
			Mat pot_thresh1;
			inRange(channels_lab[2],0,120,pot_thresh1);
			Mat pot_thresh2;
			inRange(channels_lab[2],135,200,pot_thresh2);
			Mat pot_or;
			bitwise_or(pot_thresh1,pot_thresh2,pot_or);
			Mat pot_dilate;
			dilate(pot_or, pot_dilate, Mat(), Point(-1, -1), 2, 1, 1);
			Mat pot_erode;
			erode(pot_dilate,pot_erode, Mat(), Point(-1, -1), 3, 1, 1);
			Mat pot_and;
			bitwise_and(pot_erode,mask2,pot_and);
			Mat pot_roi;
			vector<Point> cc_pot = keep_roi(pot_and,Point(300,600),Point(1610,1310),pot_roi);

			//-- Remove blue stakes
			Mat b_thresh;
			inRange(split_lab[2],80,115,b_thresh);
			Mat b_er;
			erode(b_thresh,b_er, Mat(), Point(-1, -1), 1, 1, 1);
			Mat b_roi;
			vector<Point> cc1 = keep_roi(b_er,Point(300,600),Point(1610,1310),b_roi);
			Mat b_dil;
			dilate(b_roi,b_dil,Mat(),Point(-1, -1), 6, 1, 1);
			Mat b_xor = pot_roi - b_dil;

			//-- ROI selector
			Mat mask;
			vector<Point> cc = keep_roi(b_xor,Point(550,0),Point(1810,1410),mask);

			//-- Segmenting leaves from stem
			Mat dil;
			dilate(mask, dil, Mat(), Point(-1, -1), 1, 1, 1);
			Mat skel;
			ximgproc::thinning(dil,skel,THINNING_ZHANGSUEN);
			Mat skel_filt0 = length_filter(skel,50);
			Mat pruned = prune(skel_filt0,5);
			Mat seg_skel = segment_skeleton(pruned);
			Mat tips = find_endpoints(pruned);
		    Mat no_tips = Mat::zeros(inputImage.size(),pruned.type());
		    rectangle(no_tips,Point(1164,1266),Point(1290,1407),255,cv::FILLED);
		    tips = tips -(tips & no_tips);

			Mat skel_filt1 = length_filter(seg_skel,12);
			Mat leaves = find_leaves(skel_filt1,tips);
			Mat classified = add_stem(leaves,pruned);
			Mat filled_mask = fill_mask(dil,classified);

			//-- Getting numerical data
			vector<double> shapes_data = get_shapes(cc,mask);
			Mat hue_data = get_color(adjImage, mask);
			vector<vector<double> > leaf_data = get_leaf_info(classified,filled_mask);

			//-- Write shapes to file
			string name_shape= parser.get<string>("s");
			ofstream shape_file;
			shape_file.open(name_shape.c_str(),ios_base::app);
			shape_file << parser.get<string>("i") << " ";
			for(int i=0;i<20;i++){
				shape_file << shapes_data[i];
				if(i != 19){
					shape_file << " ";
				}
			}
			if(bool_vis_CH){
				shape_file << " " << D;
			}
			shape_file << endl;
			shape_file.close();

			//-- Write color to file
			string name_hue= parser.get<string>("c");
			ofstream hue_file;
			hue_file.open(name_hue.c_str(),ios_base::app);
			hue_file << parser.get<string>("i") << " ";
			for(int i=0;i<180;i++){
				hue_file << hue_data.at<float>(i,0) << " ";
			}
			hue_file << endl;
			hue_file.close();

			//-- Write leaf data to file
			string name_leaf= parser.get<string>("d");
			ofstream leaf_file;
			leaf_file.open(name_leaf.c_str(),ios_base::app);
			for(unsigned int i = 0; i<leaf_data[0].size(); i++){
				leaf_file << parser.get<string>("i") << " " << i << " " << leaf_data[0][i] << " " << leaf_data[1][i] << " " << leaf_data[2][i] << " " << leaf_data[3][i] << endl;
			}
			leaf_file.close();

			if(parser.has("debug")){
				vector<string> sub_str;
				const string full_str = string(parser.get<string>("i"));
				char del = '.';
				split(full_str,del,sub_str);
				string new_name = sub_str[0]+"_mask.png";
				imwrite(new_name,mask);
				new_name = sub_str[0]+"_filled.png";
				imwrite(new_name,filled_mask);
			}
		}
	}
	else if(bool_vis_CH_check){
		if(!parser.has("i")){
			cout << "Using mode VIS_CH_CHECK requires input: -i=inputImage" << endl;
			cout << "In addition to this input, a directory called 'card_masks' must be present and contains binary images of each chip of the input image" << endl;
			cout << "and a CSV called 'target_homography.csv' must be present. This is obtained using the SET_TARGET mode of this program." << endl;
		}else{
			Mat inputImage = imread(parser.get<string>("i"));

			//-- Processing the VIS image
			Mat adjImage;
			float det=0;

			//-- Color homography
			MatrixXd rh, gh, bh;
			get_standardizations(inputImage, det, rh, gh, bh);
			adjImage = color_homography(inputImage,rh,gh,bh);

			//-- Writing out adjusted image
			vector<string> sub_str;
			const string full_str = string(parser.get<string>("i"));
			char del = '.';
			split(full_str,del,sub_str);
			string new_name = sub_str[0]+"_corrected.jpg";
			imwrite(new_name,adjImage);
		}
	}
    //-- Processing the NIR image
	else if(bool_nir){
		if(!(parser.has("i") && parser.has("b") && parser.has("c"))){
			cout << "Using mode NIR requires input: -i=inputImage -b=backgroundImage -c=nir_color_file.txt" << endl;
		}else{
			//-- Read in image and background
			    	Mat nirImage = imread(parser.get<string>("i"),0);
			    	//Mat nir_fixed = 1.591*nirImage-31.803;

			    	Mat nirBackground = imread(parser.get<string>("b"),0);

			    	//-- Difference between image and background
					Mat dest_nir;
					absdiff(nirBackground,nirImage,dest_nir);
					Mat dest_nir_thresh;
					inRange(dest_nir,20,255,dest_nir_thresh);

					//-- Remove white stake
					Mat dest_stake;
					inRange(dest_nir,60,255,dest_stake);
					Mat dest_stake_dil;
					dilate(dest_stake, dest_stake_dil, Mat(), Point(-1, -1), 2, 1, 1);
					Mat kept_stake;
			    	vector<Point> cc = keep_roi(dest_stake_dil,Point(270,183),Point(350,375),kept_stake);
			    	Mat dest_sub = dest_nir_thresh - kept_stake;

			        //-- ROI selector
			    	Mat kept_mask_nir;
			    	cc = keep_roi(dest_sub,Point(171,102),Point(470,363),kept_mask_nir);

			        //-- Getting numerical data
			    	Mat nir_data = get_nir(nirImage, kept_mask_nir);

			        //-- Writing numerical data
			    	string name_nir= parser.get<string>("c");
			   		ofstream nir_file;
			   		nir_file.open(name_nir.c_str(),ios_base::app);
			   		nir_file << argv[2] << " ";
			   		for(int i=0;i<255;i++){
			   		   	nir_file << nir_data.at<float>(i,0) << " ";
			   		}
			   		nir_file << endl;
			   		nir_file.close();

			   		if(parser.has("debug")){
						vector<string> sub_str;
						const string full_str = string(parser.get<string>("i"));
						char del = '.';
						split(full_str,del,sub_str);
						string new_name = sub_str[0]+"_mask.png";
						imwrite(new_name,kept_mask_nir);
					}
		}
	}
	else if(bool_getCH){
		if(!parser.has("i")){
			cout << "Using mode SET_TARGET requires input: -i=inputImage" << endl;
			cout << "In addition to this input, a directory called 'card_masks' must be present and contains binary images of each chip of the input image" << endl;
			cout << "Redirect this output to 'target_homography.csv' for VIS_CH and VIS_CH_CHECK modes to work" << endl;
		}else{
			//-- Getting RGB components of each chip in the reference picture
					Mat inputImage = imread(parser.get<string>("i"));
					vector<Mat> bgr;
					split(inputImage, bgr);
					Mat b = bgr[0];
					Mat g = bgr[1];
					Mat r = bgr[2];
					for(unsigned int i=1;i<23;i++){
					    	stringstream ss;
					    	ss << i;
					    	string str = ss.str();
					    	string file_name = "card_masks/"+str+"_mask.png";
					    	Mat mask = imread(file_name,0);
					    	Mat cc;
					   		threshold(mask,cc,90,255,THRESH_BINARY);

					    	float b_avg = extractRGB_chips(b, cc);
					    	float g_avg = extractRGB_chips(g, cc);
					    	float r_avg = extractRGB_chips(r, cc);

					    	//-- Write histogram averages to cout
					    	cout  << r_avg << ","<< g_avg << "," << b_avg << endl;
					}
		}
    }
	else if(bool_drawROIS){
		if(!(parser.has("i") && parser.has("s"))){
			cout << "Using mode DRAW_ROIS requires input: -i=input_image -s=size" << endl;
			cout << "In addition to the input requirements, a directory called 'card_masks' must exist" << endl;
		}else{
			src = imread(parser.get<string>("i"));
			roi_size = parser.get<int>("s");

			namedWindow("Image",WINDOW_NORMAL);
			setMouseCallback("Image",onMouse,NULL );
			resizeWindow("Image",src.cols,src.rows);
			imshow("Image",src);
			waitKey(0);
		}
	}
	else if(bool_avgImgs){
		if(argc != 2){
			cout << "Using mode AVG_IMGS requires only that a list of images to be averaged is piped in" << endl;
		}else{
			 //-- Taking list of pictures that are piped in and averaging them
					string line;
					Mat avg;
					vector<Mat> avg_bgr(3);
					int counter = 0;
					while(cin) {
						if(getline(cin,line)) {
							if(counter == 0){
					    		avg=imread(line);
					    		avg.convertTo(avg, CV_64F);
					   			split(avg,avg_bgr);
					    		counter++;
					    	}else{
					        	Mat inputImage = imread(line);
					        	inputImage.convertTo(inputImage, CV_64F);
					    		vector<Mat> in_bgr(3);
				    			split(inputImage,in_bgr);
				    			avg_bgr[0] = (avg_bgr[0]+in_bgr[0]);
				    			avg_bgr[1] = (avg_bgr[1]+in_bgr[1]);
				    			avg_bgr[2] = (avg_bgr[2]+in_bgr[2]);
					        	counter++;
					    	}
					    }
					}
					avg_bgr[0] = (avg_bgr[0])/counter;
					avg_bgr[1] = (avg_bgr[1])/counter;
					avg_bgr[2] = (avg_bgr[2])/counter;
					Mat adjImage;
					merge(avg_bgr,adjImage);
					adjImage.convertTo(adjImage, CV_64F);

					//-- Writing out averaged image
					imwrite("average_images.png",adjImage);
		}
	}
	else if(parser.has("h")){
		cout << "DESCRIPTION:" << endl << "\tThis program is for segmenting and measuring plants from the Bellweather Phenotyping Facility. Segmentation is achieved by supplying a background image that does not contain a plant and using the difference between that and a supplied image to threshold on. Further processing is done to remove artifacts that arise. After segmentation is complete, shapes and color profile are reported in corresponding user-specified files." << endl << endl;
		cout << "USAGE:" << endl << "\tThere are fourteen modes of use (VIS, VIS_CH, VIS_CH_CHECK, NIR, SET_TARGET, DRAW_ROIS, CHARUCO_CREATE, CHARUCO_CALIB, CHARUCO_EST, BC_CREATE, BC_PRED, SVM_CREATE, SVM_PRED, and AVG_IMGS). Depending on what is chosen, the required inputs change" << endl << endl;
		cout << "SYNOPSIS:" << endl << "\t./PhenotyperCV -m=[MODE] [INPUTS]" << endl << endl;
		cout << "MODES:"<< endl;
		cout << "\t\e[1mVIS\e[0m - Segment and measure plant in RGB images" << endl << "\t\t" << "Example: ./PhenotyperCV -m=VIS -i=input_image.png -b=background_image.png -s=shapes.txt -c=color.txt -d=leaves.txt"<<endl << endl;
		cout << "\t\e[1mVIS_CH\e[0m - standardize, segment, and measure plant in RGB images" << endl << "\t\t" << "Example: ./PhenotyperCV -m=VIS_CH -i=input_image.png -b=background_image.png -s=shapes.txt -c=color.txt -d=leaves.txt" << endl << "NOTE Processing using the VIS_CH mode requires two additional items: a card_masks/ folder that contains masks for each of the chips and target_homography.csv file that is the desired color space. The csv file can be created for you using the SET_TARGET mode of this program and redirecting the output." << endl << endl;
		cout << "\t\e[1mVIS_CH_CHECK\e[0m - standardize, and output image" << endl << "\t\t" << "Example: ./PhenotyperCV -m=VIS_CH_CHECK -i=input_image.png" << endl << "NOTE Processing using the VIS_CH_CHECK mode requires two additional items: a card_masks/ folder that contains masks for each of the chips and target_homography.csv file that is the desired color space. The csv file can be created for you using the SET_TARGET mode of this program and redirecting the output." << endl << endl;
		cout << "\t\e[1mNIR\e[0m - segment and measure plant in near-infrared images" << endl << "\t\t" << "Example: ./PhenotyperCV -m=NIR -i=input_image.png -b=background_image.png -c=nir_color.txt" << endl << endl;
		cout << "\t\e[1mSET_TARGET\e[0m - obtain and print to stdout the RGB information for each of the chips in the image" << endl << "\t\t" << "Example: ./PhenotyperCV -m=SET_TARGET -i=targetImage.png > target_homography.csv" << endl << "NOTE Processing using the SET_TARGET mode requires a card_masks/ folder that contains masks for each of the chips" << endl << endl;
		cout << "\t\e[1mDRAW_ROIS\e[0m - This is a GUI that makes the card_masks/ images to be used by VIS_CH, VIS_CH_CHECK, and SET_TARGET. When you click, an roi is drawn onto the input image but a new binary image is created as well. The input 'size' is half the length of the desired square roi. 8 is a good choice for the Bellweather Phenotyper. The directory card_masks must already be made for the images to save" << endl << "\t\t" << "Example: ./PhenotyperCV -m=DRAW_ROIS -i=input_image.png -s=size" << endl << endl;
		cout << "\t\e[1mCHARUCO_CREATE\e[0m - Creates a nx by ny ChArUco board with mw marker width and aw aruco chip width using d dictionary." << endl << "\t\t" << "Example: ./PhenotyperCV -m=CHARUCO_CREATE -d=10 -nx=5 -ny=7 -mw=0.04 -aw=0.02" << endl << endl;
		cout << "\t\e[1mCHARUCO_CALIB\e[0m - Camera calibration using multiple viewpoints of a ChArUco board. It is recommended to take enough pictures where combined the entire scene had been accounted for by the board. The images are passed into the program by means of a file of 1 column where each row is the path to each image. One output file called camera_calibration.txt is produced when running this method" << endl << "\t\t" << "Example: ./PhenotyperCV -m=CHARUCO_CALIB -ci=calib_list.txt -d=10 -nx=5 -ny=7 -mw=0.04 -aw=0.02 -cc=camera_calibration.yaml" << endl << endl;
		cout << "\t\e[1mCHARUCO_EST\e[0m - After calibrating the camera using only CHARUCO_CALIB, this mode takes an image with the same board in the scene and warps the image to the orthogonal plane projection. The camera_calibration.txt file must be in the current working directory to be read in correctly. Two images are produced: 1) The same input image but with the pose of the board overlaid, and 2) is the orthogonal plane projection." << endl << "\t\t" << "Example: ./PhenotyperCV -m=CHARUCO_EST -i=test_imgs/plate.jpg -d=10 -nx=5 -ny=7 -mw=0.04 -aw=0.02 -cc=camera_calibration.yaml" << endl << endl;
		cout << "\t\e[1mBC_CREATE\e[0m - Creates and outputs a naive bayes classifier that is trained on a RGB image and it's respective labeled image." << endl << "\t\t" << "Example: ./PhenotyperCV -m=BC_CREATE -i=input_image.png -b=labeled_image.png -s=bayes_classifier.yaml" << endl << endl;
		cout << "\t\e[1mBC_PRED\e[0m - Classifies an input image by using a pre-trained bayes classifier to identify features in the image." << endl << "\t\t" << "Example: ./PhenotyperCV -m=BC_PRED -i=input_image.png -s=bayes_classifier.yaml" << endl << endl;
		cout << "\t\e[1mSVM_CREATE\e[0m - Creates and outputs a support vector machine classifier that is trained on a RGB image and it's respective labeled image." << endl << "\t\t" << "Example: ./PhenotyperCV -m=SVM_CREATE -i=input_image.png -b=labeled_image.png -s=bayes_classifier.yaml" << endl << endl;
		cout << "\t\e[1mSVM_PRED\e[0m - Classifies an input image by using a pre-trained SVM classifier to identify features in the image." << endl << "\t\t" << "Example: ./PhenotyperCV -m=SVM_PRED -i=input_image.png -s=svm_classifier.yaml" << endl << endl;
		cout << "\t\e[1mAVG_IMGS\e[0m - takes list of input images to be averaged and outputs average_images.png" << endl << "\t\t" << "Example: cat Images/SnapshotInfo.csv | grep Fm000Z | grep VIS_SV | awk -F'[;,]' '{print \"Images/snapshot\"$2\"/\"$12\".png\"}' | ./PhenotyperCV -m=AVG_IMGS"<< endl << endl << endl;
		cout << "PIPELINES:" << endl;
		cout << "\tColor Correction VIS Pipeline:" << endl;
		cout << "\t\t* Average together all the empty pots using AVG_IMGS" << endl;
		cout << "\t\t* Create card_masks directory and then use DRAW_ROIS" << endl;
		cout << "\t\t* Set the average_image.png to target using SET_TARGET" << endl;
		cout << "\t\t* Run analysis with average_images.png as background using VIS_CH" << endl << endl;
		cout << "\tNormal VIS Pipeline:" << endl;
		cout << "\t\t* Average together all the empty pots using AVG_IMGS" << endl;
		cout << "\t\t* Run analysis with average_images.png as background using VIS" << endl << endl;
		cout << "\tNIR Pipeline:" << endl;
		cout << "\t\t* Average together all the empty pots using AVG_IMGS" << endl;
		cout << "\t\t* Run analysis with average_images.png as background using NIR" << endl << endl;
	}
	else{
    	cout << "Mode must be either VIS, VIS_CH, VIS_CH_CHECK, NIR, SET_TARGET, DRAW_ROIS,CHARUCO_CREATE, CHARUCO_CALIB, CHARUCO_EST, BC_CREATE, BC_PRED, SVM_CREATE, SVM_PRED or AVG_IMGS" << endl;
    	cout << "Use  ./PhenotyperCV -h for more information" << endl;
    }

	return 0;
}


/*
namedWindow("Image",WINDOW_NORMAL);
        	    resizeWindow("Image",800,800);
        	    imshow("Image", b_blur);
waitKey(0);
*/
