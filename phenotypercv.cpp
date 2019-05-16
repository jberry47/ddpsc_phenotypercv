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
 * Compiling Notes:
 * -I/usr/local/include/opencv -I/usr/local/include/opencv2 -I/usr/include/Eigen
 * -L/usr/local/lib lopencv_core -lopencv_features2d -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs
 *
 * On Danforth Center Bioinformatics Infrastructure:
 * g++ -I/shares/bioinfo/installs/opencv-3.1.0/install/include -I/shares/bioinfo/installs/eigen/Eigen -L/shares/bioinfo/installs/opencv-3.1.0/install/lib -lopencv_imgproc -lopencv_imgcodecs -lopencv_core phenotypercv.cpp -o PhenotyperCV
 *
 * To use the program in a parallelized fashion, pipe find into xargs with the -P flag followed by
 * number of cores like this:
 * 	Usage VIS or VIS_CH:
 * 		find Images/ -name 'VIS_SV*' | xargs -P8 -I{} ./PhenotyperCV VIS_CH {} background_image.png shapes.txt color.txt'
 * 	Usage NIR:
 * 		find Images/ -name 'NIR_SV*' | xargs -P8 -I{} ./PhenotyperCV NIR {} background_image.png nir_color.txt
 *
 */
#include <opencv2/opencv.hpp>
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

namespace {
const char* keys  =
        "{m        |       | Mode to run }"
        "{h        |       | Show help documentation }"
        "{i        |       | Input image }"
        "{b        |       | Background image }"
        "{size     |       | Square size (pixels) for DRAW_ROI mode}"
        "{s        |       | Shape file to write to }"
        "{c        |       | Color file to write to }"
		"{ci       |       | ChArUco calibrate input file }"
		"{cc       |       | Camera calibration file name }"
		"{nx       |       | Number of board spaces - x }"
		"{ny       |       | Number of board spaces - y }"
		"{mw       |       | Marker width }"
		"{aw       |       | ArUco width }"
		"{debug    |       | If used, write out final mask }"
		"{d        |       | dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,"
		        "DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, "
		        "DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12,"
		        "DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16}"
		        "{@outfile |<none> | Output file with calibrated camera parameters }";
}

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

int is_oof(Mat img){
	//-- Get contours of mask
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
    findContours( img, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );

    //-- Get contours of rectangular roi
    Mat src = Mat::zeros(img.size(),img.type())+255;

    vector<vector<Point> > contours_roi;
    vector<Vec4i> hierarchy_roi;
    findContours( src, contours_roi, hierarchy_roi, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );

    int check = 0;
    //-- Keep only those contours that have a point inside roi
    for(unsigned int i=0; i < contours.size(); i++){
      	for(unsigned int j=0; j<contours[i].size(); j++){
      		int test = pointPolygonTest(contours_roi[0],Point2f(contours[i][j]),false);
      		if(test == 0){
      			check = 1;
      		}
       	}
    }
	return check;
}

vector<Point> keep_roi(Mat img,Point tl, Point br, Mat &mask){
	//-- Get contours of mask
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
    findContours( img, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );

    //-- Get contours of rectangular roi
    Mat src = Mat::zeros(img.size(),img.type());
    rectangle(src,tl,br,255,cv::FILLED);

    vector<vector<Point> > contours_roi;
    vector<Vec4i> hierarchy_roi;
    findContours( src, contours_roi, hierarchy_roi, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );

    //-- Keep only those contours that have a point inside roi
    vector<Point> cc;
    Mat kept = Mat::zeros(img.size(),img.type());
    for(unsigned int i=0; i < contours.size(); i++){
      	for(unsigned int j=0; j<contours[i].size(); j++){
      		int test = pointPolygonTest(contours_roi[0],Point2f(contours[i][j]),false);
      		if(test==1 || test == 0){
      			for(unsigned int k=0; k<contours[i].size(); k++){
      				cc.push_back(contours[i][k]);
      			}
      			drawContours(kept, contours, i, 255, cv::FILLED);
      			break;
      		}
       	}
    }
	Mat kept_mask;
	bitwise_and(img,kept,kept_mask);

    mask = kept_mask;
	return cc;
}

float get_fd(Mat mask){
	//-- Need to remap the image to 2048x2048 so box counting can be used
	Mat img_bc;
	resize(mask, img_bc, Size(2048,2048), 0, 0, INTER_LINEAR);

	//-- Initializing variables
	double width = 2048.0;
	double p = log(width)/log(double(2.0));
	VectorXf N = VectorXf::Zero(int(p)+1);
	double sumImg = sum(img_bc)[0];
	N(int(p)) = sumImg;

	//-- Boxcounting
	double siz;
	double siz2;
	float running_sum;
    for (int g = int(p)-1; g > 0; g--){
    	siz = pow(2.0, double(p-g));
    	siz2 = round(siz/2.0);
    	running_sum = 0;
    	for (int i = 0; i < int(width-siz+1); i = i+int(siz)){
    		for (int j = 0; j < int(width-siz+1); j = j+int(siz)){
    			img_bc.at<uchar>(i,j) = (bool(img_bc.at<uchar>(i,j)) || bool(img_bc.at<uchar>(i+siz2,j))
    				|| bool(img_bc.at<uchar>(i,j+siz2)) || bool(img_bc.at<uchar>(i+siz2,j+siz2)));
    			running_sum = running_sum+float(img_bc.at<uchar>(i,j));
    		}
    	}
    	N(g) = running_sum;
	}
    N = N.colwise().reverse().eval();

    //-- Getting bin sizes
    VectorXf R = VectorXf::Zero(int(p)+1);
    R(0) = 1.0;
    for (int k = 1; k < R.size(); k++){
    	R(k) = pow(2.0, double(k));
    }

    //-- Calculating log-log slopes
	float slope [R.size()-1];
	for(int i=1;i < R.size()-1 ;i++){
		slope[i] = (log10(N(i+1))-log10(N(i)))/(log10(R(i+1))-log10(R(i)));
	}

	//-- Getting average slope (fractal dimension)
	float sum = 0.0, average;
	int s_count =0;
	for(int i=1; i < R.size()-1; i++){
		if(-slope[i] < 2 && -slope[i] > 0){
			sum += -slope[i];
			s_count++;
		}
	}
	average = sum / s_count;
	return average;
}

vector<double> get_shapes(vector<Point> cc,Mat mask){
    //-- Get measurements
    Moments mom = moments(mask,true);
    double area = mom.m00;
	if(area>0){
	    vector<Point>hull;
	    convexHull( Mat(cc), hull, false );
	    double hull_verticies = hull.size();
	    double hull_area = contourArea(Mat(hull));
	    double solidity = area/hull_area;
	    double perimeter = arcLength(Mat(cc),false);
	    double cmx = mom.m10 / mom.m00;
	    double cmy = mom.m01 / mom.m00;
	    Rect boundRect = boundingRect( cc );
	    double width = boundRect.width;
	    double height = boundRect.height;
	    double circ = 4*M_PI*area/(perimeter*perimeter);
	    double angle = -1;
	    double ex = -1;
	    double ey = -1;
	    double emajor = -1;
	    double eminor = -1;
	    double eccen = -1;
	    double round = -1;
	    double ar = -1;
	    if(cc.size() >= 6){
	        Mat pointsf;
	    	Mat(cc).convertTo(pointsf, CV_32F);
	   	    RotatedRect ellipse = fitEllipse(pointsf);
	   	    angle = ellipse.angle;
	  	    ex = ellipse.center.x;
	   	    ey = ellipse.center.y;
	   	    if(ellipse.size.height > ellipse.size.width){
	   	    	emajor = ellipse.size.height;
	   	    	eminor = ellipse.size.width;
	   	    }else{
	   	    	eminor = ellipse.size.height;
	  	   	    emajor = ellipse.size.width;
	   	    }
	   	    eccen = sqrt((1- eminor / emajor)*2);
	   	    round = eminor/emajor;
	   	    ar = emajor/eminor;
	    }
	    float fd = get_fd(mask);
	    double oof = is_oof(mask);
	    double shapes[20] = {area,hull_area,solidity,perimeter,width,height,cmx,cmy,hull_verticies,ex,ey,emajor,eminor,angle,eccen,circ,round,ar,fd,oof};
	    vector<double> shapes_v(shapes,shapes+20);
	    return shapes_v;
	}else{
	    double shapes[20] = {0, 0, -nan("1"), 0, 0, 0, -nan("1"), -nan("1"), 0, -1, -1, -1, -1, -1, -1, -nan("1"), -1, -1, -nan("1"), 0};
	    vector<double> shapes_v(shapes,shapes+20);
	    return shapes_v;
	}
}

Mat get_color(Mat img,Mat mask){
	Mat composite;
	cvtColor(img,composite,COLOR_BGR2HSV);
    vector<Mat> channels1;
    split(composite, channels1);
    Mat hist;
	int dims = 1; // Only 1 channel, the hue channel
	int histSize = 180; // 180 bins, actual range is 0-360.
	float hranges[] = { 0, 180 }; // hue varies from 0 to 179, see cvtColor
	const float *ranges = {hranges};

	//-- Compute the histogram
	calcHist(&channels1[0],1,0,mask,hist, dims, &histSize, &ranges	,true ,false);
	return hist;
}

Mat get_nir(Mat img,Mat mask){
    Mat hist;
	int dims = 1;
	int histSize = 255;
	float hranges[] = { 0, 255 };
	const float *ranges = {hranges};

	//-- Compute the histogram
	calcHist(&img,1,0,mask,hist, dims, &histSize, &ranges	,true ,false);
	return hist;
}

float extractRGB_chips(Mat img,Mat &mask){
	//-- Averages the histogram for a given channel
	Mat img1 = img;
    Mat hist;
	int dims = 1;
	int histSize = 255;
	float hranges[] = { 0, 255 };
	const float *ranges = {hranges};

	calcHist(&img1,1,0,mask,hist, dims, &histSize, &ranges ,true ,false);

	int sum=0;
	for(int i = 0;i<255;i++){
		sum += hist.at<float>(i,0);
	}
	Mat weights = hist/sum;
	float hist_avg=0.0;
	for(int i = 0;i<255;i++){
		hist_avg += i*weights.at<float>(i,0);
	}
	return hist_avg;
}


MatrixXd getRGBarray(Mat img){
	//-- Loops over chips and gets RGB values of each one
	MatrixXd sourceColors(22,3);
	vector<Mat> bgr;
	split(img, bgr);
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
	   	    sourceColors(i-1,0) = b_avg;
	   	    sourceColors(i-1,1) = g_avg;
	   	    sourceColors(i-1,2) = r_avg;
	}
    return(sourceColors);
}

void get_standardizations(Mat img, float &det, MatrixXd &rh,MatrixXd &gh,MatrixXd &bh){
	//-- Extending source RGB chips to squared and cubic terms
	MatrixXd source1, source2, source3;
	source1 = getRGBarray(img);
	source2 = (source1.array() * source1.array()).matrix();
	source3 = (source2.array() * source1.array()).matrix();
	MatrixXd source(source1.rows(),source1.cols()+source2.cols()+source3.cols());
	source << source1, source2, source3;

	//-- Computing Moore-Penrose Inverse
	MatrixXd M = (source.transpose()*source).inverse()*source.transpose();

	//-- Reading target homography
	MatrixXd target(22,3);
	fstream file;
	file.open("target_homography.csv");
	string value;
	int rowCounter = 0;
	while ( getline ( file, value) )
	{
	     vector<float> result;
	     stringstream substr(value);
	     string item;
	     while (getline(substr, item, ',')) {
		     const char *cstr = item.c_str();
		     char* pend;
		     float num = strtof(cstr,&pend);
	         result.push_back(num);
	     }
	     target(rowCounter,0) = result[0];
	     target(rowCounter,1) = result[1];
	     target(rowCounter,2) = result[2];
	     rowCounter++;
	}

	//-- Computing linear target RGB standardizations
	rh = M*target.col(2);
	gh = M*target.col(1);
	bh = M*target.col(0);

	//-- Extending target RGB chips to squared and cubic terms
	MatrixXd target1, target2, target3;
	target2 = (target.array() * target.array()).matrix();
	target3 = (target2.array() * target.array()).matrix();

	//-- Computing square and cubic target RGB standardizations
	MatrixXd r2h,g2h,b2h,r3h,g3h,b3h;
	r2h = M*target2.col(2);
	g2h = M*target2.col(1);
	b2h = M*target2.col(0);
	r3h = M*target3.col(2);
	g3h = M*target3.col(1);
	b3h = M*target3.col(0);

	//-- Computing D
	MatrixXd H(9,9);
	H << rh.col(0),gh.col(0),bh.col(0),r2h.col(0),g2h.col(0),b2h.col(0),r3h.col(0),g3h.col(0),b3h.col(0);
	det = H.transpose().determinant();
}

Mat color_homography(Mat img, MatrixXd r_coef,MatrixXd g_coef,MatrixXd b_coef){
	Mat b, g, r, b2, g2, r2, b3, g3, r3;
	vector<Mat> bgr(3);
	split(img,bgr);

	//-- Computing linear, squared, and cubed images
	b = bgr[0];
	g = bgr[1];
	r = bgr[2];
	b2 = b.mul(b);
	g2 = g.mul(g);
	r2 = r.mul(r);
	b3 = b2.mul(b);
	g3 = g2.mul(g);
	r3 = r2.mul(r);

	//-- Computing homography
	b = 0+r*b_coef(0,0)+g*b_coef(1,0)+b*b_coef(2,0)+r2*b_coef(3,0)+g2*b_coef(4,0)+b2*b_coef(5,0)+r3*b_coef(6,0)+g3*b_coef(7,0)+b3*b_coef(8,0);
	g = 0+r*g_coef(0,0)+g*g_coef(1,0)+b*g_coef(2,0)+r2*g_coef(3,0)+g2*g_coef(4,0)+b2*g_coef(5,0)+r3*g_coef(6,0)+g3*g_coef(7,0)+b3*g_coef(8,0);
	r = 0+r*r_coef(0,0)+g*r_coef(1,0)+b*r_coef(2,0)+r2*r_coef(3,0)+g2*r_coef(4,0)+b2*r_coef(5,0)+r3*r_coef(6,0)+g3*r_coef(7,0)+b3*r_coef(8,0);

	//-- Combining channels and returning
	bgr[0] = b;
	bgr[1] = g;
	bgr[2] = r;
	Mat adjImage;
	merge(bgr,adjImage);
	return adjImage;
}

void split(const string& s, char c, vector<string>& v) {
   string::size_type i = 0;
   string::size_type j = s.find(c);

   while (j != string::npos) {
      v.push_back(s.substr(i, j-i));
      i = ++j;
      j = s.find(c, j);

      if (j == string::npos)
         v.push_back(s.substr(i, s.length()));
   }
}

//-- Have to make global variables for DRAW_ROIS mode
int roi_size = 0;
Mat src;
int counter=1;

void onMouse( int event, int x, int y, int f, void* ){
    switch(event){
        case  cv::EVENT_LBUTTONDOWN  :
                                        Mat temp = Mat::zeros(src.size(),CV_8UC1);
                                        rectangle(temp,Point(x-roi_size,y-roi_size),Point(x+roi_size,y+roi_size),255,cv::FILLED);
                                        rectangle(src,Point(x-roi_size,y-roi_size),Point(x+roi_size,y+roi_size),Scalar( 255, 255, 255 ),cv::FILLED);
                                        imshow("Image",src);
                                        waitKey(1);
                                        stringstream ss;
                       				  	ss << counter;
                       				   	string str = ss.str();
                                        string file_name = "card_masks/"+str+"_mask.png";
                                        imwrite(file_name,temp);
                                        counter++;
                                        cout << "Made: " << file_name << endl;
                                        break;
    }
}

Mat skeletonize(Mat img){
	Mat mfblur;
	medianBlur(img, mfblur, 1);
	Mat skel(mfblur.size(), CV_8UC1, Scalar(0));
	Mat temp;
	Mat eroded;
	Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));
	bool done;
	int iterations=0;

	do
	{
	  erode(mfblur, eroded, element);
	  dilate(eroded, temp, element);
	  subtract(mfblur, temp, temp);
	  bitwise_or(skel, temp, skel);
	  eroded.copyTo(mfblur);

	  done = (countNonZero(mfblur) == 0);
	  iterations++;

	} while (!done && (iterations < 100));
	return skel;
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

	if(bool_charucoCreate){
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
		if(bool_vis && !(parser.has("i") && parser.has("b") && parser.has("s") && parser.has("c"))){
			cout << "Using mode VIS requires input: -i=inputImage -b=backgroundImage -s=shapes_file.txt -c=color_file.txt" << endl;
		}
		else if(bool_vis_CH && !(parser.has("i") && parser.has("b") && parser.has("s") && parser.has("c"))){
			cout << "Using mode VIS_CH requires input: -i=inputImage -b=backgroundImage -s=shapes_file.txt -c=color_file.txt" << endl;
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

			//-- Getting numerical data
			vector<double> shapes_data = get_shapes(cc,mask);
			Mat hue_data = get_color(adjImage, mask);

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

			if(parser.has("debug")){
				vector<string> sub_str;
				const string full_str = string(parser.get<string>("i"));
				char del = '.';
				split(full_str,del,sub_str);
				string new_name = sub_str[0]+"_mask.png";
				imwrite(new_name,mask);
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
			    	Mat nir_fixed = 1.591*nirImage-31.803;

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
		cout << "USAGE:" << endl << "\tThere are ten modes of use (VIS, VIS_CH, VIS_CH_CHECK, NIR, SET_TARGET, DRAW_ROIS, CHARUCO_CREATE, CHARUCO_CALIB, CHARUCO_EST, and AVG_IMGS). Depending on what is chosen, the required inputs change" << endl << endl;
		cout << "SYNOPSIS:" << endl << "\t./PhenotyperCV [MODE] [INPUTS]" << endl << endl;
		cout << "MODES:"<< endl;
		cout << "\t\e[1mVIS\e[0m - Segment and measure plant in RGB images" << endl << "\t\t" << "Example: ./PhenotyperCV -m=VIS -i=input_image.png -b=background_image.png -s=shapes.txt -c=color.txt"<<endl << endl;
		cout << "\t\e[1mVIS_CH\e[0m - standardize, segment, and measure plant in RGB images" << endl << "\t\t" << "Example: ./PhenotyperCV -m=VIS_CH -i=input_image.png -b=background_image.png -s=shapes.txt -c=color.txt" << endl << "NOTE Processing using the VIS_CH mode requires two additional items: a card_masks/ folder that contains masks for each of the chips and target_homography.csv file that is the desired color space. The csv file can be created for you using the SET_TARGET mode of this program and redirecting the output." << endl << endl;
		cout << "\t\e[1mVIS_CH_CHECK\e[0m - standardize, and output image" << endl << "\t\t" << "Example: ./PhenotyperCV -m=VIS_CH_CHECK -i=input_image.png" << endl << "NOTE Processing using the VIS_CH_CHECK mode requires two additional items: a card_masks/ folder that contains masks for each of the chips and target_homography.csv file that is the desired color space. The csv file can be created for you using the SET_TARGET mode of this program and redirecting the output." << endl << endl;
		cout << "\t\e[1mNIR\e[0m - segment and measure plant in near-infrared images" << endl << "\t\t" << "Example: ./PhenotyperCV -m=NIR -i=input_image.png -b=background_image.png -c=nir_color.txt" << endl << endl;
		cout << "\t\e[1mSET_TARGET\e[0m - obtain and print to stdout the RGB information for each of the chips in the image" << endl << "\t\t" << "Example: ./PhenotyperCV -m=SET_TARGET -i=targetImage.png > target_homography.csv" << endl << "NOTE Processing using the SET_TARGET mode requires a card_masks/ folder that contains masks for each of the chips" << endl << endl;
		cout << "\t\e[1mDRAW_ROIS\e[0m - This is a GUI that makes the card_masks/ images to be used by VIS_CH, VIS_CH_CHECK, and SET_TARGET. When you click, an roi is drawn onto the input image but a new binary image is created as well. The input 'size' is half the length of the desired square roi. 8 is a good choice for the Bellweather Phenotyper. The directory card_masks must already be made for the images to save" << endl << "\t\t" << "Example: ./PhenotyperCV -m=DRAW_ROIS -i=input_image.png -s=size" << endl << endl;
		cout << "\t\e[1mCHARUCO_CREATE\e[0m - Creates a nx by ny ChArUco board with mw marker width and aw aruco chip width using d dictionary." << endl << "\t\t" << "Example: ./PhenotyperCV -m=CHARUCO_CREATE -d=10 -nx=5 -ny=7 -mw=0.04 -aw=0.02" << endl << endl;
		cout << "\t\e[1mCHARUCO_CALIB\e[0m - Camera calibration using multiple viewpoints of a ChArUco board. It is recommended to take enough pictures where combined the entire scene had been accounted for by the board. The images are passed into the program by means of a file of 1 column where each row is the path to each image. One output file called camera_calibration.txt is produced when running this method" << endl << "\t\t" << "Example: ./PhenotyperCV -m=CHARUCO_CALIB -ci=calib_list.txt -d=10 -nx=5 -ny=7 -mw=0.04 -aw=0.02 -cc=camera_calibration.yaml" << endl << endl;
		cout << "\t\e[1mCHARUCO_EST\e[0m - After calibrating the camera using only CHARUCO_CALIB, this mode takes an image with the same board in the scene and warps the image to the orthogonal plane projection. The camera_calibration.txt file must be in the current working directory to be read in correctly. Two images are produced: 1) The same input image but with the pose of the board overlaid, and 2) is the orthogonal plane projection." << endl << "\t\t" << "Example: ./PhenotyperCV -m=CHARUCO_EST -i=test_imgs/plate.jpg -d=10 -nx=5 -ny=7 -mw=0.04 -aw=0.02 -cc=camera_calibration.yaml" << endl << endl;
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
    	cout << "Mode must be either VIS, VIS_CH, VIS_CH_CHECK, NIR, SET_TARGET, DRAW_ROIS,CHARUCO_CREATE, CHARUCO_CALIB, CHARUCO_EST, or AVG_IMGS" << endl;
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
