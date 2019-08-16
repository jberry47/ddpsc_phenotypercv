#include <opencv2/opencv.hpp>
#include "opencv2/ximgproc.hpp"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <math.h>
#include <Eigen/Dense>

#include <feature_extraction.h>

using namespace cv;
using namespace std;
using namespace Eigen;

int roi_size;
Mat src;
Mat selMat;
int counter=1;
Mat src1, dst, gray;

int threshold_value = 255;
int threshold_type = 0;
int max_value = 255;
int max_type = 4;
int max_BINARY_value = 255;

void showImage(Mat img, string title){
	namedWindow(title,WINDOW_NORMAL);
	        	    resizeWindow(title,800,800);
	        	    imshow(title, img);
	while(true){
		int c;
	    c = waitKey();
	    if( (char)c == 27 ){
	    	break;
	    }
	}
}

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

void kMouse( int event, int x, int y, int f, void* ){
	Scalar color;
	switch(event){
        case  cv::EVENT_LBUTTONDOWN  :
        	color = Scalar( 0, 0, 255 );
            rectangle(selMat,Point(x-roi_size,y-roi_size),Point(x+roi_size,y+roi_size),color,cv::FILLED);
            rectangle(src,Point(x-roi_size,y-roi_size),Point(x+roi_size,y+roi_size),color,cv::FILLED);
            imshow("Select spots",src);
            waitKey(1);
            break;
        case cv::EVENT_RBUTTONDOWN   :
        	color = Scalar( 0, 255, 0 );
            rectangle(selMat,Point(x-roi_size,y-roi_size),Point(x+roi_size,y+roi_size),color,cv::FILLED);
            rectangle(src,Point(x-roi_size,y-roi_size),Point(x+roi_size,y+roi_size),color,cv::FILLED);
            imshow("Select spots",src);
            waitKey(1);
            break;
        case cv::EVENT_MBUTTONDOWN   :
        	color = Scalar( 255, 0, 0 );
            rectangle(selMat,Point(x-roi_size,y-roi_size),Point(x+roi_size,y+roi_size),color,cv::FILLED);
            rectangle(src,Point(x-roi_size,y-roi_size),Point(x+roi_size,y+roi_size),color,cv::FILLED);
            imshow("Select spots",src);
            waitKey(1);
            break;
    }
}

void pMouse( int event, int x, int y, int f, void* ){
	Scalar color;
	switch(event){
        case  cv::EVENT_LBUTTONDOWN  :
        	color = Scalar( 255, 255, 255 );
            rectangle(selMat,Point(x-roi_size,y-roi_size),Point(x+roi_size,y+roi_size),255,cv::FILLED);
            rectangle(src,Point(x-roi_size,y-roi_size),Point(x+roi_size,y+roi_size),color,cv::FILLED);
            imshow("Select spots",src);
            waitKey(1);
            break;
    }
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

void thresholdGUI( int, void* ){
  /* 0: Binary
     1: Binary Inverted
     2: Threshold Truncated
     3: Threshold to Zero
     4: Threshold to Zero Inverted
   */

  threshold( src1, dst, threshold_value, max_BINARY_value,threshold_type );
  Mat dst_inv = 255-dst;

  vector<Mat> split_bgr;
  split(gray, split_bgr);

  Mat b,g,r;
  split_bgr[0].copyTo(b,dst_inv);
  split_bgr[1].copyTo(g,dst_inv);
  split_bgr[2].copyTo(r,dst_inv);

  split_bgr[0] = b;
  split_bgr[1] = g;
  split_bgr[2] = r;

  Mat out;
  merge(split_bgr,out);
  imshow( "threshold", out );
}

void confusionGUI(Mat orig, Mat predicted, Mat labeled, int size){
	src = orig.clone();
	roi_size = size;
	selMat = Mat::zeros(src.size(),CV_8UC1);

	namedWindow("Select spots",WINDOW_NORMAL);
	setMouseCallback("Select spots",pMouse,NULL );
	resizeWindow("Select spots",src.cols,src.rows);
	imshow("Select spots",src);
	while(true){
	    int c;
	    c = waitKey();
	    if( (char)c == 27 ){
	    	break;
	    }
	}

	vector<vector<Point> > sel_contours;
	vector<Vec4i> sel_hierarchy;
    findContours( selMat, sel_contours, sel_hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );

    double confusion[4] = {0};
    for(unsigned int i=0; i < sel_contours.size(); i++){
      	for(unsigned int j=0; j<sel_contours[i].size(); j++){
      	    Mat temp = Mat::zeros(orig.size(),CV_8UC1);
      		drawContours(temp, sel_contours, i, 255, cv::FILLED);
      		confusion[0] = confusion[0] + sum(temp & (predicted & labeled))[0]/255;
      		confusion[1] = confusion[1] + sum(temp & ((255-predicted) & labeled))[0]/255;
      		confusion[2] = confusion[2] + sum(temp & (predicted & (255-labeled)))[0]/255;
      		confusion[3] = confusion[3] + sum(temp & ((255-predicted) & (255-labeled)))[0]/255;
      	}
    }

    double tp,fp,tn,fn;
    tp = confusion[0];
    fp = confusion[2];
    fn = confusion[1];
    tn = confusion[3];

    cout << endl << "Classifier summary" << endl;
    cout << "------------------------------------------" << endl;
    cout << "True Positives: " << tp << endl <<
    		"False Positives: "<< fp << endl <<
    		"False Negatives: "<< fn << endl <<
    		"True Negatives: "<< tn << endl << endl;

    cout << "Precision: " << tp/(tp+fp) << endl;
    cout << "Recall: " << tp/(tp+fn) << endl;
    cout << "Accuracy: " << (tp+tn)/(tp+tn+fp+fn) << endl << endl;

    cout << "True positive rate: " << tp/(tp+fn) << endl;
    cout << "False negative rate: " << fn/(fn+tp) << endl << endl;

    cout << "True negative rate: " << tn/(tn+fp) << endl;
    cout << "False postive rate: " <<  fp/(fp+tn)<< endl << endl;

    cout << "False discovery rate: " << fp/(fp+tp) << endl << endl;
}

void selectionGUI(Mat orig, string orig_fname, Mat mask, int size, string shape_fname, string color_fname){
	src = orig.clone();
	roi_size = size;
	selMat = Mat::zeros(src.size(),src.type());

	cout << "\n\e[1mLeft\e[0m - Red\n"
			"\e[1mMiddle\e[0m - Blue\n"
			"\e[1mRight\e[0m - Green\n" << endl;

	namedWindow("Select spots",WINDOW_NORMAL);
	setMouseCallback("Select spots",kMouse,NULL );
	resizeWindow("Select spots",src.cols,src.rows);
	imshow("Select spots",src);
	while(true){
	    int c;
	    c = waitKey();
	    if( (char)c == 27 ){
	    	break;
	    }
	}

	vector<Mat> roi_bgr(3);
	split(selMat,roi_bgr);

	vector<vector<Point> > b_contours, g_contours, r_contours, pred_contours;
	vector<Vec4i> b_hierarchy, g_hierarchy, r_hierarchy, pred_hierarchy;
    findContours( roi_bgr[0], b_contours, b_hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );
    findContours( roi_bgr[1], g_contours, g_hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );
    findContours( roi_bgr[2], r_contours, r_hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );
    findContours( mask.clone(), pred_contours, pred_hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );

    Mat lab;
	cvtColor(src.clone(), lab, cv::COLOR_BGR2Lab);
	vector<Mat> split_lab;
	split(lab, split_lab);

	Mat map=orig.clone();
	for(unsigned int b=0; b<b_contours.size(); b++){
		Mat z = Mat::zeros(src.size(),CV_8UC1);
		for(unsigned int i=0; i < pred_contours.size(); i++){
			for(unsigned int j=0; j<pred_contours[i].size(); j++){
				int test = pointPolygonTest(b_contours[b],Point2f(pred_contours[i][j]),false);
				if(test==1 || test == 0){
					drawContours(map, pred_contours, i, Scalar(255,0,0),cv::FILLED);
					drawContours(z, pred_contours, i, 255, cv::FILLED);
					break;
				}
			}
		}

		//z = z & mask;

		Moments m = moments(z,true);
		Point p(m.m10/m.m00, m.m01/m.m00);
		putText(map,to_string(b),p,FONT_HERSHEY_DUPLEX, 0.5, Scalar(10,10,10), 2);

		Mat tmask = z;
		Mat mask;
		vector<Point> cc = keep_roi(tmask,Point(0,0),Point(src.size[0],src.size[1]),mask);
		vector<double> shapes_data = get_shapes(cc,mask);
		Mat gray_data = get_nir(split_lab[0], mask);

		//-- Write shapes to file
		string name_shape= shape_fname;
		ofstream shape_file;
		shape_file.open(name_shape.c_str(),ios_base::app);
		shape_file << orig_fname << " " << "blue" << " " << b << " ";
		for(int i=0;i<20;i++){
			shape_file << shapes_data[i];
				if(i != 19){
					shape_file << " ";
				}
		}
		shape_file << endl;
		shape_file.close();

		//-- Write color to file
		string name_gray= color_fname;
		ofstream gray_file;
		gray_file.open(name_gray.c_str(),ios_base::app);
		gray_file << orig_fname << " " << "blue" << " " << b << " ";
		for(int i=0;i<255;i++){
			gray_file << gray_data.at<float>(i,0) << " ";
		}
		gray_file << endl;
		gray_file.close();
	}

	for(unsigned int g=0; g<g_contours.size(); g++){
	    Mat z = Mat::zeros(src.size(),CV_8UC1);
		for(unsigned int i=0; i < pred_contours.size(); i++){
			for(unsigned int j=0; j<pred_contours[i].size(); j++){
				int test = pointPolygonTest(g_contours[g],Point2f(pred_contours[i][j]),false);
				if(test==1 || test == 0){
					drawContours(map, pred_contours, i, Scalar(0,255,0), cv::FILLED);
					drawContours(z, pred_contours, i, 255, cv::FILLED);
					break;
				}
			}
		}

	    Moments m = moments(z,true);
	    Point p(m.m10/m.m00, m.m01/m.m00);
        putText(map,to_string(g),p,FONT_HERSHEY_DUPLEX, 0.5, Scalar(10,10,10), 2);

	    Mat tmask = z;
		Mat mask;
		vector<Point> cc = keep_roi(tmask,Point(0,0),Point(src.size[0],src.size[1]),mask);
		vector<double> shapes_data = get_shapes(cc,mask);
		Mat gray_data = get_nir(split_lab[0], mask);

		//-- Write shapes to file
		string name_shape= shape_fname;
		ofstream shape_file;
		shape_file.open(name_shape.c_str(),ios_base::app);
		shape_file << orig_fname << " " << "green" << " " << g << " ";
		for(int i=0;i<20;i++){
			shape_file << shapes_data[i];
			if(i != 19){
				shape_file << " ";
			}
		}
		shape_file << endl;
		shape_file.close();

		//-- Write color to file
		string name_gray= color_fname;
		ofstream gray_file;
		gray_file.open(name_gray.c_str(),ios_base::app);
		gray_file << orig_fname << " " << "green" << " " << g << " ";
		for(int i=0;i<255;i++){
			gray_file << gray_data.at<float>(i,0) << " ";
		}
		gray_file << endl;
		gray_file.close();
	}

	for(unsigned int r=0; r<r_contours.size(); r++){
	    Mat z = Mat::zeros(src.size(),CV_8UC1);
		for(unsigned int i=0; i < pred_contours.size(); i++){
			for(unsigned int j=0; j<pred_contours[i].size(); j++){
				int test = pointPolygonTest(r_contours[r],Point2f(pred_contours[i][j]),false);
				if(test==1 || test == 0){
					drawContours(map, pred_contours, i, Scalar(0,0,255), cv::FILLED);
					drawContours(z, pred_contours, i, 255, cv::FILLED);
					break;
				}
			}
		}

	    Moments m = moments(z,true);
	    Point p(m.m10/m.m00, m.m01/m.m00);
        putText(map,to_string(r),p,FONT_HERSHEY_DUPLEX, 0.5, Scalar(10,10,10), 2);

	    Mat tmask = z;
		Mat mask;
		vector<Point> cc = keep_roi(tmask,Point(0,0),Point(src.size[0],src.size[1]),mask);
		vector<double> shapes_data = get_shapes(cc,mask);
		Mat gray_data = get_nir(split_lab[0], mask);

		//-- Write shapes to file
		string name_shape= shape_fname;
		ofstream shape_file;
		shape_file.open(name_shape.c_str(),ios_base::app);
		shape_file << orig_fname << " " << "red" << " " << r << " ";
		for(int i=0;i<20;i++){
			shape_file << shapes_data[i];
			if(i != 19){
				shape_file << " ";
			}
		}
		shape_file << endl;
		shape_file.close();

		//-- Write color to file
		string name_gray= color_fname;
		ofstream gray_file;
		gray_file.open(name_gray.c_str(),ios_base::app);
		gray_file << orig_fname << " " << "red" << " " << r << " ";
		for(int i=0;i<255;i++){
			gray_file << gray_data.at<float>(i,0) << " ";
		}
		gray_file << endl;
		gray_file.close();
	}

	vector<string> sub_str;
	const string full_str = orig_fname;
	char del = '.';
	split(full_str,del,sub_str);
	string new_name = sub_str[0]+"_colorMap.png";
	imwrite(new_name,map);
}
