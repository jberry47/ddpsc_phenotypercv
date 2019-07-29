#include <opencv2/opencv.hpp>
#include "opencv2/ximgproc.hpp"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <math.h>
#include <Eigen/Dense>

using namespace cv;
using namespace std;
using namespace Eigen;

int roi_size;
Mat src;
Mat kionaMat;
int counter=1;
int rcount=0, gcount=0, bcount=0;

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
            rectangle(kionaMat,Point(x-roi_size,y-roi_size),Point(x+roi_size,y+roi_size),color,cv::FILLED);
            rectangle(src,Point(x-roi_size,y-roi_size),Point(x+roi_size,y+roi_size),color,cv::FILLED);
            imshow("Select spots",src);
            waitKey(1);
            break;
        case cv::EVENT_RBUTTONDOWN   :
        	color = Scalar( 0, 255, 0 );
            rectangle(kionaMat,Point(x-roi_size,y-roi_size),Point(x+roi_size,y+roi_size),color,cv::FILLED);
            rectangle(src,Point(x-roi_size,y-roi_size),Point(x+roi_size,y+roi_size),color,cv::FILLED);
            imshow("Select spots",src);
            waitKey(1);
            break;
        case cv::EVENT_MBUTTONDOWN   :
        	color = Scalar( 255, 0, 0 );
            rectangle(kionaMat,Point(x-roi_size,y-roi_size),Point(x+roi_size,y+roi_size),color,cv::FILLED);
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

void showImage(Mat img){
	namedWindow("Image",WINDOW_NORMAL);
	        	    resizeWindow("Image",800,800);
	        	    imshow("Image", img);
	waitKey(0);
}
