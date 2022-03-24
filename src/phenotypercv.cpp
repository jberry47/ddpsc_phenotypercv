/* Title: PhenotyperCV
 * Author: Jeffrey Berry (Bart Lab)
 *
 * Usage: ./PhenotyperCV --help  for more information
 * For more info visit https://github.com/jberry47/ddpsc_phenotypercv/wiki
*/

#include <phenotypercv.h>
//#include <../include/phenotypercv.h>

namespace {
const char* keys  =
        "{m        |       | Mode to run }"
        "{h        |       | Show help documentation }"
        "{i        |       | Input image }"
        "{b        |       | Background image }"
        "{size     |       | Square size (pixels) for DRAW_ROI mode}"
        "{class    |       | machine learning classifier}"
        "{prob     |       | Probability for ML_PROC thresh}"
        "{s        |       | Shape file to write to }"
        "{c        |       | Color file to write to }"
		"{ci       |       | ChArUco calibrate input file }"
		"{cc       |       | Camera calibration file name }"
		"{nx       |       | Number of board spaces - x }"
		"{ny       |       | Number of board spaces - y }"
		"{mw       |       | Marker width }"
		"{aw       |       | ArUco width }"
		"{method   |       | bayes or svm for machine learning }"
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
	bool bool_mlProc = mode == "ML_PROC";
	bool bool_mlPred = mode == "ML_PRED";
	bool bool_mlCreate = mode == "ML_CREATE";
	bool bool_mlStat = mode == "ML_STAT";
  bool bool_qr = mode == "QR";
	bool bool_testing = mode == "TESTING";

  if(bool_testing){
    Mat img = imread(parser.get<string>("i"));
    Mat corrected = CLAHE_correct_rgb(img);
    //showImage(corrected,"corrected");
    cout << "test" << endl;

    vector<string> sub_str;
    const string full_str = string(parser.get<string>("i"));
    char del = '.';
    string suffix = "_clahe_corrected.png";
    split(full_str,del,sub_str);
    string new_name = sub_str[0]+suffix;
    imwrite(new_name,corrected);
	}
	else if(bool_qr){
	  Mat im = imread(parser.get<string>("i"));
	  vector<decodedObject> decodedObjects = decodeSymbols(im);
	  // displaySymbols(im, decodedObjects);
	  if(decodedObjects.size() > 0){
		  for(unsigned int i = 0; i < decodedObjects.size(); i++){
			  cout << "Data: " << decodedObjects[i].data << endl;
		  }
	  }else{
		  cout << "No objects detected" << endl;
	  }
	}
	else if(bool_mlProc){
		if(!(parser.has("i") && parser.has("class") && parser.has("size") && parser.has("s")  && parser.has("c") && parser.has("method"))){
			cout << "Using mode ML_PROC requires input: -method=[bayes,svm] -i=inputImage -class=input_classifier.yaml -size=number(range 0-20) -s=shapes_output.txt -c=gray_output.txt" << endl;
		}else{
			Mat inputImage = imread(parser.get<string>("i"));

			roi_size = parser.get<int>("size");
			src = inputImage.clone();

			Mat corrected = grayCorrect(inputImage);
			Mat lab;
			cvtColor(corrected, lab, cv::COLOR_BGR2Lab);
			vector<Mat> split_lab;
			split(lab, split_lab);
			Mat l_thresh;
			threshold(split_lab[0],l_thresh,20,255,THRESH_BINARY);
			Mat l_erode;
			erode(l_thresh,l_erode, Mat(), Point(-1, -1), 3, 1, 1);

			Mat response;
			string suffix;
			if(parser.get<string>("method") == "bayes"){
				suffix = "_bayes_pred.png";
				response = predictBC(corrected,parser.get<string>("class"));
			}else if(parser.get<string>("method") == "svm"){
				suffix = "_svm_pred.png";
				response = predictSVM(corrected,parser.get<string>("class"));
			}else{
				cout << "Unknown method: expecting either bayes or svm" << endl;
				return(1);
			}

			src1 = response.clone();
			gray = corrected.clone();
			namedWindow("threshold", WINDOW_NORMAL );
			resizeWindow("threshold",src1.cols,src1.rows);
			createTrackbar( "trackbar_type", "threshold", &threshold_type, max_type, thresholdGUI );
			createTrackbar( "trackbar_value", "threshold", &threshold_value, max_value, thresholdGUI );
			thresholdGUI(0,0);
			while(true){
			    int c;
			    c = waitKey();
			    if( (char)c == 27 ){
			    	destroyWindow("threshold");
			    	break;
			    }
			}

			Mat r_thresh;
			threshold(response,r_thresh,threshold_value,255,threshold_type);
			Mat pred_thresh = r_thresh & l_erode;

			selectionGUI(corrected.clone(),parser.get<string>("i"),pred_thresh.clone(),parser.get<int>("size"), parser.get<string>("s"),parser.get<string>("c"),threshold_value);

			vector<string> sub_str;
			const string full_str = string(parser.get<string>("i"));
			char del = '.';
			split(full_str,del,sub_str);
			string new_name = sub_str[0]+suffix;
			imwrite(new_name,response);
			new_name = sub_str[0]+"_grayCorrect.png";
			imwrite(new_name,corrected);
		}
	}
	else if(bool_mlCreate){
		if(!(parser.has("i") && parser.has("b") && parser.has("class") && parser.has("method"))){
			cout << "Using mode ML_CREATE requires input: -method=[bayes,svm] -i=inputImage -b=labeledImage -class=output_classifier.yaml" << endl;
		}else{
			Mat inputImage = imread(parser.get<string>("i"));
			Mat labels = imread(parser.get<string>("b"),0);
			if(parser.get<string>("method") == "bayes"){
				trainBC(inputImage, labels, parser.get<string>("class"));
			}else if(parser.get<string>("method") == "svm"){
				trainSVM(inputImage, labels, parser.get<string>("class"));
			}else{
				cout << "Unknown method: expecting either bayes or svm" << endl;
				return(1);
			}
		}
	}
	else if(bool_mlPred){
		if(!(parser.has("i") && parser.has("class") && parser.has("method"))){
			cout << "Using mode ML_PRED requires input: -method=[bayes,svm] -i=inputImage -class=input_classifier.yaml" << endl;
		}else{
			Mat inputImage = imread(parser.get<string>("i"));
			//Mat corrected = nonUniformCorrect(inputImage,5);
			string suffix;
			Mat response;
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
			vector<string> sub_str;
			const string full_str = string(parser.get<string>("i"));
			char del = '.';
			split(full_str,del,sub_str);
			string new_name = sub_str[0]+suffix;
			imwrite(new_name,response);
		}
	}
	else if(bool_mlStat){
		if(!(parser.has("i") && parser.has("class") && parser.has("b") && parser.has("size") && parser.has("method"))){
			cout << "Using mode ML_STAT requires input: -method=[bayes,svm] -i=inputImage -b=labeledImage -class=input_classifier.yaml -size=integer(range 0-20)" << endl;
		}else{
			Mat inputImage = imread(parser.get<string>("i"));
			Mat labels = imread(parser.get<string>("b"),0);
			Mat response;
			if(parser.get<string>("method") == "bayes"){
				response = predictBC(inputImage,parser.get<string>("class"));
			}else if(parser.get<string>("method") == "svm"){
				response = predictSVM(inputImage,parser.get<string>("class"));
			}else{
				cout << "Unknown method: expecting either bayes or svm" << endl;
				return(1);
			}

			Mat r_thresh,l_thresh;
			src1 = response;
			namedWindow("threshold", WINDOW_NORMAL );
			createTrackbar( "trackbar_type", "threshold", &threshold_type, max_type, thresholdGUI );
			createTrackbar( "trackbar_value", "threshold", &threshold_value, max_value, thresholdGUI );
			thresholdGUI(0,0);
			while(true){
			    int c;
			    c = waitKey();
			    if( (char)c == 27 ){
			    	break;
			    }
			}

			threshold(response,r_thresh,threshold_value,255,threshold_type);
			threshold(labels,l_thresh,25,255,THRESH_BINARY);
			confusionGUI(inputImage, r_thresh, l_thresh, parser.get<int>("size"));
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
		    charuco_calibrate(parser.get<string>("cc"),parser.get<string>("ci"),parser.get<int>("d"),parser.get<int>("nx"),parser.get<int>("ny"),parser.get<float>("mw"),parser.get<float>("aw"));
		}
	}
	else if(bool_charuco_est){
		if(!(parser.has("i") && parser.has("d") &&parser.has("cc") && parser.has("nx") && parser.has("ny") && parser.has("aw") && parser.has("mw"))){
			cout << "Using mode CHARUCO_EST requires input: -i=inputImage -d=dictionaryID -cc=camera_calibration_infile.yaml -nx=num_board_spacesX -ny=num_board_spacesX -mw=marker_width -aw=aruco_width" << endl;
		}else{
			//-- Getting camera calibration details
			Mat inputImage = imread(parser.get<string>("i"));
			Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(parser.get<int>("d")));
			Ptr<cv::aruco::CharucoBoard> board = cv::aruco::CharucoBoard::create(parser.get<int>("nx"), parser.get<int>("ny"), parser.get<float>("mw"), parser.get<float>("aw"), dictionary);

			Mat undistortedCharuco = charuco_estimate(inputImage, parser.get<string>("cc"), dictionary, board);

			vector<string> sub_str;
			const string full_str = string(parser.get<string>("i"));
			char del = '.';
			split(full_str,del,sub_str);
			string new_name = sub_str[0]+"_opp.png";
			imwrite(new_name,undistortedCharuco);
		}
	}
	else if(bool_vis | bool_vis_CH){
		if(bool_vis && !(parser.has("i") && parser.has("b") && parser.has("s") && parser.has("c"))){
			cout << "Using mode VIS requires input: -i=inputImage -b=backgroundImage -s=shapes_file.txt -c=color_file.txt -d=leaf_file.txt" << endl;
		}
		else if(bool_vis_CH && !(parser.has("i") && parser.has("b") && parser.has("s") && parser.has("c"))){
			cout << "Using mode VIS_CH requires input: -i=inputImage -b=backgroundImage -s=shapes_file.txt -c=color_file.txt -d=leaf_file.txt" << endl;
			cout << "In addition to this input, a directory called 'card_masks' must be present and contains binary images of each chip of the input image" << endl;
			cout << "and a CSV called 'target_homography.csv' must be present. This is obtained using the SET_TARGET mode of this program." << endl;
		}else{
			Mat inputImage = imread(parser.get<string>("i"));
			Mat inputBackground = imread(parser.get<string>("b"));

			//-- Processing the VIS image
			Mat adjImage, adjBackground;
			float det=0;
			float D;
			//-- Color homography
			if(bool_vis_CH){
				MatrixXd rh, gh, bh;
				get_standardizations(inputImage, det, rh, gh, bh);
				adjImage = color_homography(inputImage,rh,gh,bh);
        adjBackground = color_homography(inputBackground,rh,gh,bh);
				D = 1-det;
        if(parser.has("debug")){
          vector<string> sub_str;
          const string full_str = string(parser.get<string>("i"));
          char del = '.';
          split(full_str,del,sub_str);
          string new_name = sub_str[0]+"_corrected.png";
          imwrite(new_name,adjImage);
        }
			}else{
				adjImage = inputImage;
			}

			//-- Difference in images
			Mat dest;
			absdiff(adjBackground,adjImage,dest);
			vector<Mat> channels(3); 
			
			split(dest,channels);
			Mat dest_blur;
			blur(channels[0], dest_blur, Size( 2, 2 ) );
			Mat dest_thresh;
			threshold(dest_blur,dest_thresh,115,255,THRESH_BINARY);
			Mat dest_dilate;
			dilate(dest_thresh, dest_dilate, Mat(), Point(-1, -1), 4, 1, 1);
			Mat dest_erode;
			erode(dest_dilate,dest_erode, Mat(), Point(-1, -1), 4, 1, 1);

			//-- Removing barcode
			Mat lab,lab_back;
			cvtColor(adjImage, lab, cv::COLOR_BGR2Lab);
      cvtColor(adjBackground, lab_back, cv::COLOR_BGR2Lab);
			vector<Mat> split_lab,split_lab_back;
			split(lab, split_lab);
      split(lab_back, split_lab_back);
			Mat l_thresh1,l_thresh_back;
      threshold(split_lab[0],l_thresh1,240,255,THRESH_BINARY);
      threshold(split_lab_back[0],l_thresh_back,180,255,THRESH_BINARY);
      Mat barcode_roi,barcode_roi_back;
			vector<Point> cc_barcode = keep_roi(l_thresh1,Point(1146,1368),Point(1359,1479),barcode_roi);
      vector<Point> cc_barcode_back = keep_roi(l_thresh_back,Point(1146,1368),Point(1359,1479),barcode_roi_back);
      Mat barcode_all;
      bitwise_or(barcode_roi,barcode_roi_back,barcode_all);
      Mat barcode_dilate;
			dilate(barcode_all, barcode_dilate, Mat(), Point(-1, -1), 4, 1, 1);
      Mat mask2 = dest_erode - barcode_dilate;

			//-- Remove edges of pot
			/*
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
			*/

			//-- Remove blue stakes
			/*
			Mat b_thresh;
			inRange(split_lab[2],80,115,b_thresh);
			Mat b_er;
			erode(b_thresh,b_er, Mat(), Point(-1, -1), 1, 1, 1);
			Mat b_roi;
			vector<Point> cc1 = keep_roi(b_er,Point(300,600),Point(1610,1310),b_roi);
			Mat b_dil;
			dilate(b_roi,b_dil,Mat(),Point(-1, -1), 6, 1, 1);
			Mat b_xor = pot_roi - b_dil;
			*/

			//-- ROI selector
			Mat mask;
			vector<Point> cc = keep_roi(mask2,Point(550,0),Point(1810,1305),mask); //mask2 from pot_roi

      if(parser.has("d")){
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
        vector<vector<double> > leaf_data = get_leaf_info(classified,filled_mask);
        write_leaves(leaf_data,parser.get<string>("i"),parser.get<string>("d"));
        if(parser.has("debug")){
          vector<string> sub_str;
          const string full_str = string(parser.get<string>("i"));
          char del = '.';
          split(full_str,del,sub_str);
          string new_name = sub_str[0]+"_filled.png";
          imwrite(new_name,filled_mask);
        }
      }

			//-- Getting numerical data
			vector<double> shapes_data = get_shapes(cc,mask); //see feature_extraction.h
			if(bool_vis_CH){
				shapes_data.push_back(D);
			}
			Mat hue_data = get_color(adjImage, mask);

			//-- Write shapes to file
			write_shapes(shapes_data,parser.get<string>("i"),parser.get<string>("s"));
			write_hist(hue_data,parser.get<string>("i"),parser.get<string>("c"));

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

			//-- Remove white stake // 
			Mat dest_stake, new_stake, new_stake_dil;
			inRange(dest_nir,60,255,dest_stake);
			inRange(nirImage, 100, 255, new_stake);
			Mat subtractAway;

			dilate(new_stake, new_stake_dil, Mat(), Point(-1, -1), 1, 1, 1);
			subtractAway = new_stake_dil & dest_nir_thresh;

			Mat onlyPlant = dest_nir_thresh - subtractAway;

	        //-- ROI selector
	    	Mat kept_mask_nir;
	    	vector<Point> cc = keep_roi(onlyPlant,Point(171,102),Point(470,363),kept_mask_nir);

	        //-- Getting numerical data
	    	Mat nir_data = get_nir(nirImage, kept_mask_nir);

	        //-- Writing numerical data
	    	write_hist(nir_data,parser.get<string>("i"),parser.get<string>("c"));

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
		    	cout  << b_avg << ","<< g_avg << "," << r_avg << endl;
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
		cout << "USAGE:" << endl << "\tThere are fourteen modes of use (VIS, VIS_CH, VIS_CH_CHECK, NIR, SET_TARGET, DRAW_ROIS,CHARUCO_CREATE, CHARUCO_CALIB, CHARUCO_EST, ML_CREATE, ML_PRED, ML_STAT, ML_PROC, or AVG_IMGS). Depending on what is chosen, the required inputs change" << endl << endl;
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
		cout << "\t\e[1mML_CREATE\e[0m - Creates and outputs a classifier that is trained on a RGB image and it's respective labeled image." << endl << "\t\t" << "Example: ./PhenotyperCV -m=ML_CREATE -method=bayes -i=input_image.png -b=labeled_image.png -class=output_bayes_classifier.yaml" << endl << endl;
		cout << "\t\e[1mML_PRED\e[0m - Classifies an input image by using a trained classifier to identify features in the image." << endl << "\t\t" << "Example: ./PhenotyperCV -m=ML_PRED -method=bayes -i=input_image.png -class=input_bayes_classifier.yaml" << endl << endl;
		cout << "\t\e[1mML_STAT\e[0m - Outputs classifier statistics from labeled image and classifier input." << endl << "\t\t" << "Example: ./PhenotyperCV -m=ML_STAT -method=bayes -i=test_combine.png -b=test_combine_pink_mask_with269.png -class=input_bayes_classifier.yaml -size=8" << endl << endl;
		cout << "\t\e[1mML_PROC\e[0m - Takes classifier and input image and outputs measurements of objects within user selected regions." << endl << "\t\t" << "Example: ./PhenotyperCV -m=ML_PROC -method=bayes -i=test_combine.png -class=input_bayes_classifier.yaml -size=8 -s=shapes.txt -c=gray.txt" << endl << endl;
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
    	cout << "Mode must be either VIS, VIS_CH, VIS_CH_CHECK, NIR, SET_TARGET, DRAW_ROIS,CHARUCO_CREATE, CHARUCO_CALIB, CHARUCO_EST, ML_CREATE, ML_PRED, ML_STAT, ML_PROC, or AVG_IMGS" << endl;
    	cout << "Use  ./PhenotyperCV -h for more information" << endl;
    }
	return(0);
}


/*
namedWindow("Image",WINDOW_NORMAL);
        	    resizeWindow("Image",800,800);
        	    imshow("Image", b_blur);
waitKey(0);
*/
