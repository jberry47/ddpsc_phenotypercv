#!/bin/bash
printf "Pulling repository...\n"
cd && cd programs/PhenotyperCV/ddpsc_phenotypercv/ && git pull

printf "\nCreating shared object files...\n"
cd && cd programs/PhenotyperCV/
g++ -I/shares/bioinfo/installs/opencv-3.3.0/install/include/ -I/shares/bioinfo/installs/eigen/Eigen -I/home/jberry/programs/PhenotyperCV/ddpsc_phenotypercv/include -O0 -g3 -Wall -c -std=c++11 -MMD -MP -MF"./ddpsc_phenotypercv/src/phenotypercv.d" -MT"./ddpsc_phenotypercv/src/phenotypercv.d" -o "./ddpsc_phenotypercv/src/phenotypercv.o" "./ddpsc_phenotypercv/src/phenotypercv.cpp"
g++ -I/shares/bioinfo/installs/opencv-3.3.0/install/include/ -I/shares/bioinfo/installs/eigen/Eigen -I/home/jberry/programs/PhenotyperCV/ddpsc_phenotypercv/include -O0 -g3 -Wall -c -std=c++11 -MMD -MP -MF"./ddpsc_phenotypercv/src/color_calibration.d" -MT"./ddpsc_phenotypercv/src/color_calibration.d" -o "./ddpsc_phenotypercv/src/color_calibration.o" "./ddpsc_phenotypercv/src/color_calibration.cpp"
g++ -I/shares/bioinfo/installs/opencv-3.3.0/install/include/ -I/shares/bioinfo/installs/eigen/Eigen -I/home/jberry/programs/PhenotyperCV/ddpsc_phenotypercv/include -O0 -g3 -Wall -c -std=c++11 -MMD -MP -MF"./ddpsc_phenotypercv/src/spatial_calibration.d" -MT"./ddpsc_phenotypercv/src/spatial_calibration.d" -o "./ddpsc_phenotypercv/src/spatial_calibration.o" "./ddpsc_phenotypercv/src/spatial_calibration.cpp"
g++ -I/shares/bioinfo/installs/opencv-3.3.0/install/include/ -I/shares/bioinfo/installs/eigen/Eigen -I/home/jberry/programs/PhenotyperCV/ddpsc_phenotypercv/include -O0 -g3 -Wall -c -std=c++11 -MMD -MP -MF"./ddpsc_phenotypercv/src/feature_extraction.d" -MT"./ddpsc_phenotypercv/src/feature_extraction.d" -o "./ddpsc_phenotypercv/src/feature_extraction.o" "./ddpsc_phenotypercv/src/feature_extraction.cpp"
g++ -I/shares/bioinfo/installs/opencv-3.3.0/install/include/ -I/shares/bioinfo/installs/eigen/Eigen -I/home/jberry/programs/PhenotyperCV/ddpsc_phenotypercv/include -O0 -g3 -Wall -c -std=c++11 -MMD -MP -MF"./ddpsc_phenotypercv/src/morphology.d" -MT"./ddpsc_phenotypercv/src/morphology.d" -o "./ddpsc_phenotypercv/src/morphology.o" "./ddpsc_phenotypercv/src/morphology.cpp"
g++ -I/shares/bioinfo/installs/opencv-3.3.0/install/include/ -I/shares/bioinfo/installs/eigen/Eigen -I/home/jberry/programs/PhenotyperCV/ddpsc_phenotypercv/include -O0 -g3 -Wall -c -std=c++11 -MMD -MP -MF"./ddpsc_phenotypercv/src/miscellaneous.d" -MT"./ddpsc_phenotypercv/src/miscellaneous.d" -o "./ddpsc_phenotypercv/src/miscellaneous.o" "./ddpsc_phenotypercv/src/miscellaneous.cpp"
g++ -I/shares/bioinfo/installs/opencv-3.3.0/install/include/ -I/shares/bioinfo/installs/eigen/Eigen -I/home/jberry/programs/PhenotyperCV/ddpsc_phenotypercv/include -O0 -g3 -Wno-narrowing -c -std=c++11 -MMD -MP -MF"./ddpsc_phenotypercv/src/machine_learning.d" -MT"./ddpsc_phenotypercv/src/machine_learning.d" -o "./ddpsc_phenotypercv/src/machine_learning.o" "./ddpsc_phenotypercv/src/machine_learning.cpp"

printf "\nLinking shared object files...\n"
g++ -L/shares/bioinfo/installs/opencv-3.3.0/install/lib \
./ddpsc_phenotypercv/src/color_calibration.o ./ddpsc_phenotypercv/src/feature_extraction.o ./ddpsc_phenotypercv/src/morphology.o ./ddpsc_phenotypercv/src/phenotypercv.o ./ddpsc_phenotypercv/src/spatial_calibration.o ./ddpsc_phenotypercv/src/miscellaneous.o ./ddpsc_phenotypercv/src/machine_learning.o \
-lopencv_core -lopencv_aruco -lopencv_calib3d -lopencv_highgui -lopencv_features2d -lopencv_imgproc -lopencv_ximgproc -lopencv_imgcodecs -lopencv_ml \
-o "PhenotyperCV"

printf "\nCleaning directories...\n"
rm ddpsc_phenotypercv/src/*.o
rm ddpsc_phenotypercv/src/*.d

printf "Done\n"
