#include <vector>
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "fast/fast.h"

int main (int argc, char * argv[]) {
   const int NUM_TRIALS = 1000;
   std::cout << "Fast feature detector test" << std::endl;
   std::vector<fast::fast_xy> corners;
   cv::Mat img = cv::imread("../test/data/bench1.png", 0);
   cv::Mat downSampled; 
   cv::resize(img, downSampled, cv::Size(752, 480));
   img = downSampled;

  // cv::imshow("input_image", img);
  // cv::waitKey(50);   

   std::cout << "Testing PLAIN version" << std::endl;
   double time_accumulator = 0;
   for (int i = 0; i < NUM_TRIALS; ++i) {
      corners.clear();
      double t = (double)cv::getTickCount();
      fast::fast_corner_detect_9((fast::fast_byte *)(img.data), img.cols, img.rows, img.cols, 75, corners);
      time_accumulator +=  ((cv::getTickCount() - t) / cv::getTickFrequency());
   }
   printf("PLAIN version took %f seconds (average over %d trials).\n", time_accumulator/((double)NUM_TRIALS), NUM_TRIALS );
   std::cout << std::endl << "Fast feature detector test: "<< corners.size() << " features detected." << std::endl;

#if __ARM_NEON__
   std::cout << "Testing NEON version" << std::endl;
   time_accumulator = 0;
   for (int i = 0; i < NUM_TRIALS; ++i) {
     corners.clear();
      double t = (double)cv::getTickCount();
      fast::fast_corner_detect_9_neon((fast::fast_byte *)(img.data), img.cols, img.rows, img.cols, 75, corners);
      time_accumulator +=  ((cv::getTickCount() - t) / cv::getTickFrequency());
   }
   printf("NEON version took %f seconds (average over %d trials).\n", time_accumulator/((double)NUM_TRIALS), NUM_TRIALS);
   std::cout << std::endl << "Fast feature detector test: "<< corners.size() << " features detected." << std::endl;
#endif
   
#if __SSE2__
   std::cout << "Testing SSE2 version" << std::endl;
   time_accumulator = 0;
   for (int i = 0; i < NUM_TRIALS; ++i) {
     corners.clear();
      double t = (double)cv::getTickCount();
      fast::fast_corner_detect_9_sse2((fast::fast_byte *)(img.data), img.cols, img.rows, img.cols, 75, corners);
      time_accumulator +=  ((cv::getTickCount() - t) / cv::getTickFrequency());
   }
   printf("SSE2 version took %f seconds (average over %d trials).\n", time_accumulator/((double)NUM_TRIALS), NUM_TRIALS);
   std::cout << std::endl << "Fast feature detector test: "<< corners.size() << " features detected." << std::endl;
#endif

   
   return 0;
}
