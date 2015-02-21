#include <vector>
#include <string>
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "fast/fast.h"

typedef std::vector<fast::fast_xy> Corners;

namespace utils {

std::string getFileDir()
{
  std::string filename(__FILE__);
  for (auto s = filename.rbegin(); s < filename.rend(); ++s)
    if(*s == '/')
      return std::string(filename.begin(), (s+1).base());
  std::cout << "ERROR getFileDir(): could not decompose string" << std::endl;
  return std::string("/");
}

void showFeatures(
    const cv::Mat& img,
    const Corners& corners)
{
  cv::Mat img_rgb(img.size(), CV_8UC3);
  cv::cvtColor(img, img_rgb, cv::COLOR_GRAY2RGB);
  for(auto c : corners)
  {
    cv::circle(img_rgb, cv::Point2i(c.x, c.y), 2, cv::Scalar(0,255,0));
  }
  cv::imshow("corners", img_rgb);
  cv::waitKey(0);
}

void displayGrid(
    const cv::Mat& img,
    const std::vector<uint8_t>& grid_occupancy,
    const int cell_size)
{
  cv::Mat img_occ(img.size(), img.type());
  const int stride = (img.cols/cell_size+1);
  for(int y=0; y<img.rows; ++y)
  {
    uint8_t* img_ptr = img_occ.ptr<uint8_t>(y);
    const uint8_t* occ_ptr = &grid_occupancy[(y/cell_size)*stride];
    for(int x=0; x<img.cols; ++x)
    {
      img_ptr[x] = occ_ptr[x/cell_size] > 0 ? 255 : 0;
    }
  }
  cv::imshow("occupancy grid", img_occ);
  cv::waitKey(0);
}



} // namespace utils

int main (int /*argc*/, char** /*argv*/)
{

  const int num_trials = 100;
  const int threshold = 20;
  const int cell_size = 32;

  // load image
  std::string filename = utils::getFileDir()+"/data/test.jpg";
  std::cout << "Test FAST detector on image: " << filename << std::endl;
  cv::Mat img = cv::imread(filename, 0);
  assert(!img.empty());

  // create occupancy grid
  std::vector<uint8_t> occupancy_grid((img.cols/cell_size+1)*(img.rows/cell_size+1), 0);

  for(size_t i=0; i<occupancy_grid.size(); ++i)
    occupancy_grid[i] = i%2;
  utils::displayGrid(img, occupancy_grid, cell_size);

  std::cout << "grid size = " << occupancy_grid.size() << std::endl;


  // cv::imshow("input_image", img);
  // cv::waitKey(50);

  std::cout << "Testing PLAIN version" << std::endl;
  Corners corners;
  double time_accumulator = 0;
  for (int i = 0; i < num_trials; ++i)
  {
    corners.clear();
    double t = (double)cv::getTickCount();
    fast::fast_corner_detect_9((fast::fast_byte *)(img.data), img.cols, img.rows, img.cols, threshold, corners);
    time_accumulator +=  ((cv::getTickCount() - t) / cv::getTickFrequency());
  }
  printf("PLAIN version took %f seconds (average over %d trials).\n", time_accumulator/((double)num_trials), num_trials );
  std::cout << std::endl << "Fast feature detector test: "<< corners.size() << " features detected." << std::endl;

  // nonmaxima suppression
  time_accumulator = 0;
  for (int i = 0; i < num_trials; ++i)
  {
    std::vector<int> scores, nonmax_corners;
    double t = (double)cv::getTickCount();
    fast::fast_corner_score_10((fast::fast_byte *)(img.data), img.step,
                               corners, threshold, scores);
    fast::fast_nonmax_3x3(corners, scores, nonmax_corners);
    time_accumulator +=  ((cv::getTickCount() - t) / cv::getTickFrequency());
  }
  printf("nonmax took %f seconds\n", time_accumulator/((double)num_trials) );



  std::cout << "Testing PLAIN masked version" << std::endl;
  time_accumulator = 0;
  for (int i = 0; i < num_trials; ++i)
  {
    corners.clear();
    double t = (double)cv::getTickCount();
    fast::fast_corner_detect_9_masked(
          (fast::fast_byte *)(img.data), img.cols, img.rows, img.cols,
          threshold, occupancy_grid, cell_size, corners);
    time_accumulator +=  ((cv::getTickCount() - t) / cv::getTickFrequency());
  }
  printf("PLAIN masked version took %f seconds (average over %d trials).\n", time_accumulator/((double)num_trials), num_trials );
  std::cout << std::endl << "Fast feature detector test: "<< corners.size() << " features detected." << std::endl;

  // display results
  //utils::showFeatures(img, corners);


  // nonmaxima suppression
  time_accumulator = 0;
  for (int i = 0; i < num_trials; ++i)
  {
    std::vector<int> scores, nonmax_corners;
    double t = (double)cv::getTickCount();
    fast::fast_corner_score_10((fast::fast_byte *)(img.data), img.step,
                               corners, threshold, scores);
    fast::fast_nonmax_3x3(corners, scores, nonmax_corners);
    time_accumulator +=  ((cv::getTickCount() - t) / cv::getTickFrequency());
  }
  printf("nonmax took %f seconds\n", time_accumulator/((double)num_trials) );





  #if __ARM_NEON__
  std::cout << "Testing NEON version" << std::endl;
  time_accumulator = 0;
  for (int i = 0; i < NUM_TRIALS; ++i) {
   corners.clear();
    double t = (double)cv::getTickCount();
    fast::fast_corner_detect_9_neon((fast::fast_byte *)(img.data), img.cols, img.rows, img.cols, threshold, corners);
    time_accumulator +=  ((cv::getTickCount() - t) / cv::getTickFrequency());
  }
  printf("NEON version took %f seconds (average over %d trials).\n", time_accumulator/((double)NUM_TRIALS), NUM_TRIALS);
  std::cout << std::endl << "Fast feature detector test: "<< corners.size() << " features detected." << std::endl;
  #endif

#if __SSE2__
  std::cout << "Testing SSE2 version" << std::endl;
  time_accumulator = 0;
  for (int i = 0; i < num_trials; ++i) {
   corners.clear();
    double t = (double)cv::getTickCount();
    fast::fast_corner_detect_10_sse2((fast::fast_byte *)(img.data), img.cols, img.rows, img.cols, threshold, corners);
    time_accumulator +=  ((cv::getTickCount() - t) / cv::getTickFrequency());
  }
  printf("SSE2 version took %f seconds (average over %d trials).\n", time_accumulator/((double)num_trials), num_trials);
  std::cout << std::endl << "Fast feature detector test: "<< corners.size() << " features detected." << std::endl;


  std::cout << std::endl << "Testing SSE2 MASKED version" << std::endl;
  time_accumulator = 0;
  for (int i = 0; i < num_trials; ++i) {
   corners.clear();
    double t = (double)cv::getTickCount();
    fast::fast_corner_detect_10_sse2_masked(
          (fast::fast_byte *)(img.data), img.cols, img.rows, img.cols,
          threshold, occupancy_grid, cell_size, corners);
    time_accumulator +=  ((cv::getTickCount() - t) / cv::getTickFrequency());
  }
  printf("SSE2 MASKED version took %f seconds (average over %d trials).\n", time_accumulator/((double)num_trials), num_trials);
  std::cout << std::endl << "Fast feature detector test: "<< corners.size() << " features detected." << std::endl;
#endif

  // display results
  utils::showFeatures(img, corners);

  return 0;
}
