#include "frame.h"

std::pair<cv::Mat, cv::Mat> CropImageByDetectBox(const cv::Mat &input_image,
                                                 const ObjBox &box);
