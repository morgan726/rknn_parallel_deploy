#include "process/preprocess.h"

cv::Mat GetAffineTransform(float center_x, float center_y, float scale_width,
                           float scale_height, int output_image_width,
                           int output_image_height, bool inverse) {
  // solve the affine transformation matrix

  // get the three points corresponding to the source picture and the target
  // picture

  cv::Point2f src_point_1;
  src_point_1.x = center_x;
  src_point_1.y = center_y;

  cv::Point2f src_point_2;
  src_point_2.x = center_x;
  src_point_2.y = center_y - scale_width * 0.5;

  cv::Point2f src_point_3;
  src_point_3.x = src_point_2.x - (src_point_1.y - src_point_2.y);
  src_point_3.y = src_point_2.y + (src_point_1.x - src_point_2.x);

  float alphapose_image_center_x = output_image_width / 2;
  float alphapose_image_center_y = output_image_height / 2;

  cv::Point2f dst_point_1;
  dst_point_1.x = alphapose_image_center_x;
  dst_point_1.y = alphapose_image_center_y;

  cv::Point2f dst_point_2;
  dst_point_2.x = alphapose_image_center_x;
  dst_point_2.y = alphapose_image_center_y - output_image_width * 0.5;

  cv::Point2f dst_point_3;
  dst_point_3.x = dst_point_2.x - (dst_point_1.y - dst_point_2.y);
  dst_point_3.y = dst_point_2.y + (dst_point_1.x - dst_point_2.x);

  cv::Point2f srcPoints[3];
  srcPoints[0] = src_point_1;
  srcPoints[1] = src_point_2;
  srcPoints[2] = src_point_3;

  cv::Point2f dstPoints[3];
  dstPoints[0] = dst_point_1;
  dstPoints[1] = dst_point_2;
  dstPoints[2] = dst_point_3;

  // get affine matrix
  cv::Mat affineTransform;
  if (inverse) {
    affineTransform = cv::getAffineTransform(dstPoints, srcPoints);
  } else {
    affineTransform = cv::getAffineTransform(srcPoints, dstPoints);
  }

  return affineTransform;
}

std::pair<cv::Mat, cv::Mat> CropImageByDetectBox(const cv::Mat &input_image,
                                                 const ObjBox &box) {
  std::pair<cv::Mat, cv::Mat> result_pair;

  if (!input_image.data) {
    return result_pair;
  }

  // deep copy
  cv::Mat input_mat_copy;
  input_image.copyTo(input_mat_copy);

  // calculate the width, height and center points of the human detection box
  int box_width = box.x2 - box.x1;
  int box_height = box.y2 - box.y1;
  int box_center_x = box.x1 + box_width / 2;
  int box_center_y = box.y1 + box_height / 2;
  
  float aspect_ratio = 192.0 / 256.0;

  // adjust the width and height ratio of the size of the picture in the RTMPOSE
  // input
  if (box_width > (aspect_ratio * box_height)) {
    box_height = box_width / aspect_ratio;
  } else if (box_width < (aspect_ratio * box_height)) {
    box_width = box_height * aspect_ratio;
  }

  float scale_image_width = box_width * 1.25;
  float scale_image_height = box_height * 1.25;

  // get the affine matrix
  cv::Mat affine_transform =
      GetAffineTransform(box_center_x, box_center_y, scale_image_width,
                         scale_image_height, 192, 256, false);

  cv::Mat affine_transform_reverse =
      GetAffineTransform(box_center_x, box_center_y, scale_image_width,
                         scale_image_height, 192, 256, true);

  // affine transform
  cv::Mat affine_image;
  cv::warpAffine(input_mat_copy, affine_image, affine_transform,
                 cv::Size(192, 256), cv::INTER_LINEAR);
  // cv::imwrite("affine_img.jpg", affine_image);

  result_pair = std::make_pair(affine_image, affine_transform_reverse);

  return result_pair;
}
