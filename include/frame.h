#ifndef FRAME_H
#define FRAME_H

#include <stdio.h>

#include <fstream>
#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "rknn_api.h"

struct PosePoint {
  double x, y;
  float score;
};

struct ObjBox {
  int class_id, x1, y1, x2, y2, traker_id;
  float score;
};

enum class Status {
  SUCCESS = 0,          ///< The operation was successful
  ERROR_READWRITE = 1,  ///< Read / Write file failed
  ERROR_MEMORY = 2,     ///< Memory error, such as out of memory, memcpy failed
  INVALID_PARAM = 3,    ///< Invalid parameters
  WRONG_TYPE = 4,       ///< Invalid data type in `any`
  ERROR_BACKEND = 5,    ///< Error occurred in processor
  NOT_IMPLEMENTED = 6,  ///< Function not implemented
  TIMEOUT = 7,          ///< Time expired
  STATUS_COUNT = 8,     ///< Number of status
};

enum class InputType { kImagergb = 0, kImageyuv, kVideo, kDataset, kUndefined };

enum class AlgoType { kYolov11 = 0, kRtmpose, kYolox, kSeg, kDepth, kUndefined };

struct AlgParms {
  std::string mod_path_det;
  std::string mod_path_pose;
  std::string mod_path_seg;
  std::string mod_path_depth;

  float mod_thres{0.4};
  float frame_rate{15};
  int batch_size{1};
  int thread_num{1};
};

struct FrameInfo {
  // void *data;
  cv::Mat orig_img;
  int img_width;
  int img_height;
  std::vector<ObjBox> dete_result;
  std::vector<PosePoint> pose_result;
  InputType in_type;
  AlgoType alg_type;
  AlgParms alg_parm;
};

inline void sync_frame_size(FrameInfo& frame) {
  if (!frame.orig_img.empty()) {
    frame.img_width = frame.orig_img.cols;
    frame.img_height = frame.orig_img.rows;
  } else {
    frame.img_width = 0;
    frame.img_height = 0;
  }
}

inline cv::Mat resize_to_4aligned_width(const cv::Mat& src, const cv::Scalar& fill_color = cv::Scalar(0, 0, 0)) {
  // 1. 检查输入图像有效性
  if (src.empty()) {
    std::cerr << "[ERROR] Input image is empty!" << std::endl;
    return cv::Mat();
  }
  if (src.channels() != 3) {
    std::cerr << "[ERROR] Only support 3-channel RGB/BGR image (CV_8UC3)!" << std::endl;
    return cv::Mat();
  }

  // 2. 计算4对齐的目标宽度（向上取整到最近的4的整数倍）
  int src_width = src.cols;
  int aligned_width = ((src_width + 3) / 4) * 4;  // 核心公式：如19→20，22→24，24→24

  // 3. 若已对齐，直接返回原图像（避免不必要的拷贝）
  if (aligned_width == src_width) {
    return src.clone();  // 克隆避免浅拷贝问题
  }

  // 4. 计算补边宽度（仅在右侧补边，不影响图像主体）
  int border_right = aligned_width - src_width;  // 右侧补边宽度
  int border_left = 0;                           // 左侧不补边
  int border_top = 0;                            // 顶部不补边
  int border_bottom = 0;                         // 底部不补边

  // 5. 补边操作（使用常量边界填充，填充颜色为fill_color）
  cv::Mat aligned_img;
  cv::copyMakeBorder(src,                  // 输入图像
                     aligned_img,          // 输出对齐图像
                     border_top,           // 顶部补边
                     border_bottom,        // 底部补边
                     border_left,          // 左侧补边
                     border_right,         // 右侧补边
                     cv::BORDER_CONSTANT,  // 边界类型：常量填充
                     fill_color            // 填充颜色（默认黑色）
  );

  // 6. 打印补边信息（可选，用于调试）
  std::cout << "[INFO] Image width aligned: " << src_width << " → " << aligned_width
            << " (border right: " << border_right << "px)" << std::endl;

  return aligned_img;
}

typedef struct {
  rknn_context rknn_ctx;
  rknn_input_output_num io_num;
  rknn_tensor_attr* input_attrs;
  rknn_tensor_attr* output_attrs;

  rknn_tensor_mem* input_mems[1];
  rknn_tensor_mem* output_mems[9];
  rknn_tensor_attr* input_native_attrs;
  rknn_tensor_attr* output_native_attrs;

  int model_channel, model_width, model_height;
  bool is_quant;
} rknn_app_context_t;

#endif