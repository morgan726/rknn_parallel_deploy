#include "model/model.h"

int yolov11::preprocess(const FrameInfo& frame) {
  rga_buffer_t src;
  rga_buffer_t dst;
  im_rect src_rect;
  im_rect dst_rect;
  memset(&src_rect, 0, sizeof(src_rect));
  memset(&dst_rect, 0, sizeof(dst_rect));
  memset(&src, 0, sizeof(src));
  memset(&dst, 0, sizeof(dst));

  if (frame.img_width != app_ctx_.model_width || frame.img_height != app_ctx_.model_height) {
    preprocessed_data_ = malloc(app_ctx_.model_height * app_ctx_.model_width * app_ctx_.model_channel);
    memset(preprocessed_data_, 0x00, app_ctx_.model_height * app_ctx_.model_width * app_ctx_.model_channel);

    src = wrapbuffer_virtualaddr((void*)frame.orig_img.data, frame.img_width, frame.img_height, RK_FORMAT_RGB_888);
    dst = wrapbuffer_virtualaddr((void*)preprocessed_data_, app_ctx_.model_width, app_ctx_.model_height,
                                 RK_FORMAT_RGB_888);
    int ret = imcheck(src, dst, src_rect, dst_rect);
    if (IM_STATUS_NOERROR != ret) {
      printf("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
      return -1;
    }
    IM_STATUS STATUS = imresize(src, dst);
  }
  return 0;
}

int yolov11::postprocess(rknn_output* outputs, FrameInfo& frame) {
  auto attrs = app_ctx_.output_attrs;
  for (int i = 0; i < app_ctx_.io_num.n_output; ++i) {
    scales.push_back(attrs[i].scale);
    zps.push_back(attrs[i].zp);
  }

  for (int i = 0; i < app_ctx_.io_num.n_output; ++i) pblob[i] = (int8_t*)outputs[i].buf;

  std::vector<float> rects;
  post.GetConvDetectionResult(pblob, zps, scales, rects);

  for (size_t i = 0; i < rects.size(); i += 6) {
    ObjBox box;
    box.score = rects[i + 1];
    if (box.score < frame.alg_parm.mod_thres) continue;
    box.x1 = int(rects[i + 2] * float(frame.img_width) + 0.5);
    box.y1 = int(rects[i + 3] * float(frame.img_height) + 0.5);
    box.x2 = int(rects[i + 4] * float(frame.img_width) + 0.5);
    box.y2 = int(rects[i + 5] * float(frame.img_height) + 0.5);
    frame.dete_result.push_back(box);
    // char text[256]
    // sprintf(text, "%d:%.2f", (int)rects[i], box.score);
    // rectangle(frame.orig_img, cv::Point(box.x1, box.y1), cv::Point(box.x2, box.y2), cv::Scalar(255, 0, 0), 2);
    // putText(frame.orig_img, text, cv::Point(x1, y1 + 15), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
  }

  //   cv::imwrite("test_yolov11.jpg", frame.orig_img);
  return 0;
}

int rtmpose::preprocess(const FrameInfo& frame) {
  ObjBox box{0, 0, 0, frame.img_width - 1, frame.img_height - 1, -1, 1.0f};
  auto [crop, affine] = CropImageByDetectBox(frame.orig_img, box);
  crop_mat_ = crop;
  affine_transform_reverse_ = affine;
  cv::cvtColor(crop_mat_, input_mat_rgb_, cv::COLOR_BGR2RGB);

  preprocessed_data_ = input_mat_rgb_.data;

  return 0;
}

int rtmpose::postprocess(rknn_output* outputs, FrameInfo& frame) {
  if (app_ctx_.io_num.n_output < 2) return -1;
  float* simcc_x = (float*)outputs[0].buf;
  float* simcc_y = (float*)outputs[1].buf;
  rtmpose_postprocess(simcc_x, simcc_y, affine_transform_reverse_, frame.pose_result);
  return 0;
}
