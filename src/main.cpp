#include <chrono>
#include <iomanip>
#include <iostream>

#include "model/model.h"
#include "utils/rknnPool.hpp"

// int main(int argc, char **argv) {
//   int ret;
//   FrameInfo frame;
//   frame.alg_parm.mod_path_det = argv[3];
//   frame.alg_parm.mod_path_pose = argv[4];
//   rtmpose rtmpose_model;
//   ret = rtmpose_model.init(frame);
//   yolov11 yolo;
//   ret = yolo.init(frame);
//   while (true)
//   {
//     auto start = std::chrono::high_resolution_clock::now();

//     frame.orig_img = cv::imread(argv[1], 1);
//     frame.img_width = frame.orig_img.cols;
//     frame.img_height = frame.orig_img.rows;
//     std::cout << frame.img_width << "-----------" << frame.img_height << std::endl;
//     ret = yolo.infer(frame);
//     frame.orig_img = cv::imread(argv[2], 1);
//     frame.img_width = frame.orig_img.cols;
//     frame.img_height = frame.orig_img.rows;
//     ret = rtmpose_model.infer(frame);

//     auto end = std::chrono::high_resolution_clock::now();

//     auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
//     double cost_ms = duration.count() / 1000.0;  // 微秒 -> 毫秒

//     std::cout << "单次推理耗时: " << std::fixed << std::setprecision(3) << cost_ms << " ms" << std::endl;
//   }
// }

#include "model/model_manager.hpp"

// int main(int argc, char** argv) {
//   int threadNum = std::atoi(argv[4]);
//   ModelManager manager(threadNum);

//   FrameInfo frame;
//   frame.alg_type = AlgoType::kYolox;
//   frame.alg_parm.mod_path_det = argv[1];
//   frame.alg_parm.mod_thres = 0.5;
//   if (manager.init<yolox>(frame) != 0) {
//     std::cerr << "Yolov11 init failed" << std::endl;
//     return -1;
//   }
//   frame.alg_type = AlgoType::kRtmpose;
//   frame.alg_parm.mod_path_pose = argv[2];
//   // if (manager.init<rtmpose>(frame) != 0) {
//   //   std::cerr << "RTMPose init failed" << std::endl;
//   //   return -1;
//   // }

//   // 4. 批量读取图像（多帧并行处理示例）
//   std::vector<std::string> image_paths = {// argv[3] + std::string("/1_strip_0818.jpg"),
//                                           argv[3] + std::string("/person.jpg")};
//   while (true) {
//     auto start = std::chrono::high_resolution_clock::now();
//     for (const auto& img_path : image_paths) {
//       cv::Mat img = cv::imread(img_path);
//       if (img.empty()) {
//         std::cerr << "Read image " << img_path << " failed" << std::endl;
//         continue;
//       }

//       frame.orig_img = img;
//       sync_frame_size(frame);
//       frame.alg_type = AlgoType::kYolox;
//       if (manager.put(frame) != 0) {
//         std::cerr << "Submit yolo task failed for " << img_path << std::endl;
//       }

//       // frame.orig_img = img.clone();
//       // sync_frame_size(frame);
//       // frame.alg_type = AlgoType::kRtmpose;
//       // if (manager.put(frame) != 0) {
//       //   std::cerr << "Submit pose task failed for " << img_path << std::endl;
//       // }
//     }

//     int total_tasks = image_paths.size() * 2;  // 每个图像2个任务（yolo+pose）
//     while (total_tasks-- > 0) {
//       if (manager.get(frame) != 0) {
//         std::cerr << "Get result failed, remaining tasks: " << total_tasks << std::endl;
//         continue;
//       }

//       // 6.1 处理yolov11结果
//       if (frame.alg_type == AlgoType::kYolov11) {
//         std::cout << "Yolov11: " << frame.orig_img.size() << ", detect " << frame.dete_result.size() << " objects"
//                   << std::endl;
//         // // 绘制检测框（示例）
//         // cv::Mat yolo_img = frame.orig_img.clone();
//         // for (const auto& box : frame.dete_result) {
//         //   cv::rectangle(yolo_img, cv::Point(box.x1, box.y1), cv::Point(box.x2, box.y2), cv::Scalar(255, 0, 0), 2);
//         // }
//         // cv::imwrite("yolo_result_" + std::to_string(total_tasks) + ".jpg", yolo_img);
//       }

//       // 6.2 处理rtmpose结果
//       else if (frame.alg_type == AlgoType::kRtmpose) {
//         std::cout << "RTMPose: " << frame.orig_img.size() << ", get " << frame.pose_result.size() << " key points"
//                   << std::endl;
//         // // 绘制姿态关键点（示例）
//         // cv::Mat pose_img = frame.orig_img.clone();
//         // for (const auto& pt : frame.pose_result) {
//         //   if (pt.score > 0.5) {
//         //     cv::circle(pose_img, cv::Point(pt.x, pt.y), 3, cv::Scalar(0, 255, 0), -1);
//         //   }
//         // }
//         // cv::imwrite("pose_result_" + std::to_string(total_tasks) + ".jpg", pose_img);
//       }
//     }
//     auto end = std::chrono::high_resolution_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
//     double cost_ms = duration.count() / 1000.0;  // 微秒 -> 毫秒
//     std::cout << "单次推理耗时: " << std::fixed << std::setprecision(3) << cost_ms << " ms" << std::endl;
//   }

//   return 0;
// }

#include <sys/time.h>
int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "[ERROR] 参数数量错误！" << std::endl;
    std::cerr << "正确用法：" << argv[0] << " <检测模型路径> <姿态模型路径> <图片/视频路径> <线程数>" << std::endl;
    std::cerr << "示例：" << argv[0] << " yn.rknn strip.rknn person.jpg 2" << std::endl;
    return -1;
  }
  char *det_model_name = argv[1];
  char *pose_model_name = argv[2];
  char *vedio_name = argv[3];
  int threadNum = std::atoi(argv[4]);
  cv::Mat src_img = cv::imread(vedio_name);
  FrameInfo frame;
  frame.img_width = src_img.cols;  // 列
  frame.img_height = src_img.rows;
  frame.orig_img = src_img;
  frame.alg_parm.thread_num = threadNum;
  frame.alg_parm.mod_path_det = det_model_name;
  frame.alg_parm.mod_path_pose = pose_model_name;
  frame.alg_type = AlgoType::kYolox;
  int ret;
  rknnPool<yolox, FrameInfo, int> RknnPool_;
  if (RknnPool_.init(frame) != 0) {
    printf("rknnPool init fail!\n");
    return 1;
  }
  // int fps = 30;
  // BYTETracker tracker(fps, 30);

  int frames = 0;
  struct timeval time;
  gettimeofday(&time, nullptr);
  auto startTime = time.tv_sec * 1000 + time.tv_usec / 1000;
  auto beforeTime = startTime;

  while (true) {
    frame.alg_type = AlgoType::kYolox;
    frame.img_width = src_img.cols;  // 列
    frame.img_height = src_img.rows;
    frame.orig_img = src_img;
    // frame.alg_type = AlgoType::kRtmdet;
    if (RknnPool_.put(frame) != 0) break;
    if (frames >= threadNum && RknnPool_.get(ret) != 0) break;
    frames++;

    if (frames % 120 == 0) {
      gettimeofday(&time, nullptr);
      auto currentTime = time.tv_sec * 1000 + time.tv_usec / 1000;
      printf("平均帧率:\t %f fps/s\n", 120.0 / float(currentTime - beforeTime) * 1000.0 * 0.5);
      beforeTime = currentTime;
      // break;
    }
  }

  // Clear the thread pool
  while (true) {
    if (RknnPool_.get(ret) != 0) break;
  }

  return 0;
}