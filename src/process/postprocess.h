#include "frame.h"

class GetResultRectyolov11
{
public:
    GetResultRectyolov11();

    ~GetResultRectyolov11();

    int GenerateMeshgrid();

    int GetConvDetectionResult(int8_t **pBlob, std::vector<int> &qnt_zp, std::vector<float> &qnt_scale, std::vector<float> &DetectiontRects);

    float sigmoid(float x);

private:
    std::vector<float> meshgrid;

    const int class_num = 80;
    int headNum = 3;

    int input_w = 640;
    int input_h = 640;
    int strides[3] = {8, 16, 32};
    int mapSize[3][2] = {{80, 80}, {40, 40}, {20, 20}};

    std::vector<float> regdfl;
    float regdeq[16] = {0};

    float nmsThresh = 0.45;
    float objectThresh = 0.5;
};

int rtmpose_postprocess(float *simcc_x_result, float *simcc_y_result,
                        cv::Mat affine_transform_reverse,
                        std::vector<PosePoint> &pose_result);