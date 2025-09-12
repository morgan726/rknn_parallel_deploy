#ifndef MODEL_H
#define MODEL_H

#include <cstddef>
#include <iomanip>
#include "rknn_api.h"
#include "rga.h"
#include "RgaUtils.h"
#include "im2d.h"

#include "model/rknn.h"
#include "process/preprocess.h"
#include "process/postprocess.h"


class yolox : public RknnModelBase
{
public:
    int init(const FrameInfo &frame) override
    {
        return RknnModelBase::init_rknn(frame.alg_parm.mod_path_det);
    }

    AlgoType get_algo_type() const override
    {
        return AlgoType::kYolox;
    }

protected:
    int preprocess(const FrameInfo &frame) override;
    int postprocess(rknn_output* outputs,FrameInfo &frame) override;

private:
    GetResultRectyolov11 post;
    std::vector<float> scales;
    std::vector<int32_t> zps;
    int8_t* pblob[6];
};

class yolov11 : public RknnModelBase
{
public:
    int init(const FrameInfo &frame) override
    {
        return RknnModelBase::init_rknn(frame.alg_parm.mod_path_det);
    }

    AlgoType get_algo_type() const override
    {
        return AlgoType::kYolov11;
    }

protected:
    int preprocess(const FrameInfo &frame) override;
    int postprocess(rknn_output* outputs,FrameInfo &frame) override;

private:
    GetResultRectyolov11 post;
    std::vector<float> scales;
    std::vector<int32_t> zps;
    int8_t* pblob[6];
};

class rtmpose : public RknnModelBase
{
public:
    int init(const FrameInfo &frame) override
    {
        return RknnModelBase::init_rknn(frame.alg_parm.mod_path_pose);
    }

    AlgoType get_algo_type() const override
    {
        return AlgoType::kRtmpose;
    }

protected:
    int preprocess(const FrameInfo& frame) override;
    int postprocess(rknn_output* outputs,FrameInfo &frame) override;

private:
    cv::Mat crop_mat_, input_mat_rgb_, affine_transform_reverse_;
    std::vector<PosePoint> pose_result_;
};

#endif
