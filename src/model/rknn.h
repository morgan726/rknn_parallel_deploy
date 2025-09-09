#ifndef RKNN_H
#define RKNN_H
#include "frame.h"
#include "rknn_api.h"
#include "rga.h"
#include "RgaUtils.h"
#include "im2d.h"

class RknnModelBase {
public:
    virtual ~RknnModelBase();
    virtual int init(const FrameInfo& frame) = 0;
    virtual int infer(FrameInfo& frame);
    virtual AlgoType get_algo_type() const = 0;

protected:
    virtual int preprocess(const FrameInfo& frame) = 0;
    virtual int postprocess(rknn_output* outputs,FrameInfo &frame) = 0;

    int init_rknn(const std::string& model_path);
    
    void* preprocessed_data_ = nullptr;

private:
    int init_model(const char* model_path);
    int read_data_from_file(const char *path, char **out_data);
    void dump_tensor_attr(rknn_tensor_attr *attr);

public:
    rknn_app_context_t app_ctx_;
};

#endif