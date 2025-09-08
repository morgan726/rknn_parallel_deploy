#include "rknn.h"

#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstring>

#include "RgaUtils.h"
#include "im2d.h"
#include "rga.h"

RknnModelBase::~RknnModelBase() {
  if (app_ctx_.rknn_ctx != 0) rknn_destroy(app_ctx_.rknn_ctx);
  if (app_ctx_.input_attrs) free(app_ctx_.input_attrs);
  if (app_ctx_.output_attrs) free(app_ctx_.output_attrs);
}

int RknnModelBase::init_rknn(const std::string &model_path) { return init_model(model_path.c_str(), &app_ctx_); }
int RknnModelBase::infer(FrameInfo &frame) {
  if (preprocess(frame) != 0) return -1;
  rknn_input inputs[app_ctx_.io_num.n_input];
  rknn_output outputs[app_ctx_.io_num.n_output];
  memset(inputs, 0, sizeof(inputs));
  memset(outputs, 0, sizeof(outputs));
  inputs[0].index = 0;
  inputs[0].type = RKNN_TENSOR_UINT8;
  inputs[0].fmt = RKNN_TENSOR_NHWC;
  inputs[0].pass_through = 0;
  inputs[0].size = app_ctx_.model_width * app_ctx_.model_height * app_ctx_.model_channel;
  inputs[0].buf = preprocessed_data_;

  int ret = rknn_inputs_set(app_ctx_.rknn_ctx, app_ctx_.io_num.n_input, inputs);
  if (ret != RKNN_SUCC) return ret;

  for (int i = 0; i < app_ctx_.io_num.n_output; i++) outputs[i].want_float = app_ctx_.is_quant ? 0 : 1;

  ret = rknn_run(app_ctx_.rknn_ctx, NULL);
  if (ret != RKNN_SUCC) return ret;

  ret = rknn_outputs_get(app_ctx_.rknn_ctx, app_ctx_.io_num.n_output, outputs, NULL);
  if (ret != RKNN_SUCC) return ret;

  ret = postprocess(outputs, frame);
  rknn_outputs_release(app_ctx_.rknn_ctx, app_ctx_.io_num.n_output, outputs);
  return ret;
}

void RknnModelBase::dump_tensor_attr(rknn_tensor_attr *attr) {
  printf(
      "  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
      "zp=%d, scale=%f\n",
      attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3], attr->n_elems,
      attr->size, get_format_string(attr->fmt), get_type_string(attr->type), get_qnt_type_string(attr->qnt_type),
      attr->zp, attr->scale);
}

int RknnModelBase::read_data_from_file(const char *path, char **out_data) {
  FILE *fp = fopen(path, "rb");
  if (fp == NULL) {
    printf("fopen %s fail!\n", path);
    return -1;
  }
  fseek(fp, 0, SEEK_END);
  int file_size = ftell(fp);
  char *data = (char *)malloc(file_size + 1);
  data[file_size] = 0;
  fseek(fp, 0, SEEK_SET);
  if (file_size != fread(data, 1, file_size, fp)) {
    printf("fread %s fail!\n", path);
    free(data);
    fclose(fp);
    return -1;
  }
  if (fp) {
    fclose(fp);
  }
  *out_data = data;
  return file_size;
}

int RknnModelBase::init_model(const char *model_path, rknn_app_context_t *app_ctx) {
  int ret;
  int model_len = 0;
  char *model;
  rknn_context ctx = 0;

  // Load RKNN Model
  model_len = read_data_from_file(model_path, &model);
  if (model == NULL) {
    printf("load_model fail!\n");
    return -1;
  }

  ret = rknn_init(&ctx, model, model_len, 0, NULL);
  free(model);
  if (ret < 0) {
    printf("rknn_init fail! ret=%d\n", ret);
    return -1;
  }

  // Get Model Input Output Number
  rknn_input_output_num io_num;
  ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret != RKNN_SUCC) {
    printf("rknn_query fail! ret=%d\n", ret);
    return -1;
  }
  printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

  // Get Model Input Info
  printf("input tensors:\n");
  rknn_tensor_attr input_native_attrs[io_num.n_input];
  memset(input_native_attrs, 0, sizeof(input_native_attrs));
  for (int i = 0; i < io_num.n_input; i++) {
    input_native_attrs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &(input_native_attrs[i]), sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      printf("rknn_query fail! ret=%d\n", ret);
      return -1;
    }
    dump_tensor_attr(&(input_native_attrs[i]));
  }

  // default input type is int8 (normalize and quantize need compute in outside)
  // if set uint8, will fuse normalize and quantize to npu
  input_native_attrs[0].type = RKNN_TENSOR_UINT8;
  app_ctx->input_mems[0] = rknn_create_mem(ctx, input_native_attrs[0].size_with_stride);

  // Set input tensor memory
  ret = rknn_set_io_mem(ctx, app_ctx->input_mems[0], &input_native_attrs[0]);
  if (ret < 0) {
    printf("input_mems rknn_set_io_mem fail! ret=%d\n", ret);
    return -1;
  }

  // Get Model Output Info
  printf("output tensors:\n");
  rknn_tensor_attr output_native_attrs[io_num.n_output];
  memset(output_native_attrs, 0, sizeof(output_native_attrs));
  for (int i = 0; i < io_num.n_output; i++) {
    output_native_attrs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &(output_native_attrs[i]), sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      printf("rknn_query fail! ret=%d\n", ret);
      return -1;
    }
    dump_tensor_attr(&(output_native_attrs[i]));
  }

  // Set output tensor memory
  for (uint32_t i = 0; i < io_num.n_output; ++i) {
    app_ctx->output_mems[i] = rknn_create_mem(ctx, output_native_attrs[i].size_with_stride);
    ret = rknn_set_io_mem(ctx, app_ctx->output_mems[i], &output_native_attrs[i]);
    if (ret < 0) {
      printf("output_mems rknn_set_io_mem fail! ret=%d\n", ret);
      return -1;
    }
  }

  // Set to context
  app_ctx->rknn_ctx = ctx;

  // TODO
  if (output_native_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC &&
      output_native_attrs[0].type == RKNN_TENSOR_INT8) {
    app_ctx->is_quant = true;
  } else {
    app_ctx->is_quant = false;
  }

  rknn_tensor_attr input_attrs[io_num.n_input];
  memset(input_attrs, 0, sizeof(input_attrs));
  for (int i = 0; i < io_num.n_input; i++) {
    input_attrs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      printf("rknn_query fail! ret=%d\n", ret);
      return -1;
    }
  }

  rknn_tensor_attr output_attrs[io_num.n_output];
  memset(output_attrs, 0, sizeof(output_attrs));
  for (int i = 0; i < io_num.n_output; i++) {
    output_attrs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      printf("rknn_query fail! ret=%d\n", ret);
      return -1;
    }
  }

  app_ctx->io_num = io_num;
  app_ctx->input_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
  memcpy(app_ctx->input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
  app_ctx->output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
  memcpy(app_ctx->output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

  app_ctx->input_native_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
  memcpy(app_ctx->input_native_attrs, input_native_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
  app_ctx->output_native_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
  memcpy(app_ctx->output_native_attrs, output_native_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

  if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
    printf("model is NCHW input fmt\n");
    app_ctx->model_channel = input_attrs[0].dims[1];
    app_ctx->model_height = input_attrs[0].dims[2];
    app_ctx->model_width = input_attrs[0].dims[3];
  } else {
    printf("model is NHWC input fmt\n");
    app_ctx->model_height = input_attrs[0].dims[1];
    app_ctx->model_width = input_attrs[0].dims[2];
    app_ctx->model_channel = input_attrs[0].dims[3];
  }
  printf("model input height=%d, width=%d, channel=%d\n", app_ctx->model_height, app_ctx->model_width,
         app_ctx->model_channel);

  return 0;
}