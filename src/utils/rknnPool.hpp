#ifndef RKNNPOOL_H
#define RKNNPOOL_H

#include "threadpool.hpp"
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <vector>

// rknnModel模型类, inputType模型输入类型, outputType模型输出类型
template <typename rknnModel, typename inputType, typename outputType>
class rknnPool {
private:
  int id;
  std::mutex idMtx, queueMtx;
  std::unique_ptr<dpool::ThreadPool> pool;
  std::queue<std::future<outputType>> futs;
  std::vector<std::shared_ptr<rknnModel>> models;

protected:
  int getModelId();

public:
  int threadNum;


  rknnPool();
  int init(const FrameInfo& frame);
  // 模型推理/Model inference
  int put(inputType inputData);
  // 获取推理结果/Get the results of your inference
  int get(outputType &outputData);
  ~rknnPool();
};

template <typename rknnModel, typename inputType, typename outputType>
rknnPool<rknnModel, inputType, outputType>::rknnPool() {
  this->id = 0;
}

template <typename rknnModel, typename inputType, typename outputType>
int rknnPool<rknnModel, inputType, outputType>::init(const FrameInfo& frame) {
  try {
    threadNum = frame.alg_parm.thread_num;
    if (threadNum <= 0) return -1;
    this->pool = std::make_unique<dpool::ThreadPool>(threadNum);
    for (int i = 0; i < this->threadNum; i++){
      if(frame.alg_type == AlgoType::kYolox){
        models.push_back(std::make_shared<rknnModel>());
      }
    }
  } catch (const std::bad_alloc &e) {
    std::cout << "Out of memory: " << e.what() << std::endl;
    return -1;
  }
  // 初始化模型/Initialize the model
  for (int i = 0, ret = 0; i < threadNum; i++) {
    ret = models[i]->init(frame);
    if (ret != 0)
      return ret;
  }
  return 0;
}

template <typename rknnModel, typename inputType, typename outputType>
int rknnPool<rknnModel, inputType, outputType>::getModelId() {
  std::lock_guard<std::mutex> lock(idMtx);
  int modelId = id % threadNum;
  if (id == threadNum) {
    id = 0;
  }
  id++;
  return modelId;
}

template <typename rknnModel, typename inputType, typename outputType>
int rknnPool<rknnModel, inputType, outputType>::put(inputType inputData) {
  futs.push(
      pool->submit(&rknnModel::infer, models[this->getModelId()], inputData));
  return 0;
}

template <typename rknnModel, typename inputType, typename outputType>
int rknnPool<rknnModel, inputType, outputType>::get(outputType &outputData) {
  std::lock_guard<std::mutex> lock(queueMtx);
  if (futs.empty() == true)
    return 1;
  outputData = futs.front().get();
  futs.pop();
  return 0;
}

template <typename rknnModel, typename inputType, typename outputType>
rknnPool<rknnModel, inputType, outputType>::~rknnPool() {
  while (!futs.empty()) {
    outputType temp = futs.front().get();
    futs.pop();
  }
}

#endif