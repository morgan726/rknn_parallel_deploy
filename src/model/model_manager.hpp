#ifndef MODEL_MANAGER_H
#define MODEL_MANAGER_H

#include "utils/threadpool.hpp"
#include "model/model.h"
#include "frame.h"
#include <memory>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <vector>

class ModelManager {
public:
    explicit ModelManager(size_t max_threads) 
        : thread_pool_(std::make_unique<dpool::ThreadPool>(max_threads)),
          max_threads_(max_threads) {}

    ModelManager(const ModelManager&) = delete;
    ModelManager& operator=(const ModelManager&) = delete;

    template <typename ModelType>
    int init(const FrameInfo& frame) {
        std::lock_guard<std::mutex> lock(mtx_);
        AlgoType alg_type = ModelType().get_algo_type();

        if (model_cache_.count(alg_type)) {
            return 0;
        }

        std::vector<std::shared_ptr<RknnModelBase>> model_instances;
        try {
            for (size_t i = 0; i < max_threads_; ++i) {
                auto model = std::make_shared<ModelType>();
                int ret = model->init(frame);
                if (ret != 0) return ret;
                model_instances.push_back(model);
            }
        } catch (const std::bad_alloc&) {
            return -1;
        }

        model_cache_[alg_type] = model_instances;
        return 0;
    }

    int put(const FrameInfo& inputData) {
        std::lock_guard<std::mutex> lock(mtx_);
        AlgoType alg_type = inputData.alg_type;

        auto it = model_cache_.find(alg_type);
        if (it == model_cache_.end() || it->second.empty()) {
            return -1;
        }

        size_t model_id = get_next_model_id(alg_type);
        auto model = it->second[model_id];

        auto task = [model, inputData]() -> FrameInfo {
            FrameInfo frame = inputData;
            model->infer(frame);
            return frame;
        };

        result_futs_.emplace(thread_pool_->submit(task));
        return 0;
    }

    int get(FrameInfo& outputData) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (result_futs_.empty()) return 1;

        outputData = result_futs_.front().get();
        result_futs_.pop();
        return 0;
    }

    ~ModelManager() {
        std::lock_guard<std::mutex> lock(mtx_);
        while (!result_futs_.empty()) {
            result_futs_.front().wait();
            result_futs_.pop();
        }
    }

private:
    size_t get_next_model_id(AlgoType alg_type) {
        auto& id = model_id_counter_[alg_type];
        size_t current_id = id;
        id = (id + 1) % model_cache_[alg_type].size();
        return current_id;
    }

private:
    std::unique_ptr<dpool::ThreadPool> thread_pool_;
    size_t max_threads_;

    std::unordered_map<AlgoType, std::vector<std::shared_ptr<RknnModelBase>>> model_cache_;
    std::unordered_map<AlgoType, size_t> model_id_counter_;

    std::queue<std::future<FrameInfo>> result_futs_;
    mutable std::mutex mtx_;
};

#endif
