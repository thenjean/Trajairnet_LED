#include <onnxruntime_cxx_api.h>

#include <iostream>
#include <numeric>
#include <vector>

int64_t product(const std::vector<int64_t>& v) {
  return std::accumulate(v.begin(), v.end(), int64_t{1}, std::multiplies<int64_t>());
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model.onnx>" << std::endl;
    return 1;
  }

  const char* model_path = argv[1];

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "trajairnet");
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  Ort::Session session(env, model_path, session_options);
  Ort::AllocatorWithDefaultOptions allocator;

  // Keep these shapes consistent with export_onnx.py dummy export settings.
  std::vector<int64_t> obs_shape = {1, 7, 3, 11};
  std::vector<int64_t> priors_shape = {1, 7, 3, 12, 3};

  std::vector<float> obs_data(product(obs_shape), 0.0f);
  std::vector<float> priors_data(product(priors_shape), 0.0f);

  auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value obs_tensor = Ort::Value::CreateTensor<float>(
      mem_info, obs_data.data(), obs_data.size(), obs_shape.data(), obs_shape.size());
  Ort::Value priors_tensor = Ort::Value::CreateTensor<float>(
      mem_info, priors_data.data(), priors_data.size(), priors_shape.data(), priors_shape.size());

  const char* input_names[] = {"obs_traj", "route_priors"};
  const char* output_names[] = {"generated_y"};

  std::array<Ort::Value, 2> inputs = {std::move(obs_tensor), std::move(priors_tensor)};
  auto outputs = session.Run(Ort::RunOptions{nullptr},
                             input_names,
                             inputs.data(),
                             inputs.size(),
                             output_names,
                             1);

  auto out_info = outputs[0].GetTensorTypeAndShapeInfo();
  auto out_shape = out_info.GetShape();
  std::cout << "Inference done. Output shape: [";
  for (size_t i = 0; i < out_shape.size(); ++i) {
    std::cout << out_shape[i] << (i + 1 < out_shape.size() ? ", " : "");
  }
  std::cout << "]\n";
  return 0;
}
