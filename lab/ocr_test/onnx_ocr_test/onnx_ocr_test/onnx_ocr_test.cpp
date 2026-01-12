#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <fstream>
#include <chrono> // 用于耗时计算

using namespace std;


// 加载字典文件
std::vector<std::string> LoadDict(const std::string& dict_path) {
	std::vector<std::string> dict;
	dict.push_back("blank");
	std::ifstream file(dict_path);
	if (!file.is_open()) {
		std::cerr << "[错误] 找不到字典文件: " << dict_path << std::endl;
		return dict;
	}
	std::string line;
	while (std::getline(file, line)) {
		if (!line.empty() && line.back() == '\r') line.pop_back();
		dict.push_back(line);
	}
	file.close();
	return dict;
}

// 预处理逻辑
void Preprocess(const cv::Mat& src, std::vector<float>& input_tensor_values, int64_t width, int64_t height) {
	cv::Mat resized, float_img;
	cv::resize(src, resized, cv::Size((int)width, (int)height));
	cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
	resized.convertTo(float_img, CV_32FC3, 1.0 / 127.5, -1.0);

	int size = (int)(height * width);
	float* ptr = (float*)float_img.data;
	for (int c = 0; c < 3; ++c) {
		for (int i = 0; i < size; ++i) {
			input_tensor_values[c * size + i] = ptr[i * 3 + c];
		}
	}
}

void test_ocr(Ort::Session& session, const std::vector<std::string>& dict, const std::string& img_path) {
	try {
		
		// 加载图片与预处理
		cv::Mat img = cv::imread(img_path);
		if (img.empty()) return;

		int64_t h = 48;
		int64_t w = 320;
		std::vector<float> input_data(1 * 3 * h * w);
		Preprocess(img, input_data, w, h);

		auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		std::vector<int64_t> shape = { 1, 3, h, w };
		Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem, input_data.data(), input_data.size(), shape.data(), 4);

		const char* in_names[] = { "x" };
		const char* out_names[] = { "softmax_11.tmp_0" };

		// --- 纯 GPU 推理计时开始 ---
		auto infer_start = chrono::high_resolution_clock::now();

		auto outputs = session.Run(Ort::RunOptions{ nullptr }, in_names, &input_tensor, 1, out_names, 1);

		auto infer_end = chrono::high_resolution_clock::now();
		// --- 纯 GPU 推理计时结束 ---

		// 后处理解码
		float* raw = outputs[0].GetTensorMutableData<float>();
		auto out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
		int seq_len = (int)out_shape[1];
		int dict_size = (int)out_shape[2];

		
		std::string result = "";
		int last_idx = -1;

		for (int i = 0; i < seq_len; ++i) {
			float* scores = raw + (i * dict_size);
			int max_idx = (int)(std::max_element(scores, scores + dict_size) - scores);
			if (max_idx > 0 && max_idx != last_idx) {
				if (max_idx < dict.size()) {
					result.append(dict[max_idx]);
				}
			}
			last_idx = max_idx;
		}

		

		// 计算耗时 (毫秒)
		double infer_ms = chrono::duration<double, milli>(infer_end - infer_start).count();

		std::cout << "\n------------------------------------" << std::endl;
		std::cout << "识别图片: " << img_path << std::endl;
		std::cout << "结果：" << result << std::endl;
		std::cout << "纯引擎推理耗时: " << infer_ms << " ms" << std::endl;
		std::cout << "------------------------------------" << std::endl;

	}
	catch (const std::exception& e) {
		std::cout << "出错: " << e.what() << std::endl;
	}
}

void test_ocr_images(const std::wstring& model_path) {
	std::vector<std::string> dict = LoadDict("ppocr_keys_v1.txt");

	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "FGO_OCR");
	Ort::SessionOptions options;
	OrtCUDAProviderOptions cuda_options;
	int target_gpu_id = 0;
	cuda_options.device_id = target_gpu_id;

	try {
		options.AppendExecutionProvider_CUDA(cuda_options);
		cout << "[引擎] 尝试在 GPU 设备 " << target_gpu_id << " 上启动 CUDA..." << endl;
	}
	catch (...) {
		cout << "[警告] 设备 " << target_gpu_id << " 启动 CUDA 失败，尝试默认设备 0" << endl;
		cuda_options.device_id = 0;
		options.AppendExecutionProvider_CUDA(cuda_options);
	}

	Ort::Session session(env, model_path.c_str(), options);

	// --- 检查 CUDA 是否真的被 Session 接受 ---
	std::vector<std::string> available_providers = Ort::GetAvailableProviders();
	bool has_cuda = false;
	for (const auto& p : available_providers) {
		if (p == "CUDAExecutionProvider") has_cuda = true;
		std::cout << "[系统支持的加速器]: " << p << std::endl;
	}

	if (!has_cuda) {
		std::cout << "[警告] 你的 ONNX Runtime 库根本不支持 CUDA，请检查是否安装的是 GPU 版 NuGet/库" << std::endl;
	}

	// 数字截图需要精准，排除文字或干扰线，例如血条跟数字最好不要一起出现
	test_ocr(session, dict, "ocr_test1.png");
	test_ocr(session, dict, "ocr_test2.png");
	test_ocr(session, dict, "ocr_test3.png");
	test_ocr(session, dict, "ocr_test4.png");
	test_ocr(session, dict, "ocr_test5.png");
}


int main() {
	test_ocr_images(L"ch_PP-OCRv4_rec_infer.onnx");
	system("PAUSE");
	return 0;
}