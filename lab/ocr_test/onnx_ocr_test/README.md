**1.下载 onnx-runtime-gpu 1.23.2**

解压到 "onnxruntime-win-x64-gpu-1.23.2" 文件夹



**2.下载 opencv-4.10.0**

解压到 "opencv-4.10.0" 文件夹



部分目录树：

```
├─onnxruntime-win-x64-gpu-1.23.2
│  ├─include
│  │  └─core
│  │      └─providers
│  │          └─cuda
│  └─lib
├─onnx_ocr_test
│  └─x64
│      └─Debug
│          └─onnx_ocr_test.tlog
├─opencv-4.10.0
│  └─opencv
│      ├─build
│      │  ├─bin
```



**3.安装[CUDA Toolkit](https://developer.nvidia.com/cuda/toolkit)**



**4.安装[cuDNN](https://developer.nvidia.com/cudnn-downloads)**



**5.以上步骤完毕可直接编译运行，运行环境 Win10+**



**其他**

+ onnx模型参数查看 [netron.app](https://netron.app/)，直接上传 onnx 文件
