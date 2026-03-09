**1.下载 onnx-runtime-gpu 1.24.3（CUDA 13.1）**

解压并配置环境变量

"ONNXRUNTIME_LIB" = "D:\Environment\onnxruntime\onnxruntime-win-x64-gpu-1.24.3"

注意下载对应`CUDA`版本的`ONNX`运行时



**2.下载 opencv-4.12.0**

解压并配置环境变量

"OPENCV_LIB_412" = "D:\Environment\opencv\opencv-4.12.0"



**3.安装[CUDA Toolkit](https://developer.nvidia.com/cuda/toolkit)**

注意这里安装的`CUDA`版本跟`ONNX`运行时对应



**4.安装[cuDNN](https://developer.nvidia.com/cudnn-downloads)**



**5.修改生成事件**

项目`onnx_ocr_test`右键属性，生成事件 → 生成后事件

编辑 “复制 cuDNN 9 的所有相关 DLL”命令行，项目编译自动把依赖`DLL`复制到目标文件夹，以便脱离编译环境运行

修改目录为当前安装得到 `cuDNN`运行时目录

例如：

```
xcopy /y /d "$(ProgramFiles)\NVIDIA\CUDNN\v9.19\bin\13.1\x64\cudnn*.dll" "$(TargetDir)"
```



**6.以上步骤完毕可直接编译运行，运行环境 Win10+**



**其他**

+ onnx模型参数查看 [netron.app](https://netron.app/)，直接上传 onnx 文件
