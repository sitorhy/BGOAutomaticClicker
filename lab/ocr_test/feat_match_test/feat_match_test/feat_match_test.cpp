#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>
#include <chrono>

using namespace std;
using namespace cv;

// 定义特征数据结构
struct HeroTemplate {
    string name;
    Mat descriptors;
};

// 全局变量
vector<HeroTemplate> library;

// ORB 的全称是 Oriented FAST and Rotated BRIEF。
// ORB - (Oriented Fast and Rotated BRIEF)
// 算法是基于FAST特征检测与BRIEF特征描述子匹配实现
// 大致原理：它结合了两种算法。首先利用 FAST 算法快速找到图像中具有代表性的“特征点”（如角点、边缘点）；然后利用 BRIEF 算法将这些点周围的像素信息编码成一个二进制字符串（描述子）。
// 核心优势：它是为了替代收费的 SIFT / SURF 而设计的。它最大的特点是具有旋转不变性（Rotation Invariance）和抗噪性。即使你的从者头像在屏幕上稍微歪了一点，或者光影变了，它依然能通过特征点之间的相对关系匹配出来。

// 其实只需要几十个关键特征点（眼睛、发尖、饰品角点）就能锁定唯一性。
// Ptr<ORB> orb = ORB::create(1000); // 表示尝试从图中找最明显的 1000 个点。实际上降到100即可
Ptr<ORB> orb = ORB::create(100);

// 模拟将头像特征存入内存
void addToLibrary(const string& name, const string& path) {
    Mat img = imread(path, IMREAD_GRAYSCALE);
    if (img.empty()) return;

    vector<KeyPoint> kps;
    Mat des;
    orb->detectAndCompute(
        img,    // 输入图：通常为 8 位灰度图。
        noArray(), // 掩码：指定在图的哪个区域找。如果不限制，传入 cv::noArray() 或 cv::Mat() 即可。
        kps, // 输出点：std::vector<cv::KeyPoint> 类型。存储点的坐标、角度、直径等信息。
        des // 输出矩阵：cv::Mat 类型。每一行代表一个特征点的二进制“指纹”。如果提取 1000 个点，它就是一个 1000 行的矩阵。
    );

    if (!des.empty()) {
        library.push_back({ name, des });
        cout << "[Library] 已加载从者: " << name << " (特征点: " << kps.size() << ")" << endl;
    }
}

void matchHero(const Mat& scene) {
    if (scene.empty()) return;

    Mat grayScene;
    if (scene.channels() == 3) cvtColor(scene, grayScene, COLOR_BGR2GRAY);
    else grayScene = scene;

    // 提取当前画面的特征
    auto t1 = chrono::high_resolution_clock::now();
    vector<KeyPoint> sceneKps;
    Mat sceneDes;
    orb->detectAndCompute(grayScene, noArray(), sceneKps, sceneDes);

    if (sceneDes.empty()) return;

    // 使用 BFMatcher + NORM_HAMMING (针对 ORB 的二进制特征)
    BFMatcher matcher(NORM_HAMMING); // 通过 BFMatcher（暴力匹配器）计算 A 和 B 中哪些指纹最接近。
    string bestMatch = "None";
    int maxGoodMatches = 0;

    // 遍历内存中的特征库
    for (const auto& hero : library) {
        vector<vector<DMatch>> knnMatches;

        // knnMatch 的全称是 k-Nearest Neighbors Match（k-最近邻匹配）。
        matcher.knnMatch(
            sceneDes, 
            hero.descriptors, 
            knnMatches, 
            2 // 寻找每个特征点的最近 2 个匹配点
        );

        // Lowe's Ratio Test 筛选优秀匹配点
        int goodMatches = 0;
        for (size_t i = 0; i < knnMatches.size(); i++) {
            // 如果最近邻的距离不到次近邻的 75%，才认为这是一个“好点”
            if (knnMatches[i][0].distance < 0.75 * knnMatches[i][1].distance) {
                goodMatches++;
            }
        }

        if (goodMatches > maxGoodMatches) {
            maxGoodMatches = goodMatches;
            bestMatch = hero.name;
        }
    }

    auto t2 = chrono::high_resolution_clock::now();
    double ms = chrono::duration<double, milli>(t2 - t1).count();

    cout << "------------------------------------" << endl;
    cout << "匹配结果: " << bestMatch << " (有效匹配点: " << maxGoodMatches << ")" << endl;
    cout << "特征提取与检索耗时: " << ms << " ms" << endl;
}

int main() {
    // 应该用头像建库，这里忽略了，匹配复杂度是 O(1)所以都是差不都的
    addToLibrary("C呆-3破", "tmpl_01.png");
    addToLibrary("RBA", "tmpl_02.png");
    addToLibrary("CBA", "tmpl_03.png");
    addToLibrary("孔明", "tmpl_04.png");
    addToLibrary("C呆-0破", "tmpl_05.png");
    addToLibrary("C呆-3破-头像", "tmpl_06.png");


    Mat test1 = imread("test_01.png"); // C呆-0破
    Mat test2 = imread("test_02.png");  // CBA
    Mat test3 = imread("test_03.png");  // C呆-3破
    Mat test4 = imread("test_04.png");  // C呆-3破

    if (!test1.empty()) {
        matchHero(test1);
    }

    if (!test2.empty()) {
        matchHero(test2);
    }

    if (!test3.empty()) {
        matchHero(test3);
    }

    if (!test4.empty()) {
        matchHero(test4);
    }

    system("pause");
    return 0;
}