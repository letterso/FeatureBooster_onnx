#include "Config.h"
#include "FeatureBoosterOnnxRunner.h"

int main()
{
    cv::Mat img1 = cv::imread(THIS_COM + std::string("/../qualitative/img2/1.jfif"), 0);
    cv::Mat img2 = cv::imread(THIS_COM + std::string("/../qualitative/img2/2.jfif"), 0);

    cv::Ptr<cv::Feature2D> detector = cv::ORB::create(1000, 1.2, 8);

    // orb detect
    std::vector<cv::KeyPoint> key1;
    cv::Mat desc1;
    detector->detectAndCompute(img1, cv::Mat(), key1, desc1);
    std::vector<cv::KeyPoint> key2;
    cv::Mat desc2; // CV8UC1
    detector->detectAndCompute(img2, cv::Mat(), key2, desc2);

    FeatureBoosterOnnxRunner featureBoosterOnnxRunner;
    featureBoosterOnnxRunner.InitOrtEnv(THIS_COM + std::string("/../weights/ORB+Boost-B.onnx"));

    auto time_start = std::chrono::high_resolution_clock::now();

    cv::Mat desc1_boost;
    featureBoosterOnnxRunner.InferenceDescriptor(img1.size(), key1, desc1, desc1_boost);
    cv::Mat desc2_boost;
    featureBoosterOnnxRunner.InferenceDescriptor(img2.size(), key2, desc2, desc2_boost);

    auto time_end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
    std::cout << "cost time : " << diff << std::endl;

    // 匹配
    std::vector<std::vector<cv::DMatch>> knn_matches;
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    matcher.knnMatch(desc1_boost, desc2_boost, knn_matches, 2);

    // 筛选
    const float ratio_thresh = 0.75f;
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    // 绘制匹配
    cv::Mat img_matches;
    cv::drawMatches(img1, key1, img2, key2, good_matches, img_matches, cv::Scalar::all(-1),
                    cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imshow("match_img", img_matches);
    cv::waitKey(0);
}