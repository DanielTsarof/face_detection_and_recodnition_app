
#ifndef UTILS_H
#define UTILS_H
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>


// Функция для извлечения числовой части из имени файла
double getNumberFromFilename(const std::string& filename);

// Функция для обработки директории с кадрами видео
std::vector<cv::Mat> loadAndSortImages(const std::string& directoryPath);

// Функция, которая сохраняет видео в формате mp4
void saveVideoFromFrames(const std::vector<cv::Mat>& frames, const std::string& outputPath);

std::vector<std::vector<cv::Rect>> detectAndTrackFaces(
    std::vector<cv::Mat>& frames,
     const std::string& faceCascadePath,
      const std::string& descriptorDatabasePath,
       double threshold,
       std::vector<std::vector<std::string>>& names);

std::vector<std::string> get_file_paths(std::string dir_path);

void saveFaceDescriptors(const std::vector<std::string>& filePaths, const std::string& outputFile, const std::string& faceCascadePath);

struct FaceDescriptor {
    std::string name;
    cv::Mat descriptors;
};

// Загрузка дескрипторов из файла
std::vector<FaceDescriptor> loadDescriptors(const std::string& descriptorDatabasePath);

// Функция распознавания лиц
std::string recognizeFace(const cv::Mat& face, const std::vector<FaceDescriptor>& knownDescriptors, double threshold);

cv::Mat cropImage(const cv::Mat& image, const cv::Rect& rect);

std::vector<std::string> split(const std::string& str, char delim);

struct DetectionMetrics {
    double TPR;
    double FNR;
    double FPR;
};

DetectionMetrics computeDetectionMetrics(const std::vector<cv::Rect>& targets, const std::vector<std::vector<cv::Rect>>& predictions);

double computeIoU(const cv::Rect& rect1, const cv::Rect& rect2);

#endif
