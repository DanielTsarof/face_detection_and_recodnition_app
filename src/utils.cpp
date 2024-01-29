#include "utils.h"

#include <vector>

#include <iostream>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <algorithm>

 #include <opencv2/core/utility.hpp>
 #include <opencv2/highgui.hpp>

#include "xfeatures2d.hpp"
#include "nonfree.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

namespace fs = std::filesystem;

cv::Mat cropImage(const cv::Mat& image, const cv::Rect& rect) {
    if (image.empty()) {
        std::cerr << "The image is empty." << std::endl;
        return cv::Mat();
    }

    if (rect.x >= 0 && rect.y >= 0 && rect.width > 0 && rect.height > 0 
        && rect.x + rect.width <= image.cols && rect.y + rect.height <= image.rows) {
        return image(rect).clone(); // Возвращаем клон подматрицы, чтобы избежать зависимости от исходного изображения
    } else {
        std::cerr << "Invalid rectangle dimensions." << std::endl;
        return cv::Mat();
    }
}

std::vector<FaceDescriptor> loadDescriptors(const std::string& descriptorDatabasePath) {
    std::vector<FaceDescriptor> descriptors;
    std::ifstream infile(descriptorDatabasePath);
    std::string line;

    while (std::getline(infile, line)) {
        std::stringstream lineStream(line);
        std::string cell;

        FaceDescriptor fd;
        std::getline(lineStream, fd.name, ';');

        std::vector<float> descriptorValues;
        while (std::getline(lineStream, cell, ';')) {
            std::stringstream cellStream(cell);
            std::string value;

            while (std::getline(cellStream, value, ',')) {
                if (!value.empty()) {
                    descriptorValues.push_back(std::stof(value));
                }
            }
        }

        if (!descriptorValues.empty()) {
            fd.descriptors = cv::Mat(descriptorValues.size() / 128, 128, CV_32F, descriptorValues.data()).clone();
            descriptors.push_back(fd);
        }
    }

    return descriptors;
}

// Функция распознавания лиц
std::string recognizeFace(const cv::Mat& face, const std::vector<FaceDescriptor>& knownDescriptors, double threshold) {
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    sift->detectAndCompute(face, cv::Mat(), keypoints, descriptors);

    if (descriptors.empty()) {
        return "undefined";
    }

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    double minDist = DBL_MAX;
    std::string bestMatch = "undefined";

    for (const auto& known : knownDescriptors) {
        if (!known.descriptors.empty()) {
            std::vector<std::vector<cv::DMatch>> knnMatches;
            matcher->knnMatch(known.descriptors, descriptors, knnMatches, 1);

            if (!knnMatches.empty() && !knnMatches[0].empty()) {
                double dist = knnMatches[0][0].distance;
                if (dist < minDist && dist < threshold) {
                    minDist = dist;
                    bestMatch = known.name;
                }
            }
        }
    }

    return bestMatch;
}


std::string extractFilename(const std::string& path) {
    // Находим последний слэш в пути
    size_t lastSlash = path.find_last_of("/\\");
    std::string filename = lastSlash == std::string::npos ? path : path.substr(lastSlash + 1);

    // Удаляем расширение файла
    size_t dotPosition = filename.rfind('.');
    if (dotPosition != std::string::npos) {
        filename.erase(dotPosition);
    }

    return filename;
}

void saveFaceDescriptors(const std::vector<std::string>& filePaths, const std::string& outputFile, const std::string& faceCascadePath) {
    std::ofstream out(outputFile);
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    cv::CascadeClassifier faceCascade;

    if (!faceCascade.load(faceCascadePath)) {
        std::cerr << "Error loading face cascade." << std::endl;
        return;
    }
    
    for (const auto& path : filePaths) {
        cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            std::cout << "Could not read the image: " << path << std::endl;
            continue;
        }

        std::vector<cv::Rect> faces;
        faceCascade.detectMultiScale(image, faces);

        if (faces.size() == 1) {
            cv::Mat face = image(faces[0]); // Вырезаем лицо

            std::vector<cv::KeyPoint> keypoints;
            cv::Mat descriptors;
            sift->detectAndCompute(face, cv::Mat(), keypoints, descriptors);

            if (!descriptors.empty()) {
                std::string filename = extractFilename(path);
                out << filename << ";"; // Сохранение имени файла
                for (int i = 0; i < descriptors.rows; ++i) {
                    for (int j = 0; j < descriptors.cols; ++j) {
                        out << descriptors.at<float>(i, j);
                        if (j < descriptors.cols - 1) out << ",";
                    }
                    out << ";"; // Разделитель строк дескриптора
                }
                out << std::endl; // Сохранение дескрипторов
            }
        } else {
            std::cout << "No face detected or multiple faces in the image: " << path << std::endl;
        }
    }

    out.close();
}

std::vector<std::string> get_file_paths(std::string dir_path)
{
    std::vector<std::string> res;
    for (const auto & entry : fs::directory_iterator(dir_path))
    {
       res.push_back(entry.path());
    }
    return res;
}

// Функция для извлечения числовой части из имени файла
double getNumberFromFilename(const std::string& filename) {
    std::string number_part = filename.substr(filename.find('.') + 1);
    number_part = number_part.substr(0, number_part.find('.'));
    std::istringstream iss(number_part);
    double number;
    iss >> number;
    return number;
}

std::vector<cv::Mat> loadAndSortImages(const std::string& directoryPath) {
    std::vector<std::pair<double, cv::Mat>> imagePairs;

    for (const auto& entry : fs::directory_iterator(directoryPath)) {
        cv::Mat image = cv::imread(entry.path().string());
        if (!image.empty()) {
            double number = getNumberFromFilename(entry.path().filename().string());
            imagePairs.emplace_back(number, image);
        }
    }

    std::sort(imagePairs.begin(), imagePairs.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    std::vector<cv::Mat> sortedImages;
    for (const auto& pair : imagePairs) {
        sortedImages.push_back(pair.second);
    }

    return sortedImages;
}

void saveVideoFromFrames(const std::vector<cv::Mat>& frames, const std::string& outputPath) {
    if (frames.empty()) {
        std::cerr << "The frames vector is empty." << std::endl;
        return;
    }

    int frameWidth = 512;
    int frameHeight = 384;
    double fps = 30.0; // Частота кадров

    cv::VideoWriter videoWriter;
    int fourcc = cv::VideoWriter::fourcc('M', 'P', '4', 'V'); // Кодек для mp4

    videoWriter.open(outputPath, fourcc, fps, cv::Size(frameWidth, frameHeight), true);

    if (!videoWriter.isOpened()) {
        std::cerr << "Could not open the output video file for write." << std::endl;
        return;
    }

    for (const auto& frame : frames) {
        cv::Mat resizedFrame;
        cv::resize(frame, resizedFrame, cv::Size(frameWidth, frameHeight));
        videoWriter.write(resizedFrame);
    }

    videoWriter.release();
    std::cout <<"videofile saved" << std::endl;
}

// Структура для хранения информации о трекере и имени
struct TrackedFace {
    cv::Ptr<cv::Tracker> tracker;
    cv::Rect boundingBox;
    std::string name;
};


bool isOverlap(const cv::Rect& rect1, const cv::Rect& rect2) {
    return ((rect1 & rect2).area() > 0);
}

std::vector<std::vector<cv::Rect>> detectAndTrackFaces(std::vector<cv::Mat>& frames, const std::string& faceCascadePath, const std::string& descriptorDatabasePath, double threshold, std::vector<std::vector<std::string>>& names) {
    std::vector<std::vector<cv::Rect>> allFaces;
    cv::CascadeClassifier faceDetector;
    std::vector<FaceDescriptor> knownDescriptors = loadDescriptors(descriptorDatabasePath);

    if (!faceDetector.load(faceCascadePath)) {
        std::cerr << "Error loading face cascade." << std::endl;
        return allFaces;
    }

    std::vector<TrackedFace> trackedFaces;
    int frameCounter = 0;

    for (cv::Mat &frame : frames) {
        std::vector<cv::Rect> facesInCurrentFrame;

        if (frameCounter % 10 == 0) {
            // Очистка старых трекеров
            trackedFaces.clear();

            std::vector<cv::Rect> detectedFaces;
            faceDetector.detectMultiScale(frame, detectedFaces);

            std::vector<std::string> faces = {};
            for (const auto& face : detectedFaces) {
                cv::Ptr<cv::Tracker> tracker = cv::TrackerKCF::create();
                tracker->init(frame, face);

                cv::Mat faceROI = frame(face);
                std::string name = recognizeFace(faceROI, knownDescriptors, threshold);
                faces.push_back(name);

                trackedFaces.push_back({tracker, face, name});
            }
            names.push_back(faces);
        }

        for (auto& trackedFace : trackedFaces) {
            if (trackedFace.tracker->update(frame, trackedFace.boundingBox)) {
                cv::rectangle(frame, trackedFace.boundingBox, cv::Scalar(255, 0, 0), 2);
                cv::putText(frame, trackedFace.name, cv::Point(trackedFace.boundingBox.x, trackedFace.boundingBox.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
                facesInCurrentFrame.push_back(trackedFace.boundingBox);
            }
        }

        allFaces.push_back(facesInCurrentFrame);
        frameCounter++;
    }

    return allFaces;
}

std::vector<std::string> split(const std::string& str, char delim) {
    std::vector<std::string> tokens;
    std::istringstream strStream(str);
    std::string token;

    while (std::getline(strStream, token, delim)) {
        tokens.push_back(token);
    }

    return tokens;
}

double computeIoU(const cv::Rect& rect1, const cv::Rect& rect2) {
    cv::Rect intersection = rect1 & rect2;
    double intersectionArea = intersection.area();
    double unionArea = rect1.area() + rect2.area() - intersectionArea;
    return intersectionArea / unionArea;
}

DetectionMetrics computeDetectionMetrics(const std::vector<cv::Rect>& targets, const std::vector<std::vector<cv::Rect>>& predictions) {
    int TP = 0, FP = 0, FN = 0;

    for (const auto& target : targets) {
        bool isTP = false;
        for (const auto& predictionFrame : predictions) {
            for (const auto& prediction : predictionFrame) {
                if (computeIoU(target, prediction) >= 0.8) {
                    TP++;
                    isTP = true;
                    break;
                }
            }
            if (isTP) break;
        }
        if (!isTP) {
            FN++;
        }
    }

    for (const auto& predictionFrame : predictions) {
        for (const auto& prediction : predictionFrame) {
            bool isFP = true;
            for (const auto& target : targets) {
                if (computeIoU(prediction, target) >= 0.8) {
                    isFP = false;
                    break;
                }
            }
            if (isFP) {
                FP++;
            }
        }
    }

    DetectionMetrics metrics;
    metrics.TPR = TP / double(TP + FN);
    metrics.FNR = FN / double(TP + FN);
    metrics.FPR = FP / double(FP + TP);

    return metrics;
}
