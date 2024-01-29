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
#include <map>
#include <numeric>

#include <opencv2/core/utility.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
namespace fs = std::filesystem;

bool processing = false;
bool trainig = false;
bool dir_set = false;
bool metrics = false;
std::string dir = "";


namespace fs = std::filesystem;

std::pair<std::map<std::string, std::vector<std::string>>, std::map<std::string, std::map<std::string, std::vector<cv::Rect>>>> parseDirectory(const std::string& path) {
    std::map<std::string, std::vector<std::string>> directories;
    std::map<std::string, std::map<std::string, std::vector<cv::Rect>>> faceRects;

    for (const auto& entry : fs::directory_iterator(path)) {
        if (fs::is_directory(entry)) {
            std::string dirName = entry.path().filename().string();
            std::vector<std::string> subDirs;

            for (const auto& subEntry : fs::directory_iterator(entry.path())) {
                if (fs::is_directory(subEntry)) {
                    subDirs.push_back(subEntry.path().string());
                }
            }

            directories[dirName] = subDirs;
        } else if (entry.path().extension() == ".txt") {
            std::ifstream infile(entry.path());
            std::string line;

            while (std::getline(infile, line)) {
                std::istringstream iss(line);
                std::vector<std::string> tokens;
                std::string token;

                while (std::getline(iss, token, ',')) {
                    tokens.push_back(token);
                }
                if (tokens.size() >= 6) {
                    std::string filename = tokens[0];
                    int x = std::stoi(tokens[2]);
                    int y = std::stoi(tokens[3]);
                    int width = std::stoi(tokens[4]);
                    int height = std::stoi(tokens[5]);
                    cv::Rect rect(cv::Point(x - width / 2, y - height / 2), cv::Size(width, height));
                    faceRects[split(filename, '\\')[0]][split(filename, '\\')[1]].push_back(rect);
                }
            }
        }
    }

    return std::make_pair(directories, faceRects);
}



int parseCmdArgs(int argc, char** argv)
{
    if (argc == 1)
    {
        return EXIT_FAILURE;
    }


    for (int i = 1; i < argc; ++i)
    {
        if (std::string(argv[i]) == "--help" || string(argv[i]) == "/?")
        {
            return EXIT_FAILURE;
        }
        else if (string(argv[i]) == "-mode")
        {
            if (string(argv[i + 1]) == "processing")
            {
                processing = true;
            }
            else if (string(argv[i + 1]) == "training")
            {
                trainig = true;
            }
            else
            {
                std::cout << "Bad -mode flag value" << std::endl;
                return EXIT_FAILURE;
            }

            i++;
        }
        if (std::string(argv[i]) == "-dir")
        {
            dir = string(argv[i + 1]);
            dir_set = true;
        }
        if (std::string(argv[i]) == "-metrics")
        {
            if (processing)
            {
                metrics = true;
            }
            else
            {
                std::cout << "-mode flag should be 'processing' to calculate metrics" << std::endl;
                return EXIT_FAILURE;
            }
        }
    }

    if (!(processing || trainig)) {
        std::cout << "Please, set -mode" << endl;
        return EXIT_FAILURE;
    }

    if (!dir_set) {
        std::cout << "Please, set source directory -dir" << endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

void train(std::string dir)
{
    saveFaceDescriptors(get_file_paths(dir),
    "desc.txt",
    "haarcascade_frontalface_alt.xml");
}

struct RecognitionMetrics {
    double accuracy;
    double FNR;
    double FPR;
};

RecognitionMetrics computeRecognitionMetrics(const std::vector<std::vector<std::string>>& predictions, const std::string& target) {
    int TP = 0, FP = 0;
    int totalPredictions = 0;

    for (const auto& predictionSet : predictions) {
        for (const auto& prediction : predictionSet) {
            if (prediction == target) {
                TP++;
            } else {
                FP++;
            }
            totalPredictions++;
        }
    }

    RecognitionMetrics metrics;
    metrics.accuracy = totalPredictions > 0 ? static_cast<double>(TP) / totalPredictions : 0;
    metrics.FNR = 0;  // FN = 0, так как прогноз всегда делается
    metrics.FPR = totalPredictions > 0 ? static_cast<double>(FP) / totalPredictions : 0;

    return metrics;
}


void process(std::string dir)
{
    auto res = parseDirectory(dir);
    std::map<std::string, std::vector<std::string>> names = res.first;
    std::map<std::string, std::map<std::string, std::vector<cv::Rect>>> target_bboxes = res.second;
    std::vector<double> tpr;
    std::vector<double> fpr;
    std::vector<double> fnr;

    std::vector<double> acc_rec;
    std::vector<double> fpr_rec;
    std::vector<double> fnr_rec;

    // TODO: Добавить ещё массивы и сделать усреднение метрик для recognition

    for (const auto& [key, value] : names){
        for(int i = 0; i< value.size(); i++){
            std::cout << '[' << key << "] - " << value[i] << "; ";
            std::vector<cv::Mat> frames = loadAndSortImages(value[i]);
            std::vector<std::vector<std::string>> names;
            std::vector<std::vector<cv::Rect>> bboxes = detectAndTrackFaces(
                frames,
                "/home/dtsarev/master_of_cv/pthn1/2d_img_proc/final_project/haarcascade_frontalface_alt.xml",
                "/home/dtsarev/master_of_cv/pthn1/2d_img_proc/final_project/desc.txt",
                350,
                names);

            saveVideoFromFrames(frames, "/home/dtsarev/master_of_cv/pthn1/2d_img_proc/final_project/result/" + key +"_"+ std::to_string(i) + ".mp4");

            if(metrics)
            {
                std::vector<std::string> tokens = split(value[i], '/');
                DetectionMetrics metrics = computeDetectionMetrics(target_bboxes[tokens[tokens.size() - 2]][tokens[tokens.size()-1]], bboxes);
                RecognitionMetrics rec_metrics = computeRecognitionMetrics(names, tokens[tokens.size() - 2]);

                acc_rec.push_back(rec_metrics.accuracy);
                fpr_rec.push_back(rec_metrics.FPR);
                fnr_rec.push_back(rec_metrics.FNR);

                tpr.push_back(metrics.TPR);
                fpr.push_back(metrics.FPR);
                fnr.push_back(metrics.FNR);
            }

        }
    }
    std::cout << std::endl << "metrics (detection & tracking)" << std::endl;
    std::cout << std::endl << "TPR: " << std::reduce(tpr.begin(), tpr.end())/tpr.size() << std::endl;
    std::cout << std::endl << "FPR: " << std::reduce(fpr.begin(), fpr.end())/fpr.size() << std::endl;
    std::cout << std::endl << "FNR: " << std::reduce(fnr.begin(), fnr.end())/fnr.size() << std::endl;

    std::cout << std::endl << "metrics (recognition)" << std::endl;
    std::cout << std::endl << "Accuracy: " << std::reduce(acc_rec.begin(), acc_rec.end())/acc_rec.size() << std::endl;
    std::cout << std::endl << "FPR: " << std::reduce(fpr_rec.begin(), fpr_rec.end())/fpr_rec.size() << std::endl;
    std::cout << std::endl << "FNR: " << std::reduce(fnr_rec.begin(), fnr_rec.end())/fnr_rec.size() << std::endl;

}

int main(int argc, char* argv[]) {

    if (parseCmdArgs(argc, argv))
        return EXIT_FAILURE;

    if (processing)
    {
        process(dir);
    }
    else if(trainig)
    {
        train(dir);
    }
    //process("/home/dtsarev/master_of_cv/pthn1/2d_img_proc/final_project/data");
    //train("/home/dtsarev/master_of_cv/pthn1/2d_img_proc/final_project/faces");

    return 0;
}
