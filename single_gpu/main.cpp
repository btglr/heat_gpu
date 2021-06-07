#include "single_gpu.h"
#include "constants.h"
#include <cstdlib>

#if defined WRITE_IMG
    #include "opencv2/opencv.hpp"
#endif

int main(int argc, char *argv[]) {
    double *h_a;
    
    h_a = jacobi();

#if defined WRITE_IMG
    printf("Writing output matrix to two files\n");
    cv::Mat img(HEIGHT, WIDTH, CV_64FC1, h_a);
    cv::Mat colormap_img;
    
    img.convertTo(img, CV_8UC1, 255.0); 
    cv::applyColorMap(img, colormap_img, cv::COLORMAP_JET);
    cv::imwrite("output_color.jpg", colormap_img);
    cv::imwrite("output.jpg", img);
#endif

    free(h_a);

    return 0;
}