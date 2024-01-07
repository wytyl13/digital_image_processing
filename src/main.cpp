#include "general.h"


/**
 * @Author: weiyutao
 * @Date: 2023-08-28 18:27:34
 * @Parameters: 
 * @Return: 
 * @Description: 注意我们是基于8比特单通道的图像处理的，我们首先来了解下
 * Mat是opencv的基本数据类型，保存图像的，我们可以看到有29个重载的构造器，意味着
 * 我们可以通过任何一种方式去构造我们的数据，
 */
void test() 
{
     // Mat(int rows, int cols, int type, const cv::Scalar &s)
    // CV_8UC1 8bit unsigned char 1 channel.单通道的8比特图像
    // 需要制定图像的宽高
    int width = 640;
    int height = 640;
    cv::Mat singleImage1(width, height, CV_8UC1, cv::Scalar(0));
    cv::Mat singleImage2(width, height, CV_8UC1, cv::Scalar(200));
    cv::Mat singleImage3(width, height, CV_8UC1, cv::Scalar(255));

    // void merge(const cv::Mat *mv, size_t count, cv::OutputArray dst)
    // 因为我们不能使用numpy，我们使用merge去拼接下单通道图像吧
    // 我们需要传入一个图像列表
    // colorImage在这里是一个传出参数
    cv::Mat channels[3] = {singleImage1, singleImage2, singleImage3};
    cv::Mat colorImage;
    cv::merge(channels, 3, colorImage);



    cv::imshow("gray image", colorImage);
    cv::waitKey(0);
}

int main(int argc, char const *argv[])
{
    std::string dark_image = "C:\\Users\\weiyutao\\development_code_2023-01-28\\vscode\\digital_image_processing\\source\\images\\高圆圆4.jpeg";
    Mat image = imread(dark_image);

    Mat inputImage = imread(dark_image, 0);
    Mat outputImage;

    vector<Mat> multiImage;
    cv::Mat noise_image;
    // guassian_noise(inputImage, noise_image, 0.0, 50.0);
    saltPepper(inputImage, noise_image, 8, 100);
    multiImage.push_back(noise_image);
    median_filter_thread(noise_image, outputImage, 3, 4);
    multiImage.push_back(outputImage);
    median_filter_thread(noise_image, outputImage, 11, 4);
    multiImage.push_back(outputImage);
    median_filter_thread(noise_image, outputImage, 15, 4);
    multiImage.push_back(outputImage);
    string str_window = "histogram local equalization";
    multiImshow(str_window, multiImage);
    
    waitKey(0);
    destroyAllWindows();
    system("pause");
    return 0;
}
