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
    // 读取一张图像，0表示以灰度值模式读取
    Mat image = imread("c:/users/80521/desktop/dark.webp");

    // 我们现在来构建灰度直方图均衡化
    // 首先需要获取灰度图像的直方图，也即图像像素值的分布
    // double *distribution = getDistribution(image);
    // 我们打印下这个指针地址试试，应该是p好久没用过了
    // 注意看这个地址是错误的。我们排查下。因为这个地址是不存在，说明这个distribution
    // 没有被返回。问题在这个函数中。
    // 我们打印下该分布
    //这个并不是分布，我们看下哪里错误了 
    // for (int i = 0; i < 256; i++)
    // {
    //     // printf("%ul", distribution[i]);
    //     cout << distribution[i];
    // }
    
    // 直方图均衡化
    // Mat outputImage;
    // histogram_equalization(image, outputImage);

    // // 我现在想要同时展示两张灰度图像，可以基于imshow函数定义一个
    // // 可以同时显示多张图像的函数
    // // 我们需要接受一个图像列表，可以使用自定义指针，也可以使用vector容器
    // // 因为指针的缺陷所以我们采用vector
    // vector<Mat> multiImage;
    // multiImage.push_back(image);
    // multiImage.push_back(outputImage);
    // // 我们还需要传入一个图像展示窗口名称
    // string str_window = "multi image";
    // multiImshow(str_window, multiImage);
    // // imshow("histogram equalization", outputImage);

    // 我们想要实现直方图规定化
    Mat inputImage = imread("c:/users/80521/desktop/dark.webp");
    Mat objectImage = imread("c:/users/80521/desktop/object.webp");
    Mat outputImage;
    // 我去，这里是不是没有定义引用传参，。。。。。。我看下不是这个问题
    histogram_match(inputImage, objectImage, outputImage);
    // vector<Mat> multiImage;
    // multiImage.push_back(inputImage);
    // multiImage.push_back(objectImage);
    // multiImage.push_back(outputImage);
    // histogram_equalization(inputImage, outputImage);
    // multiImage.push_back(outputImage);
    // multiImshow("compare the efficient", multiImage);

    // 下面我们试下三通道图像的规定化和均衡化
    // 首先将图像分类为三个通道
    Mat channel_input[3];
    Mat channel_object[3];
    split(inputImage, channel_input);
    // 这里的问题
    split(objectImage, channel_object);
    namedWindow("original color image", 0);
    namedWindow("object color image", 0);
    namedWindow("histogram equalization color image", 0);
    namedWindow("histogram match color image", 0);
    for (int i = 0; i < 3; i++)
    {
        histogram_equalization(channel_input[i], channel_input[i]);
    }
    merge(channel_input, 3, outputImage);

    imshow("original color image", inputImage);
    imshow("object color image", objectImage);
    imshow("histogram equalization color image", outputImage);

    for (int i = 0; i < 3; i++)
    {
        histogram_match(channel_input[i], channel_object[i], channel_object[i]);
    }
    Mat outputImageObject;
    merge(channel_object, 3, outputImageObject);
    imshow("histogram match color image", outputImageObject);






    waitKey(0);
    destroyAllWindows();
    system("pause");
    return 0;
}
