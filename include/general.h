#ifndef MAIN_H_
#define MAIN_H_

#include <opencv2/opencv.hpp>



using namespace std;
using namespace cv;

/**
 * @Author: weiyutao
 * @Date: 2023-08-28 19:54:30
 * @Parameters: inputImage, 输入灰度图像
 * @Return: 需要返回一个灰度值的分布，我们可以选择传出参数也可以使用
 * 返回值的形式，我们首先使用一下返回值的形式，因为后续我们一般
 * 使用传出参数。返回值可以选择两种，第一是返回一个我们自定义的指针
 * 一个是使用c++的容器vector，其实就相当于一个列表，我们首先使用下
 * 指针的形式吧。我们在返回值出接受一个double类型的指针，这个指针
 * 是一个地址，我们可以使用改地址索引多个double
 * @Description: 获取灰度图像的分布的函数
 */
double* getDistribution(Mat inputImage);

/**
 * @Author: weiyutao
 * @Date: 2023-08-28 20:28:14
 * @Parameters: inputImage, 输入图像
 * @Parameters: outputImage, 输出图像，这里使用了传出参数，传出参数使用&负号，是c++
 * 特有的，c语言不支持&负号在函数参数中作为传出参数的功能。这个可以理解为我们任何传输
 * 到第二个位置的参数都是将地址绑定上去了，任何在本函数中，也就是在执行栈帧中的更改
 * 都会对主函数中对应内存的变量生效。
 * @Return: null
 * @Description: 注意直方图均衡化是根据原始图像的分布去更改原始图像的像素值
 * new_number = (number - 1) * Σdistribution_i
 * 注意对这个计算表达式的理解
 * number是图像像素值的个数，不包含0就减去1
 * distribution_i是像素值为i的分布，前面加西格玛
 * 就是在当前像素值之前的所有像素值的分布总和，从0开始增加，一直到255
 * 比如
 * 1 0.1
 * 2 0.2
 * 3 0.3
 * 4 0.2
 * new_number_1 = 255*0.1
 * new_number_2 = 255*(0.1+0.2)
 * new_number_3 = 255*(0.1+0.2+0.3)
 * new_number_4 = 255*(0.1+0.2+0.3+0.2)
 * 注意因为新的像素值是uchar类型，所以我们需要对计算的结果向下取整。
 * 当然我们先来计算下如何从旧的像素值映射到新的像素值
 * 
 */
void histogram_equalization(Mat inputImage, Mat &outputImage);
void multiImshow(string str, vector<Mat> vectorImage);


/**
 * @Author: weiyutao
 * @Date: 2023-08-29 22:45:36
 * @Parameters: inputImage需要传入一张原始图像
 * @Parameters: outputImage传出参数，输出结果图像
 * @Parameters: objectImage这个就当做我们的目标图像吧，我们需要我们的输出图像的分布无限接近他的分布
 * @Return: null
 * @Description: 这里注意一点，我们是自己给定一个已知的分布，还是从一张图像中获取分布。我们一般选择
 * 后者，比较简单那。所以我们还需要定义一个传入参数。
 */
void histogram_match(Mat inputImage, Mat objectImage, Mat &outputImage);
/**
 * @Author: weiyutao
 * @Date: 2023-08-29 22:50:42
 * @Parameters: 
 * @Return: 
 * @Description: 累计概率分布，我们需要选择一种方式拿大这个概率分布，我们试下传出参数把
 */
void histogram_match(Mat inputImage, Mat objectImage, Mat &outputImage);
#endif