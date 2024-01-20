#ifndef MAIN_H_
#define MAIN_H_

#include <opencv2/opencv.hpp>
#include <thread>
#include <random>

using namespace std;
using namespace cv;

enum THREAD_NUMBERS
{
    THREAD_2 = 2,
    THREAD_4 = 4,
    THREAD_6 = 6,
    THREAD_8 = 8,
    THREAD_10 = 10,
    THREAD_12 = 12,
};

enum TRANSFORMATION
{
    REVERSE, //反转
    BINARY, // 二值
    LOGORITHM, // 对数变换
    GAMA, //伽马变换
};

extern cv::Mat_<uchar> MEAN_FILTER;

#define IS_STATIC_MEMBER(className, memberName) \
    std::is_same<decltype(className::memberName), decltype(&className::memberName)>::value

class General
{
public:

    static const cv::Mat_<int> LAPULASI_FILTER_1;
    static const cv::Mat_<int> LAPULASI_FILTER_2;

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
    double *
    getDistribution(Mat inputImage);

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
     * @Date: 2023-08-30 17:43:01
     * @Parameters: inputImage,输入图像
     * @Parameters: outputImage,传出参数，输出图像
     * @Parameters: 因为是局部扫描，所以我们需要定义一个扫描尺寸,我们暂且称之为核大小吧
     * @Return: null
     * @Description: 
     */
    void histogramLocalEqualization(Mat inputImage, Mat &outputImage, int kernelSize);

    /**
     * @Author: weiyutao
     * @Date: 2023-09-03 22:32:29
     * @Parameters: inputImage， 输入图像
     * @Parameters: outputImage ， 输出图像
     * @Parameters: side_length ， 扫描尺寸，一个奇数矩阵
     * @Parameters: side_length ， 扫描尺寸，一个奇数矩阵
     * @Parameters: thread_numbers ，需要开启的线程数，也就是切分原始图像的数量
     * @Return: 
     * @Description: 多线程操作灰度直方图自适应。。我们至少需要以下四个参数
     */
    void histogram_local_thread(Mat inputImage, Mat &outputImage,
                                int side_length, int thread_numbers);

    /**
     * @Author: weiyutao
     * @Date: 2023-09-04 00:07:50
     * @Parameters: temp_mat，传入图像，注意这里不需要使用引用地址，因为它本身就是一个地址
     *          我们对他的所有操作都会影响到原始图像
     * @Parameters: cols_thread， rows_thread 每一个线程的宽高，注意对应的是原图的宽高
     * @Parameters: half_side_length， 扫描尺寸的一半
     * @Parameters: side_length 扫描尺寸
     * @Return: 
     * @Description: 现成函数，我们至少需要这么几个参数,我们先来定义，缺什么参数添加什么
     */
    void thread_function(Mat temp_mat, Mat temp_mat_, int cols_thread, int rows_thread, \
        int half_side_length, int side_length);

    double* getCumulative(Mat inputImage);

    void histogram_local_statistics_thread(Mat inputImage, Mat &outputImage,
                                int side_length, int thread_numbers, double k[]);

    void thread_statistics_function(Mat temp_mat, Mat temp_mat_, int cols_thread, int rows_thread, \
        int half_side_length, int side_length, double mean_variance[], double global_max_value, double k[]);

    /**
     * @Author: weiyutao
     * @Date: 2023-09-04 22:38:42
     * @Parameters: inputImage, 输入图像
     * @Parameters: mean_variance, 传出参数
     * @Return: null
     * @Description: 我们使用传出参数来接受返回值吧，我们传递了一个指针，该指针是mean_variance数组的
     * 首地址
     */
    void cal_mean_variance(Mat inputImage, double mean_variance[]);

    /**
     * @Author: weiyutao
     * @Date: 2023-09-06 01:10:14
     * @Parameters: 
     * @Return: 
     * @Description: 首先实现图像反转
     */
    void gray_transformation(Mat inputImage, Mat &outputImage, int mode, int threshold_value, int c, double gama_value);

    /**
     * @Author: weiyutao
     * @Date: 2023-09-06 01:18:55
     * @Parameters: threshold_value,阈值
     * @Return: 
     * @Description: 阈值处理函数,这里为了使用同一个函数去定义不同的常规变换，我们还需要定义
     * 一个传入参数，我们可以使用枚举类定义该参数，区分不同的变换类型
     */
    void binary_transformation(Mat inputImage, Mat &outputImage, int threshold_value);

    /**
     * @Author: weiyutao
     * @Date: 2023-09-06 01:32:41
     * @Parameters: 
     * @Return: 
     * @Description: 我们现在的操作是线性缩放，当然还有非线性缩放的方式。线性缩放的意思是将图像原像素
     * 点采用线性的方式缩放到8比特图像0-255区间内。
     */
    void linear_scaling(Mat inputImage, Mat &outputImage);

    /**
     * @Author: weiyutao
     * @Date: 2023-09-06 02:22:29
     * @Parameters: inputImage ，输入图像
     * @Parameters: outputImage，输出图像
     * @Parameters: bit 比特层
     * @Return: 
     * @Description: 比特分层函数
     */
    void bit_plane(Mat inputImage, Mat &outputImage, int bit);


    // -------------------------------------------------------------
    void spatial_filter_thread(Mat inputImage, Mat &outputImage,
                                const cv::Mat filter, int thread_numbers);

    void mean_filter_function(Mat temp_mat, Mat temp_mat_, \
        const cv::Mat filter, int cols_thread, cv::Mat result, \
        int rows_thread, int half_side_length, int side_length, \
        const int total_pixel_filter);
        
    cv::Mat get_mean_filter(const int kernel_size, int weight_flag = 0);
    void guassian_noise(cv::Mat input_image, cv::Mat &output_image, const double mean, const double std);
    void saltPepper(cv::Mat input_image, cv::Mat &output_image, const int noise_size, int count);
    void median_filter_thread(Mat inputImage, Mat &outputImage,
                            int side_length, int thread_numbers);
    void median_filter_function(Mat temp_mat, Mat temp_mat_, \
        int cols_thread, cv::Mat result, \
        int rows_thread, int half_side_length, int side_length);


    // -------------------------------------------------------------
    void general_filter_function(Mat temp_mat, Mat temp_mat_, \
        const cv::Mat filter, int cols_thread, \
        int rows_thread, int half_side_length, int side_length, \
        const int total_pixel_filter = 0);

    void general_filter_thread(Mat inputImage, Mat &outputImage,
                            const cv::Mat filter, int thread_numbers);
    // -------------------------------------------------------------
};

#endif