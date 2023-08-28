#include "general.h"

double *getDistribution(Mat inputImage) 
{
    // 首先我们需要遍历输入图像中的每一个像素值

    // 获取输入图像的宽和高
    int rows = inputImage.rows;
    int cols = inputImage.cols;
    int total_number = rows * cols;
    uchar *rowMat;

    // 注意这里我们犯了一个严重的错误，就是
    // double distribution[256];这个其实已经是定义一个指针了。但是我们在前面又加了
    // *意味着我们定义了一个指针数组。所以咋成了后续的报错
    // 注意我们这个是声明了一个指针变量，注意指针变量在没有定义前都是无效的。
    // 申请256个double大小的内存在堆区,应该需要强制转换为指针
    double *distribution;
    distribution = (double *)malloc(sizeof(double) * 256);
    // 让我们初始化下内存，因为可能存在脏数据
    // void * memset(void *_Dst, int _Val, size_t _Size)
    // 注意这个size_t变量，他应该是内存大小，而不是数量，我们申请了256个double
    // 那么它占用的内存大小就是sizeof(double)*256
    memset(distribution, 0.0, sizeof(double) * 256);
    for (int i = 0; i < rows; i++)
    {
        // 遍历读取每一行像素，使用uchar类型的指针呢接受
        // 因为我们是基于8比特单通道图像分析的，所以使用uchar类型
        // 存储，uchar即unsigned char占一个字节，8个比特
        rowMat = inputImage.ptr<uchar>(i);
        for (int j = 0; j < cols; j++)
        {
            // 获取每一个元素
            // 如何表示每一个像素值出现的频率，首先我们需要统计每一个
            // 像素值出现的次数。我们构建一个指针列表，然后使用索引
            // 作为每一个像素值，索引值作为该像素值出现的次数或者频率
            // 像素值的取值范围为0-255，总共256个，定义一个256个double元素的指针
            // 然后让每一个取到的像素值作为索引，遍历一个++一下，我们可以得到
            // 每一个像素值在图像中出现的次数
            distribution[rowMat[j]]++;
        }
    }

    // 然后我们还需要遍历一次distribution，基于它计算每一个像素值出现的频率
    // 注意这里出现一个问题，就是我们自定义的指针是没有获取具体元素值的方法的，
    // 除非我们规定了一个结尾值，然后我们可以基于该结尾值去判断是否遍历结束，
    // 否则我们无法获取元素的具体数量，当然这里我们已知的元素数量是256，所以我们
    // 可以直接写死。后续我们会使用vector容器去作为存储工具
    // 注意是占比整张图像的出现频率。
    for (int i = 0; i < 256; i++)
    {
        // 因为之前的定义错误导致了这块获取到的时一个指针，指针是无法进行数学计算的
        // Zheli错误了，真粗心
        distribution[i] /= total_number;
    }

    return distribution;
}

void histogram_equalization(Mat inputImage, Mat &outputImage) 
{

    // 首先获取分布
    double *distribution = getDistribution(inputImage);
    double xigema_distribution_value = 0.0;
    // 这两个变量名重复了。。。。。
    double new_gray_value[256] = {0};
    // 映射像素值
    //注意这里，哦我们在对xigema_distribution_value复制之后，没有重置
    // 导致第二个循环中使用了第一个循环中的数值。
    for (int i = 0; i < 256; i++)
    {
        // 这样也可以。
        // double xigema_distribution_value = 0.0;
        for (int j = 0; j < i; j++)
        {
            xigema_distribution_value += distribution[j];
        }
        // round返回的是指针类型，但是他提示我必须
        new_gray_value[i] = round(255 * xigema_distribution_value);
        xigema_distribution_value = 0.0;
    }
    // 然后我们生成了新的像素值，下面我们去改变原始的像素值即可
    // 在outputImage的基础上更改。不去改变原始图像
    outputImage = inputImage.clone();
    int rows = outputImage.rows;
    int cols = outputImage.cols;
    uchar *rowMat;
    for (int i = 0; i < rows; i++)
    {
        rowMat = outputImage.ptr<uchar>(i);
        for (int j = 0; j < cols; j++)
        {
            // 注意像素值的数值类型，我们需要的是uchar类型
            // 而我们传入的是一个double，强制转换下试试
            // 但是注意这并不是一个解决问题的办法，因为double -> uchar
            // double是8个字节，而uchar是一个字节，如果数值很大很容易出现
            // 溢出现象。
            rowMat[j] = (uchar)new_gray_value[rowMat[j]];
        }
    }
}

void multiImshow(string str, vector<Mat> vectorImage) 
{
    int numImage = (int)vectorImage.size();
    int w, h; // w means the image numbers for row, h means the image numbers for columns.
    // just like, w is 2 if you want to show two image in one window.
    // w is 1 if you want to show one image in one window.
    int height, width; // the height, width that each image based on the numbers of input image.

    if (numImage <= 0)
    {
        printf("the image numbers arguments you passed too small!");
        return;
    }
    else if (numImage > 12)
    {
        printf("the image number arguments you passed too large!");
        return;
    }

    if (numImage <= 4)
    {
        height = 360; width = 600;
        switch (numImage)
        {
        case 1:
            h = 1; w = 1;
            break;
        case 2:
            h = 1; w = 2;
            break;
        default:
            h = 2; w = 2;
            break;
        }
    }
    else if (numImage >= 5 && numImage <= 9)
    {
        height = 180; width = 300;
        switch (numImage)
        {
        case 5:
            h = 2; w = 3;
            break;
        case 6:
            h = 2; w = 3;
            break;
        default:
            h = 3; w = 3;
            break;
        }
    }
    else
    {
        height = 90; width = 150;
        h = 4; w = 3;
    }

    Mat dstImage = Mat::zeros(60 + height * h, 90 + width * w, CV_8UC1);
    // notice, you should start from 20,20. because you should reserved space between two image.
    // m, n is cooresponding the element corrdinates x, y in the dstImage.
    // this bigImage is the output image than involved all input image.
    for (int i = 0, m = 20, n = 10; i < numImage; i++, m += (10 + width))
    {
        if (i % w == 0 && m != 20)
        {
            // if true, you should start from 20, because it must be the right of the window.
            m = 20;
            n += 10 + height;
        }
        // frame of a region in original image dstImage.
        // this region used variable imgROI to show.
        Mat imgROI = dstImage(Rect(m, n, width, height));
        // notice. the first param of Size is width, the second param is height.
        resize(vectorImage[i], imgROI, Size(width, height));
    }
    imshow(str, dstImage);
}