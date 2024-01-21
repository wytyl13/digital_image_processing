#include "general.h"

cv::Mat_<uchar> MEAN_FILTER = (cv::Mat_<uchar>(3, 3) << 1, 1, 1, 1, 1, 1, 1, 1, 1);
const cv::Mat_<int> General::LAPULASI_FILTER_1 = (cv::Mat_<int>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);
const cv::Mat_<int> General::LAPULASI_FILTER_2 = (cv::Mat_<int>(3, 3) << 1, 1, 1, 1, -8, 1, 1, 1, 1);
const cv::Mat_<int> General::SOBEL_LEFT = (cv::Mat_<int>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
const cv::Mat_<int> General::SOBEL_RIGHT = (cv::Mat_<int>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);


double *General::getDistribution(Mat inputImage) 
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

void General::histogram_equalization(Mat inputImage, Mat &outputImage) 
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
            // 这个就是累计概率分布
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

void General::multiImshow(string str, vector<Mat> vectorImage) 
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

double* General::getCumulative(Mat inputImage) 
{
    double *cumulativeDistribution;
    cumulativeDistribution = (double *)malloc(sizeof(double) * 256);
    memset(cumulativeDistribution, 0.0, sizeof(double) * 256);
    double *distribution = getDistribution(inputImage);
    double s = 0.0;
    for (int i = 0; i < 256; i++)
    {
        for (int j = 0; j < i; j++)
        {
            s += distribution[j];
        }
        cumulativeDistribution[i] = s;
        s = 0.0;
    }
    return cumulativeDistribution;
}

void General::histogram_match(Mat inputImage, Mat objectImage, Mat &outputImage) 
{
    // 我们刚才分析过了
    // 1 获取输入图像和给定分布的累计概率分布
    // 2 计算双方累计概率分布的差异
    // 3 找到最小的差异，返回对应的灰度值，就是对原始图像灰度值的映射
    // 4 根据映射map改变原始图像，即使最终规定化后的结果。
    // 我们之前在计算直方图规定化的时候其实已经计算好了累计概率分布。
    // 我们快速的重新定义一个函数把
    // 这样定义比较麻烦，我们还需要在调用函数中定义该传入参数，注意我们如果需要在主函数中
    // 调用该返回值，那么我们必须将该传出参数从主函数中一步步传过来，否则将无法调用
    // 我们需要分别获取输入图像和给定分布，本案例中是objectImage图像的累计概率分布
    // 使用传出参数不方便，我快速改下吧，使用返回值吧
    // Jixu 
    // 分别获取两个图像的累计概率分布
    
    
    double *cumulative_distribution_input = getCumulative(inputImage);
    double *cumulative_distribution_object = getCumulative(objectImage);
    

    // 下一步是计算差异
    // 我们选择的是定好一个cumulative_distribution_input，然后计算这个值和每一个
    // cumulative_distribution_object的差异，找到最小的那个差异对应的索引值就是映射的值
    int index;
    // 注意这里的数据类型定义错误了，应该是double。因为这里的double是概率分布，如果使用int定义
    // 接收到的值会强制转换，转换后的值都是0.所以造成刚才的图像像素值全是0.纯黑背景。大家这块要注意
    // 图像数据的存储很重要。
    double min_value;
    double s = 0.0;
    Mat mapping(1, 256, CV_8UC1);
    for (int i = 0; i < 256; i++)
    {
        // 先定义一个min_value,第一个索引差异,直接从第二个开始循环
        // 这个出现负数的概率不大
        min_value = cumulative_distribution_input[i] - cumulative_distribution_object[0];
        // 这里翻了一个严重的错误，我哦们应该根据两个1*256去计算得到差异矩阵256*256
        // 这块习惯性的定义循环按照之前计算累计概率分布的循环去定义了。
        // 大家在晚上脑袋蒙的时候还是出去转一圈休息一下，要不效率很低。
        for (int j = 1; j < 256; j++)
        {
            // 我这应该没有取绝对值。因为可能出现负数
            s = fabs(cumulative_distribution_input[i] - cumulative_distribution_object[j]);
            if (min_value > s)
            {
                min_value = s;
                index = j;
            }
        }
        // 这样循环一次，我们就会得到对应的index，然后我们可以根据index和i去定义映射图像
        // 我们可以使用指针去定义该映射，也可以使用Mat数据类型去定义，Mat数据类型去定义的
        // 好处是我们可以直接使用opencv官方定义的LUT函数去映射原始图像。我们使用Mat定义
        // 映射图,这里注意index的数据类型，index是int。我们将其强制转换为uchar
        // 这个关系应该也不大
        mapping.at<uchar>(i) = static_cast<uchar>(index);
    }
    // 然后使用映射图去更新原始图像即可,还是很简单的
    LUT(inputImage, mapping, outputImage);
}

void General::histogramLocalEqualization(Mat inputImage, Mat &outputImage, int kernelSize) 
{
    // 刚才我们已经分析了扫描的操作
    // 1 扫描固定区域，并且计算扫描区域的累计概率分布
    // 2 根据扫描区域的中心点对应的概率分布去计算对应的映射，这里注意是什么方法，比如本次使用的是均衡化，那就去计算均衡化映射的灰度值
    // 3 改变中心点的灰度值根据映射图
    // 4 逐次扫描，步伐是1
    // 就是这些，其实只是对之前的灰度值变换方法的一个改进

    // 首先定义扫描的起始点。扫描的起始点其实就是核尺寸的一半，这里注意因为我们需要改变的每次扫描的中心点
    // 所以我们的核尺寸一定是奇数。
    int half_kernel_size = kernelSize / 2;
    int width = inputImage.cols;
    int height = inputImage.rows;
    double *cumulative_distribution;
    outputImage = inputImage.clone();

    // 为了解决图像边框没有被处理的问题，我们进行0填充，也即给图像的边框填充0
    // 首先确定填充的范围，可以发现每个边填充的大小就是核尺寸的一半
    // 我们定义一个填充完后的图像吧
    int out_width = width + half_kernel_size * 2;
    int out_height = height + half_kernel_size * 2;
    // 我们定义一个填充后的图像，使用0填充，8比特单通道
    Mat padding_mat = Mat::zeros(Size(out_width, out_height), CV_8UC1);
    // 然后将原始输入图像粘贴到该图像制定的额区域。
    // 首先定义区域cv::Mat operator()(const cv::Rect &roi) const
    // Rect_(int _x, int _y, int _width, int _height)
    Mat roi = padding_mat(Rect(half_kernel_size, half_kernel_size, width, height));
    inputImage.copyTo(roi);

    // 打印一下看看对不对,可以应该没问题
    // 然后我们基于新制作的图像来改变下面的循环
    // 首先开始扫描
    // 注意宽对应的列数，高对应的行数，宽是x，高是y，原点是左上角0,0
    time_t start, end;
    start = time(NULL);
    // 我感觉是这个for循环出现问题了
    for (int row = half_kernel_size; row < (out_height - half_kernel_size); row++)
    {
        // 定义子区域
        // Rect_(int _x, int _y, int _width, int _height), 起始点和宽高
        // 起始点是原点，也即扫描的中心点减去half_kernel_size
        // 注意这里我们还需要定义列的循环才可以进行子区域的定义
        for (int col = half_kernel_size; col < (out_width - half_kernel_size); col++)
        {
            Mat subMat = padding_mat(Rect((col - half_kernel_size), (row - half_kernel_size), kernelSize, kernelSize));
            // 好了我们已经定义好了扫描的子区域，这里注意不要对子区域进行更改操作，否则将会更改原图
            // 我感觉这个定义Mat的方法应该绑定的是原始图像的地址。大家可以试下
            // 不过更改原始图像也没关系，我们后续不需要使用到它
            // 下面我们计算子区域的累计概率分布
            cumulative_distribution = getCumulative(subMat);
            // 然后就可以根据映射改编原始图像了
            // 我们需要改变扫描区域的中心点，对应的原始图像的坐标是
            // 因为是均衡化处理，所以使用累计概率分布乘以255即可，8比特图像最大灰度值是255
            // 我们需要找到子区域图像中心点的灰度值，因为我们要使用其对应的映射去改变原始图像中心点的灰度值
            // 注意子区域的中心点就是核尺寸的一半
            // 有问题
            int index = (int)subMat.at<uchar>(half_kernel_size, half_kernel_size);
            // 试下看看
            outputImage.at<uchar>(row - half_kernel_size, col - half_kernel_size) = (uchar)(cumulative_distribution[index] * 255);
        }
    }
    end = time(NULL);
    printf("%ld", (end - start));
}

void General::histogram_local_thread(Mat inputImage, Mat &outputImage,
                            int side_length, int thread_numbers) 
{

    time_t start, end;
    start = time(NULL);
    // 我们正式开始
    // 首先我们需要根据传入的线程参数将原始图像进行切分
    /* 
    让我们首先把问题简单化
        我们这样切分线程，我们分别将宽和高切分成m和n等分，m*n = thread_numbers.
        以上案例我们将原始图像切分为4个线程，长边和短边均是2
        我们可以根据长边和短边的线程数来定义每个线程图像对应的尺寸
            问题简单化是我们不考虑不能整除的部分，我们直接使用int接受，向下取整

        我们这样定义每个线程扫描的图像
            根据原点和宽高去定义，我们会根据对应的线程索引去计算该线程对应的
            图像中的原点，宽和高。
    切分完线程以后我们可以对每一个线程进行同样的操作，也就是自适应直方图均衡化
    然后每个线程回去改变原始图像的数据，最后得到均衡化的图像
     */

    // 让我们先来定义每一个线程的x, y, width, height.
    // 加入我们先写死宽边为4个线程cols_thread_numbers，然后我们可以定义高边的线程数
    // 宽边线程数的一半不要超过6.自己可以随意定义，也可以写死
    // 注意这里我们虽然没有写死宽边线程数，但是我们并没有对线程不能整除的情况下作兼容
    // 所以如果出现rows_thread_numbers不能被整除的情况下会出现错误
    // 大家可以自己对不能整除的情况做兼容。
    int cols_thread_numbers = (thread_numbers / 2 > 5) ? 4 : (thread_numbers / 2);
    int rows_thread_numbers = thread_numbers / cols_thread_numbers;

    // 然后我们可以根据线程数来计算对应的每个线程的宽和高
    int rows = inputImage.rows;
    int cols = inputImage.cols;


    // 注意我们之前使用int去接受这个结果，造成我们舍弃了未被整除部分，现在把它加上
    // 按照刚才那个案例，原始图像宽是7，宽边线程数是3，那么我们之前使用int接受会造成
    // int i = 7 / 3 = 2,这样我们使用3个线程，每个线程2个，覆盖6个，舍弃掉1个，现在我们把1覆盖上
    // 考虑还是原来的3个线程
    // int mod = 7 % 3 = 1;
    // int cols_thread = cols / cols_thread_numbers + mod = 3; 这是前两个线程的宽
    // 最后一个线程的宽是 7 - cols_thread * 2 = 1；
    // 下面我们按照这个想法来定义每一个线程的宽高，来覆盖之前舍弃掉的图像最右侧和最下侧部分
    int cols_mod = cols % cols_thread_numbers;
    int rows_mod = rows % rows_thread_numbers;

    // 注意考虑未整除部分，我们在上个视频已经分析了
    // 所以我们直接在for循环之前这样定义即可
    // 这个是除了最右侧和最左侧线程的所有线程的宽高，注意该宽高对应的是原始图像，不是padding后的图像
    int cols_thread = cols / cols_thread_numbers + cols_mod;
    int rows_thread = rows / rows_thread_numbers + rows_mod;

    // 这是前面的线程，但是我们需要在循环中定义，因为我们要做判断
    // 我们可以根据线程索引去定义每一个线程的x，y
    // 我们先来看第一行，线程索引号每增加一个，对应的x轴坐标改变，y轴坐标不变
    // 注意我们这里需要定义的是每一个线程对应的padding后的原始图像的原点坐标
    // 我们需要考虑x++多少，应该是每一个线程的宽cols_thread
    // 这里需要注意一下我们对应的是padding后的原始图像
    // 分别判断当前索引对应的线程是否是左上角，右上角，左下角和右下角
    bool left_upper, right_upper, left_bottom, right_bottom;
    int left_upper_index = 0;
    int right_upper_index = cols_thread - 1;
    int left_bottom_index = thread_numbers - cols_thread;
    int right_bottom_index = thread_numbers - 1;
    bool up, bottom, left, right;
    int half_side_length = side_length / 2;
    // 因为问题简单化了，所以每一个线程对应的宽高都是一样的，都是原始图像
    // 分配的宽高+2倍的half_side_length;
    int width, height;

    // 注意Mat的横轴和纵轴分别对应是宽和高，也就是cols和rows
    // 原点在左上角
    int padding_cols = cols + 2 * half_side_length;
    int padding_rows = rows + 2 * half_side_length;
    // 注意opencv中基本上所有类的构造函数都是width, height.
    // 只有极少部分，比如at是先高后宽
    Mat padding_mat = Mat::zeros(Size(padding_cols, padding_rows), CV_8UC1);
    Mat roi = padding_mat(Rect(half_side_length, half_side_length, cols, rows));
    inputImage.copyTo(roi);

    vector<thread *> thread_vectors;
    outputImage = padding_mat.clone();

    // 注意上次视频我们在for循环中定义了cols_thread和rows_thread,但是注意这里我们需要用到cols_thread
    // 所以我哦们必须再for循环之前定义该变量，注意因为最右侧和最下侧的cols_thread不同与其他的线程
    // 但是for循环只会循环到最右侧和最下侧之前的线程，所以这里我们不用在for循环之前重新定义最右侧和最下侧
    // 线程的宽高，直接使用其他线程的宽高即可。
    // 注意这里cols_thread不用考虑最右侧和最下侧，因为我们考虑的x和y对应的是当前线程的原点坐标
    // 不需要考虑最右侧和最下侧的线程
    for (int i = 0, x = 0, y = 0; i < thread_numbers; i++, x += cols_thread)
    {

        left_upper = (i == left_upper_index);
        right_upper = (i == right_upper_index);
        left_bottom = (i == left_bottom_index);
        right_bottom = (i == right_bottom_index);
        // 0 % 2 = 0, 2 % 2 = 0
        // 注意这里的变量错误了，不应该是cols_thread,它是每一个线程的宽
        // 我们需要的是宽线程数，cols_thread_numebrs;
        left = (i % cols_thread_numbers == 0);
        up = (i >= left_upper_index && i <= right_upper_index);
        bottom = (i >= left_bottom_index && i <= right_bottom_index);
        // 1 % 2 = 1; 3 % 2 = 1
        right = (i % cols_thread_numbers == 1);
        // 我们需要定义第二行的线程
        // 现成每加一行y会加rows_thread，但是注意需要排除掉左上角的线程
        // 并且每行的首个线程。下面我们来定义这些条件
        // 我们还需要定义一个判断是否为上下左右边的变量
        // 只有当当前线程在最左侧并且非左上角线程的时候这个条件才会生效
        if (left && !left_upper)
        {
            x = 0;
            y += rows_thread;
        }

        // 我们将简单问题复杂化，我们首先考虑的时最右侧和最下侧这两边的宽高和其他地方不一样
        // 最右侧的宽不一样，最下侧的高不一样。但是这样我们就不能在循环外面定义
        // width height 了
        // 我们需要考虑最右侧和最下侧和其他
        // 让我们来定义一下每一个线程对应的宽高，我们需要区分最右侧最下侧和其他
        // 先考虑最右侧，我们保证输入的线程数不变。我们更改宽高来覆盖整体图像。此时不能使用
        // int去接受参数了，因为它起到的是一个向下取整的效果
        // 因为我们在for循环之外已经定义了当前线程的宽高，所以这里我们不需要重复定义了
        // 我们需要考虑当前线程为最右侧和最下侧线程的时候的线程宽高
        // 这里不能根据cols_thread去定义cols_thread.因为在下一个循环中可能会存在错误赋值的情况
        // 所以我们需要重新计算最右侧和最下侧的cols_thread和rows_thread
        if (right)
        {
            cols_thread = cols - (cols / cols_thread_numbers + cols_mod) * (cols_thread_numbers - 1);
        }
        if (bottom)
        {
            rows_thread = rows - (rows / rows_thread_numbers + rows_mod) * (rows_thread_numbers - 1);
        }
        width = cols_thread + 2 * half_side_length;
        height = rows_thread + 2 * half_side_length;


        // 好了我们测试下
        // 但是需要注意的是我们的width和height都要在原始尺寸的基础上考虑加上padding
        // width = cols_thread + half_side_length * 2;
        // height = rows_thread + half_side_length * 2;
        // OKceshiyixia
        // OK,让我们测试下是否正确
        // 可以了线程的原点坐标和宽高已经测试正确
        // 下面我们就可以对每一个线程图像进行操作了
        // 注意我们这里的每一个线程是从padding后的图像中拿
        // 拿到的是引用地址，也就是说我们对temp_mat中的所有
        // 像素的修改也会影响到原始图像
        // 我们来定义padding_mat
        // 我们应该让temp_mat作为outputImage的子图
        // temp_mat供线程函数去改变原图
        // temp_mat_供线程函数去扫描
        Mat temp_mat = outputImage(Rect(x, y, width, height));
        Mat temp_mat_ = padding_mat(Rect(x, y, width, height));

        // 下面我们要在该函数中创建多个线程分别处理每一个线程图像
        // 注意这个是一个构造函数，是c++封装的windows系统的线程操作。
        // 我们在该函数中创建多个线程，数量根据thread_numbers的数量决定
        // 然后每一个线程共享所有虚拟内存，注意线程和进程的区别。。。
        // 进程是CPU的最小调度单元，现成是CPU的最小哦计算单元
        // 在linux中我们可以在一个主函数中创建多个进程，当某一个进程被创建以后，
        // 该进程享有创建进程后的所有代码。
        // 而线程一般是windows的概念，在windows中当一个进程在创建线程以后
        // 该进程将被称为主线程。。。。linux中也可以创建线程
        // 这里我们可以根据以下两点来区分进程和线程，可以将线程理解为一个函数
        // 该函数被执行完以后结束，而进程不同，进程共享该进程被创建以后的所有代码。
        // 如果有需要详细了解的可以参考深入理解计算机系统，c++已经将系统编程中
        // 的线程操作封装好了，我们只需要new一个实例即可。在该实例中我们需要传入
        // 线程函数和对应的需要传入线程函数的参数
            // 进程是CPU的最小调度单元，现成是CPU的最小哦计算单元
            // 在linux中我们可以在一个主函数中创建多个进程，当某一个进程被创建以后，
            // 该进程享有创建进程后的所有代码。
        // thread<_Callable, _Args...>(_Callable &&__f, _Args &&...__args)
        // 我们需要创建线程函数，注意该线程函数的定义有一个要求，因为我们在for循环中
        // 针对不同的线程调用的时同一个线程函数，所以我们需要将该线程函数可以通用的
        // 去操作每一个线程扫描的图像，这个就是唯一的要求。
        // 当代码执行到改行的时候，线程被创建，然后执行线程函数。
        // 该线程函数的执行是该子线程的操作，而主线程会继续在本函数中执行for循环
        // 接着创建下一个线程。。如此往复，子线程和主线程的操作可以同步。CPU只是
        // 在操作上下文切换。但是这里需要注意主线程可能会先于子线程提前结束
        // 所以需要注意防止主线程先于子线程提前结束的情况。因为我们这里每一个子线程的操作
        // 量都很大，所以主线程极有可能先结束。主线程结束意味着我们所有在主线程中定义的
        // 变量都会被释放。如果这些变量被释放，子线程中调用的这些变量将无效。或者出现错误
        // 下面我们来定义该线程函数
        // 我们首先可以确定到这个函数这块都是正确的，因为我们
        // 刚才测试了每一个temp_mat都是正常的
        thread *thread_pointer = new thread(&General::thread_function, this, temp_mat, temp_mat_, \
            cols_thread, rows_thread, half_side_length, side_length);
        // 主线程在创建完每一个子线程以后把该线程放在容器中
        thread_vectors.push_back(thread_pointer);
    }

    // 注意for循环执行完毕以后，所有的子线程都已经被创建了
        // 但是有一个问题就是for循环式主线程在执行，执行完毕以后主线程
        // 指导执行结束，这可能出现一个问题就是主函数先执行完毕了
        // 那么此时我们在线程函数中处理主函数的数据，temp_mat对应的就是主函数
        // 的部分地址。主函数中所有的变量都已经被释放了，我们在对该地址进行操作
        // 会报错。所以我们需要处理一下，也就是让主函数等待所有的子线程处完毕以后
        // 再退出，我们需要在for循环外处理，否则我们不能保证每一个线程
        // 都先被正常创建。
    // 然后我们在主线程调用的该函数即将结束之前使用join函数
    // 让主线程等待每一个子线程执行结束后再执行，也即阻塞
    for (int i = 0; i < thread_vectors.size(); i++)
    {
        thread *thread_i = thread_vectors[i];
        // 排查了好久，原来是这块错误了
        // 这块应该是判断thread_i != null，如果判断错误，我们将没有
        // 定义主线程阻塞等待子线程执行结束后再执行
        if (thread_i != NULL)
        {
            thread_i->join();
            // 阻塞结束后删除子线程，并且置空
            delete thread_i;
            thread_i = NULL;
        }
    }
    // HAOLE，测试下
    end = time(NULL);
    std::cout << end - start << std::endl;
    // 注意这里我们需要排除outputimage的padding。
    outputImage = outputImage(Rect(half_side_length, half_side_length, cols, rows));
}
void General::thread_function(Mat temp_mat, Mat temp_mat_, int cols_thread, int rows_thread, \
    int half_side_length, int side_length) 
{
    // 注意这里的循环次数,循环次数还是3，在刚才那个案例中，其实就是
    // 我们计算的cols_thread and rows_thread.
    // Mat temp_mat_ = temp_mat.clone();

    
    for (int row = 0; row < rows_thread; row++)
    {
        for (int col = 0; col < cols_thread; col++)
        {
            // 然后就是常规操作
            // 我们需要进行扫描操作
            Mat sub_mat = temp_mat_(Rect(col, row, side_length, side_length));
            // 计算该子图的累计分布
            double *cumulative = getCumulative(sub_mat);
            // 找到扫描区域最中心的数值作为索引
            // half_side_length,当然我们也可以计算，但是我们传参更方便
            int index = (int)sub_mat.at<uchar>(half_side_length, half_side_length);
            // 注意这步是常规操作，直方图均衡化需要使用累计分布乘上最大灰度值做映射。
            // 但是这里注意，我们在该子线程对temp_mat做了修改，然后下个循环又会
            // 从temp_mat取值，而且取值的可能会存在取到修改后的图像，如果我们不做干预
            // 处理后的图像会存在明显的白线，这是因为我们前脚刚处理完均衡化
            // 然后后脚我们就使用均衡化的数据去对下一个扫描区域进行均衡化，
            // 均衡化一般会提高当前像素的灰度值
            // 所以这里我们需要干预。我们对另一个传参进行扫描操作就可以很好的
            // 解决这个问题。
            temp_mat.at<uchar>(row + half_side_length, col + half_side_length) = \
            (uchar)(cumulative[index] * 255);
        }
    }
}

void General::histogram_local_statistics_thread(Mat inputImage, Mat &outputImage,
                            int side_length, int thread_numbers, double k[]) 
{

    time_t start, end;
    start = time(NULL);
    int cols_thread_numbers = (thread_numbers / 2 > 5) ? 4 : (thread_numbers / 2);
    int rows_thread_numbers = thread_numbers / cols_thread_numbers;

    int rows = inputImage.rows;
    int cols = inputImage.cols;
    int cols_mod = cols % cols_thread_numbers;
    int rows_mod = rows % rows_thread_numbers;
    int cols_thread = cols / cols_thread_numbers + cols_mod;
    int rows_thread = rows / rows_thread_numbers + rows_mod;
    bool left_upper, right_upper, left_bottom, right_bottom;
    int left_upper_index = 0;
    int right_upper_index = cols_thread - 1;
    int left_bottom_index = thread_numbers - cols_thread;
    int right_bottom_index = thread_numbers - 1;
    bool up, bottom, left, right;
    int half_side_length = side_length / 2;
    int width, height;

    int padding_cols = cols + 2 * half_side_length;
    int padding_rows = rows + 2 * half_side_length;
    Mat padding_mat = Mat::zeros(Size(padding_cols, padding_rows), CV_8UC1);
    Mat roi = padding_mat(Rect(half_side_length, half_side_length, cols, rows));
    inputImage.copyTo(roi);

    vector<thread *> thread_vectors;
    outputImage = padding_mat.clone();

    // 我们计算输入图像的统计量
    // 我们需要分别计算均值和方差，然后将均值和方差传递进县城函数中
    double mean_variance[2] = {0.0};
    cal_mean_variance(inputImage, mean_variance);
    double global_max_value;
    minMaxLoc(inputImage, 0, &global_max_value, 0, 0);

    for (int i = 0, x = 0, y = 0; i < thread_numbers; i++, x += cols_thread)
    {

        left_upper = (i == left_upper_index);
        right_upper = (i == right_upper_index);
        left_bottom = (i == left_bottom_index);
        right_bottom = (i == right_bottom_index);
        left = (i % cols_thread_numbers == 0);
        up = (i >= left_upper_index && i <= right_upper_index);
        bottom = (i >= left_bottom_index && i <= right_bottom_index);
        // 1 % 2 = 1; 3 % 2 = 1
        right = (i % cols_thread_numbers == 1);
        if (left && !left_upper)
        {
            x = 0;
            y += rows_thread;
        }

        if (right)
        {
            cols_thread = cols - (cols / cols_thread_numbers + cols_mod) * (cols_thread_numbers - 1);
        }
        if (bottom)
        {
            rows_thread = rows - (rows / rows_thread_numbers + rows_mod) * (rows_thread_numbers - 1);
        }
        width = cols_thread + 2 * half_side_length;
        height = rows_thread + 2 * half_side_length;
        Mat temp_mat = outputImage(Rect(x, y, width, height));
        Mat temp_mat_ = padding_mat(Rect(x, y, width, height));
        // 注意这个函数没有更换过来
        thread *thread_pointer = new thread(&General::thread_statistics_function, this, temp_mat, temp_mat_, \
            cols_thread, rows_thread, half_side_length, side_length, mean_variance, global_max_value, k);
        thread_vectors.push_back(thread_pointer);
    }

    for (int i = 0; i < thread_vectors.size(); i++)
    {
        thread *thread_i = thread_vectors[i];
        if (thread_i != NULL)
        {
            thread_i->join();
            delete thread_i;
            thread_i = NULL;
        }
    }
    // HAOLE，测试下
    end = time(NULL);
    std::cout << end - start << std::endl;
    outputImage = outputImage(Rect(half_side_length, half_side_length, cols, rows));
}

void General::thread_statistics_function(Mat temp_mat, Mat temp_mat_, int cols_thread, int rows_thread, \
    int half_side_length, int side_length, double mean_variance[], double global_max_value, double k[]) 
{
    // 首先，我们所有的工作都是在线程函数中进行的
    // 现成函数需要拿到一些指标，如输入图像的统计量，线程函数中扫描区域的统计量
    // 我们可以在调用线程函数的函数中定义全局图像的统计量，然后将该全局统计量传入线程函数中
    // 好了我们已经接受到参数了，我们还需要以下指标
    double G_mean = mean_variance[0];
    double G_variance = mean_variance[1];
    double local_mean_variance[2] = {0};
    double L_mean, L_variance, const_value, local_max_value;
    // 顺序分贝代表均值的最小因子，最大因子；方差的最小因子，最大因子
    double k0 = k[0];
    double k1 = k[1];
    double k2 = k[2];
    double k3 = k[3];
    for (int row = 0; row < rows_thread; row++)
    {
        for (int col = 0; col < cols_thread; col++)
        {
            Mat sub_mat = temp_mat_(Rect(col, row, side_length, side_length));
            // 计算子图也就是扫描区域的统计指标
            cal_mean_variance(sub_mat, local_mean_variance);
            L_mean = local_mean_variance[0];
            L_variance = local_mean_variance[1];
            // 到这里我们已经拿到了局部和全局的均值和方差
            // 我们还缺好三个指标，一个是我们需要制定两个范围，分别对应
            // 均值和方差的范围，一个是我们需要计算常量，因为我们要使用该常量
            // 放大原始图像中的低灰度值,自定义范围我们需要从主函数中传入，
            // 我们先来计算常量吧
            minMaxLoc(sub_mat, 0, &local_max_value, 0, 0);
            const_value = global_max_value / local_max_value;
            // OK没问题，我们继续，我们从主函数中传入对应的区间范围
            // 我们需要筛选范围
            if (L_mean >= G_mean * k0 && L_mean <= G_mean * k1 &&
                L_variance >= G_variance * k2 && L_variance <= G_variance * k3)
            {
                // 只有在这个统计范围内我们才去使用常数缩放对应的灰度值
                // 到此，程序结束
                temp_mat.at<uchar>(row + half_side_length, col + half_side_length) *= const_value;
            }
        }
    }
}

void General::cal_mean_variance(Mat inputImage, double mean_variance[]) 
{
    // 我们需要基于直方图去计算统计量
    double *distribution = getDistribution(inputImage);
    double mean = 0.0;
    double variance = 0.0;
    // 这里注意为什么要乘以分布，因为每个灰度值出现的概率不是均等的，如果是均等
    // 的我们直接可以除以256即可
    for (int i = 0; i < 256; i++)
    {
        mean += (i * distribution[i]);
    }

    for (int i = 0; i < 256; i++)
    {
        variance += (pow((i - mean), 2) * distribution[i]);
    }
    mean_variance[0] = mean;
    mean_variance[1] = variance;
}

void General::gray_transformation(Mat inputImage, Mat &outputImage, int mode, int threshold_value, int c, double gama) 
{
    // 图像反转
    // 我们可以直接使用Mat实例的广播机制进行数学运算
    // y = -x + (L - 1)
    if (mode == REVERSE)
    {
        int L = pow(2, 8) - 1;
        outputImage = -1 * inputImage + (L - 1);
    }

    // 阈值变换
    // 我们可以定义下该阈值函数
    if (mode == BINARY)
    {
        binary_transformation(inputImage, outputImage, threshold_value);
        // 注意这里打印为空，说明变换函数是错误的，
        // 好了，测试完毕，但是有一个问题就是我们将该图像二值化以后，图像是不能正常展示的，所以我们
        // 可以定义一个像素缩放函数，该函数可以将图像的像素映射到0-255区间内，无论该图像的灰度值
        // 是什么区间内，这个函数都可以做到将原始图像的像素值映射到0-255，这个是我们想要达到的效果
        // 我们的目的是将图像原始像素0,1映射到0-255区间内，映射后的结果是0,255
    }
    if (mode == LOGORITHM)
    {
        // 如果inputImage的最大灰度值为255，那么执行该操作以后8比特图像将不能存储该结果，所以
        // 我们要在这之前重新定义一个图像数据类型去接受该结果
        // 注意我们这里定义了一个inputimage尺寸大小的float64数据类型，他可以存储超过255的图像数据
        // 而且可以保存浮点数据类型
        // 这个错误应该是在进行图像广播运算的时候发出的.
        inputImage.convertTo(outputImage, CV_64F);
        outputImage += 1;
        // void log(cv::InputArray src, cv::OutputArray dst)
        // 这里需要注意的是，灰度变换可能会出现非uchar类型的灰度值，所以我们这里需要进行
        // 图像类型的转换
        // 这个错误是图像数据类型出现的错误
        cv::log(outputImage, outputImage);
        outputImage *= c;
        // 将图像数据类型从浮点数类型转换为uchar类型，当然最后还是需要进行线性缩放
        // 将图像的灰度值映射为0-255， 因为我们后续都是基于8比特图像进行的分析
        outputImage.convertTo(outputImage, CV_8UC1);
    }
    if (mode == GAMA)
    {
        // 注意gama的数据类型是double
        // y = c * x^r
        inputImage.convertTo(outputImage, CV_64F);
        // void pow(cv::InputArray src, double power, cv::OutputArray dst)
        cv::pow(outputImage, gama, outputImage);
        // 注意还有一个系数c，但是其实我们只需要更换gama值就可以满足我们的变换了。
        outputImage.convertTo(outputImage, CV_8UC1);
    }
    linear_scaling(outputImage, outputImage);
}

void General::binary_transformation(Mat inputImage, Mat &outputImage, int threshold_value)
{
    uchar *row_mat;
    // 需要遍历每一个像素点
    outputImage = inputImage.clone();
    for (int i = 0; i < inputImage.rows; i++)
    {
        row_mat = inputImage.ptr<uchar>(i);
        for (int j = 0; j < inputImage.cols; j++)
        {
            // 根据阈值改变原像素值
            // 我们可以使用三目运算符处理该值
            outputImage.at<uchar>(i, j) = (row_mat[j] >= threshold_value) ? 1 : 0;
        }
    }
}

void General::linear_scaling(Mat inputImage, Mat &outputImage) 
{
    // 线性缩放的表达式为
    // 我们首先需要对图像进行归一化,但是这里注意去均值化好像并不能满足我们的要求，我们的目标是将
    // 图像映射到0-255，我们首先要做的时将原始图像映射到0-1区间内。简单的去均值化并不能做到这一点
    // 比如对于二值图像，0,1 -> 0,255
    // y = (x - min) / max 
        // = (0 - 0) / 1 = 0
        // = (1 - 0) / 1 = 1
    // 这样我们可以首先把原始图像映射到0,1 然后乘以255，可以将原始图像映射到0,255
    double min_value, max_value;
    minMaxLoc(inputImage, &min_value, &max_value, 0, 0);
    outputImage = (inputImage - min_value) / max_value * 255;
}

void General::bit_plane(Mat inputImage, Mat &outputImage, int bit) 
{
    // 我们首先根据比特数构建灰度值区间范围
    // 8比特层范围为2^7-2^8-1 = 128, 255
    int min_value = pow(2, bit - 1);
    int max_value = pow(2, bit) - 1;
    // 我们需要循环遍历每一个像素点
    uchar *row_mat;
    // uchar gray_value;
    outputImage = inputImage.clone();
    for (int i = 0; i < inputImage.rows; i++)
    {
        row_mat = inputImage.ptr<uchar>(i);
        for (int j = 0; j < inputImage.cols; j++)
        {
            // 将非区间内的灰度值置为0
            // 注意这里不能这么定义，因为我们可能会遗漏掉灰度值为0的像素点
            // 所以我们反着来
            // 注意这个结果不用做线性缩放
            // gray_value = row_mat[j];
            // 注意这里是或不是并
            outputImage.at<uchar>(i, j) = (row_mat[j] < min_value || \
                row_mat[j] > max_value) ? 0 : row_mat[j];
        }
    }
}

// ---------------------------------------------------------------------------------

void General::spatial_filter_thread(Mat inputImage, Mat &outputImage,
                            const cv::Mat filter, int thread_numbers) 
{
    int side_length = filter.cols;
    int total_pixel_filter = cv::sum(filter)[0];
    time_t start, end;
    start = time(NULL);

    int cols_thread_numbers = (thread_numbers / 2 > 5) ? 4 : (thread_numbers / 2);
    int rows_thread_numbers = thread_numbers / cols_thread_numbers;

    int rows = inputImage.rows;
    int cols = inputImage.cols;

    int cols_mod = cols % cols_thread_numbers;
    int rows_mod = rows % rows_thread_numbers;

    int cols_thread = cols / cols_thread_numbers + cols_mod;
    int rows_thread = rows / rows_thread_numbers + rows_mod;

    bool left_upper, right_upper, left_bottom, right_bottom;
    int left_upper_index = 0;
    int right_upper_index = cols_thread - 1;
    int left_bottom_index = thread_numbers - cols_thread;
    int right_bottom_index = thread_numbers - 1;
    bool up, bottom, left, right;
    int half_side_length = side_length / 2;
    int width, height;

    int padding_cols = cols + 2 * half_side_length;
    int padding_rows = rows + 2 * half_side_length;
    Mat padding_mat = Mat::zeros(Size(padding_cols, padding_rows), CV_8UC1);
    Mat roi = padding_mat(Rect(half_side_length, half_side_length, cols, rows));
    inputImage.copyTo(roi);

    vector<thread *> thread_vectors;
    outputImage = padding_mat.clone();
    cv::Mat result;
    for (int i = 0, x = 0, y = 0; i < thread_numbers; i++, x += cols_thread)
    {

        left_upper = (i == left_upper_index);
        right_upper = (i == right_upper_index);
        left_bottom = (i == left_bottom_index);
        right_bottom = (i == right_bottom_index);
        left = (i % cols_thread_numbers == 0);
        up = (i >= left_upper_index && i <= right_upper_index);
        bottom = (i >= left_bottom_index && i <= right_bottom_index);
        right = (i % cols_thread_numbers == 1);
        if (left && !left_upper)
        {
            x = 0;
            y += rows_thread;
        }

        if (right)
        {
            cols_thread = cols - (cols / cols_thread_numbers + cols_mod) * (cols_thread_numbers - 1);
        }
        if (bottom)
        {
            rows_thread = rows - (rows / rows_thread_numbers + rows_mod) * (rows_thread_numbers - 1);
        }
        width = cols_thread + 2 * half_side_length;
        height = rows_thread + 2 * half_side_length;

        Mat temp_mat = outputImage(Rect(x, y, width, height));
        Mat temp_mat_ = padding_mat(Rect(x, y, width, height));

        thread *thread_pointer = new thread(&General::mean_filter_function, this, temp_mat, temp_mat_, \
            filter, cols_thread, result, rows_thread, half_side_length, side_length, total_pixel_filter);
        thread_vectors.push_back(thread_pointer);
    }

    for (int i = 0; i < thread_vectors.size(); i++)
    {
        thread *thread_i = thread_vectors[i];
        if (thread_i != NULL)
        {
            thread_i->join();
            delete thread_i;
            thread_i = NULL;
        }
    }
    end = time(NULL);
    std::cout << end - start << endl;
    outputImage = outputImage(Rect(half_side_length, half_side_length, cols, rows));
}

void General::mean_filter_function(Mat temp_mat, Mat temp_mat_, \
    const cv::Mat filter, int cols_thread, cv::Mat result, \
    int rows_thread, int half_side_length, int side_length, \
    const int total_pixel_filter) 
{
    for (int row = 0; row < rows_thread; row++)
    {
        for (int col = 0; col < cols_thread; col++)
        {
            // 扫描到的子图
            Mat sub_mat = temp_mat_(Rect(col, row, side_length, side_length));
            cv::multiply(sub_mat, filter, result);
            temp_mat.at<uchar>(row + half_side_length, col + half_side_length) = \
            (uchar)(cv::sum(result)[0] / total_pixel_filter);
        }
    }
}

void General::median_filter_function(Mat temp_mat, Mat temp_mat_, \
    int cols_thread, cv::Mat result, \
    int rows_thread, int half_side_length, int side_length) 
{
    for (int row = 0; row < rows_thread; row++)
    {
        for (int col = 0; col < cols_thread; col++)
        {
            // 扫描到的子图
            Mat sub_mat = temp_mat_(Rect(col, row, side_length, side_length));
            // 找到中值
            // Mat --> vector.
            std::vector<uchar> sub_mat_vector = std::vector<uchar>(sub_mat.begin<uchar>(), sub_mat.end<uchar>());
            std::sort(sub_mat_vector.begin(), sub_mat_vector.end());

            temp_mat.at<uchar>(row + half_side_length, col + half_side_length) = \
            (uchar)(sub_mat_vector[side_length * side_length / 2]);
        }
    }
}

void General::median_filter_thread(Mat inputImage, Mat &outputImage,
                        int side_length, int thread_numbers) 
{
    time_t start, end;
    start = time(NULL);

    int cols_thread_numbers = (thread_numbers / 2 > 5) ? 4 : (thread_numbers / 2);
    int rows_thread_numbers = thread_numbers / cols_thread_numbers;

    int rows = inputImage.rows;
    int cols = inputImage.cols;

    int cols_mod = cols % cols_thread_numbers;
    int rows_mod = rows % rows_thread_numbers;

    int cols_thread = cols / cols_thread_numbers + cols_mod;
    int rows_thread = rows / rows_thread_numbers + rows_mod;

    bool left_upper, right_upper, left_bottom, right_bottom;
    int left_upper_index = 0;
    int right_upper_index = cols_thread - 1;
    int left_bottom_index = thread_numbers - cols_thread;
    int right_bottom_index = thread_numbers - 1;
    bool up, bottom, left, right;
    int half_side_length = side_length / 2;
    int width, height;

    int padding_cols = cols + 2 * half_side_length;
    int padding_rows = rows + 2 * half_side_length;
    Mat padding_mat = Mat::zeros(Size(padding_cols, padding_rows), CV_8UC1);
    Mat roi = padding_mat(Rect(half_side_length, half_side_length, cols, rows));
    inputImage.copyTo(roi);

    vector<thread *> thread_vectors;
    outputImage = padding_mat.clone();
    cv::Mat result;
    for (int i = 0, x = 0, y = 0; i < thread_numbers; i++, x += cols_thread)
    {

        left_upper = (i == left_upper_index);
        right_upper = (i == right_upper_index);
        left_bottom = (i == left_bottom_index);
        right_bottom = (i == right_bottom_index);
        left = (i % cols_thread_numbers == 0);
        up = (i >= left_upper_index && i <= right_upper_index);
        bottom = (i >= left_bottom_index && i <= right_bottom_index);
        right = (i % cols_thread_numbers == 1);
        if (left && !left_upper)
        {
            x = 0;
            y += rows_thread;
        }

        if (right)
        {
            cols_thread = cols - (cols / cols_thread_numbers + cols_mod) * (cols_thread_numbers - 1);
        }
        if (bottom)
        {
            rows_thread = rows - (rows / rows_thread_numbers + rows_mod) * (rows_thread_numbers - 1);
        }
        width = cols_thread + 2 * half_side_length;
        height = rows_thread + 2 * half_side_length;

        Mat temp_mat = outputImage(Rect(x, y, width, height));
        Mat temp_mat_ = padding_mat(Rect(x, y, width, height));

        thread *thread_pointer = new thread(&General::median_filter_function, this, temp_mat, temp_mat_, \
            cols_thread, result, rows_thread, half_side_length, side_length);
        thread_vectors.push_back(thread_pointer);
    }

    for (int i = 0; i < thread_vectors.size(); i++)
    {
        thread *thread_i = thread_vectors[i];
        if (thread_i != NULL)
        {
            thread_i->join();
            delete thread_i;
            thread_i = NULL;
        }
    }
    end = time(NULL);
    std::cout << end - start << std::endl;
    outputImage = outputImage(Rect(half_side_length, half_side_length, cols, rows));
}

cv::Mat General::get_mean_filter(const int kernel_size, int weight_flag) 
{
    cv::Mat result = cv::Mat::ones(kernel_size, kernel_size, CV_8UC1);

    /* 
    1 2 1
    2 4 2
    1 2 1

    1 1 1 1 1
    1 1 2 1 1
    1 2 4 2 1
    1 1 2 1 1
    1 1 1 1 1
     */
    if (weight_flag)
    {
        for (size_t i = 0; i < kernel_size; i++)
        {
            for (size_t j = 0; j < kernel_size; j++)
            {
                if (i == (kernel_size / 2) && j == (kernel_size / 2))
                {
                    result.at<uchar>(i, j) = 4;
                } else if (i == (kernel_size / 2 + 1) && j == (kernel_size / 2))
                {
                    result.at<uchar>(i, j) = 2;
                } else if (i == (kernel_size / 2 - 1) && j == (kernel_size / 2))
                {
                    result.at<uchar>(i, j) = 2;
                } else if (i == (kernel_size / 2) && j == (kernel_size / 2 - 1))
                {
                    result.at<uchar>(i, j) = 2;
                } else if (i == (kernel_size / 2) && j == (kernel_size / 2 + 1))
                {
                    result.at<uchar>(i, j) = 2;
                }
            }
        }
    }
    return result;
}

void General::guassian_noise(cv::Mat input_image, cv::Mat &output_image, const double mean, const double std) 
{
    output_image = input_image.clone();
    cv::Mat noise = cv::Mat(output_image.rows, output_image.cols, CV_8UC1);
    cv::randn(noise, mean, std);
    output_image += noise;
}

void General::saltPepper(cv::Mat input_image, cv::Mat &output_image, const int noise_size, int count) 
{
    output_image = input_image.clone();
    int noise_type;
    int x, y;
    while (count--)
    {
        x = rand() % (output_image.rows - noise_size + 1);
        y = rand() % (output_image.cols - noise_size + 1);
        noise_type = rand() % 2;
        for (size_t i = 0; i < noise_size; i++)
        {
            for (size_t j = 0; j < noise_size; j++)
            {
                output_image.at<uchar>(x + i, y + j) = noise_type ? 255 : 0;
            }
        }
    }
}

// ---------------------------------------------------------------------------------
void General::general_filter_function(Mat temp_mat, Mat temp_mat_, \
    const cv::Mat filter, int cols_thread, \
    int rows_thread, int half_side_length, int side_length, \
    const cv::Mat filter_sobel_right, const int edge_flag, const int total_pixel_filter) 
{
    cv::Mat_<double> result_;
    cv::Mat_<double> result__;
    double amplitude_value, sum_result_, sum_result__, mean_sub_mat, change_value;
    for (int row = 0; row < rows_thread; row++)
    {
        for (int col = 0; col < cols_thread; col++)
        {
            // 扫描到的子图
            Mat sub_mat = temp_mat_(Rect(col, row, side_length, side_length));
            cv::multiply(sub_mat, filter, result_);
            sum_result_ = cv::sum(result_)[0];
            if (!filter_sobel_right.empty())
            {
                cv::multiply(sub_mat, filter_sobel_right, result__);
                sum_result__ = cv::sum(result__)[0];
                mean_sub_mat = cv::mean(sub_mat)[0];
                amplitude_value = std::sqrt(std::pow(sum_result_, 2) + std::pow(sum_result__, 2));
                // change_value = edge_flag ? 0.0 : amplitude_value;
                // temp_mat.at<double>(row + half_side_length, col + half_side_length) = \
                //     (amplitude_value <= mean_sub_mat) ? 0.0 : amplitude_value;
                temp_mat.at<double>(row + half_side_length, col + half_side_length) = amplitude_value;
            }
            else
            {
                temp_mat.at<double>(row + half_side_length, col + half_side_length) = sum_result_;
            }
        }
    }
}

void General::general_filter_thread(Mat inputImage, Mat &outputImage, \
    const cv::Mat filter, const cv::Mat filter_sobel_right, const int edge_flag, \
    int thread_numbers) 
{
    int side_length = filter.cols;
    int total_pixel_filter = cv::sum(filter)[0];
    time_t start, end;
    start = time(NULL);

    int cols_thread_numbers = (thread_numbers / 2 > 5) ? 4 : (thread_numbers / 2);
    int rows_thread_numbers = thread_numbers / cols_thread_numbers;

    int rows = inputImage.rows;
    int cols = inputImage.cols;

    int cols_mod = cols % cols_thread_numbers;
    int rows_mod = rows % rows_thread_numbers;

    int cols_thread = cols / cols_thread_numbers + cols_mod;
    int rows_thread = rows / rows_thread_numbers + rows_mod;

    bool left_upper, right_upper, left_bottom, right_bottom;
    int left_upper_index = 0;
    int right_upper_index = cols_thread - 1;
    int left_bottom_index = thread_numbers - cols_thread;
    int right_bottom_index = thread_numbers - 1;
    bool up, bottom, left, right;
    int half_side_length = side_length / 2;
    int width, height;

    int padding_cols = cols + 2 * half_side_length;
    int padding_rows = rows + 2 * half_side_length;
    Mat padding_mat = Mat::zeros(Size(padding_cols, padding_rows), CV_64FC1);
    Mat roi = padding_mat(Rect(half_side_length, half_side_length, cols, rows));
    cv::Mat input_image_double;
    inputImage.convertTo(input_image_double, CV_64FC1);
    input_image_double.copyTo(roi);

    vector<thread *> thread_vectors;
    outputImage = padding_mat.clone();
    for (int i = 0, x = 0, y = 0; i < thread_numbers; i++, x += cols_thread)
    {

        left_upper = (i == left_upper_index);
        right_upper = (i == right_upper_index);
        left_bottom = (i == left_bottom_index);
        right_bottom = (i == right_bottom_index);
        left = (i % cols_thread_numbers == 0);
        up = (i >= left_upper_index && i <= right_upper_index);
        bottom = (i >= left_bottom_index && i <= right_bottom_index);
        right = (i % cols_thread_numbers == 1);
        if (left && !left_upper)
        {
            x = 0;
            y += rows_thread;
        }

        if (right)
        {
            cols_thread = cols - (cols / cols_thread_numbers + cols_mod) * (cols_thread_numbers - 1);
        }
        if (bottom)
        {
            rows_thread = rows - (rows / rows_thread_numbers + rows_mod) * (rows_thread_numbers - 1);
        }
        width = cols_thread + 2 * half_side_length;
        height = rows_thread + 2 * half_side_length;

        Mat temp_mat = outputImage(Rect(x, y, width, height));
        Mat temp_mat_ = padding_mat(Rect(x, y, width, height));
        cv::Mat_<double> result;
        thread *thread_pointer = new thread(&General::general_filter_function, this, temp_mat, temp_mat_, \
            filter, cols_thread, rows_thread, half_side_length, side_length, \
            filter_sobel_right, edge_flag, total_pixel_filter);
        thread_vectors.push_back(thread_pointer);
    }

    for (int i = 0; i < thread_vectors.size(); i++)
    {
        thread *thread_i = thread_vectors[i];
        if (thread_i != NULL)
        {
            thread_i->join();
            delete thread_i;
            thread_i = NULL;
        }
    }
    end = time(NULL);
    std::cout << end - start << std::endl;
    outputImage = outputImage(Rect(half_side_length, half_side_length, cols, rows));
    if (filter_sobel_right.empty())
    {
        if (edge_flag)
        {
            outputImage.convertTo(outputImage, CV_8UC1);
            return;
        }
        printf("WHOAMI\n");
        cv::Mat_<double> sub_image;
        cv::subtract(inputImage, outputImage, sub_image);
        sub_image.convertTo(outputImage, CV_8UC1);
    }
    else
    {
        // 进入了这个分支，导致计算错误
        double alpha = edge_flag ? 0 : 0.5;
        double beta = 1 - alpha;
        cv::Mat_<double> add_weight_image;
        cv::addWeighted(inputImage, alpha, outputImage, beta, 0, add_weight_image);
        add_weight_image.convertTo(outputImage, CV_8UC1);
    }
}
// ---------------------------------------------------------------------------------