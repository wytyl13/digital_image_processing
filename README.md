# digital_image_processing
## gray transformation
### add the histogram equalization and histogram match
### add local histogram transformation
### add the content of histogram local thread and statistics
### add the content of some general gray transformation function

## 仿射变换

## 滤波
### 相关和卷积
### A图像，b滤波核或者卷积核
### 卷积核和滤波核的区别是旋转180度。
### 滤波操作，不符合交换律。卷积操作符合交换律

<!-- 1 0 1   1 1 0
0 0 1   0 1 0
1 1 0   1 1 0

A         B -->


### PADDING, 完全相关,相似卷积
### (original_image_size + 2*padding - kernel_size / 2) / step + 1 = original_image_size;
### kernel_size / 2;向下取整，int padding_size = kernel_size / 2;

### 平滑空间滤波
### 均值滤波，（中值滤波，类似于灰度变换,统计形式的滤波）
<!-- 1 255 1 1 
1 20 20 1
1 1  1  1
1 5  1  1 -->
### 为什么要有滤波？其实滤波只是将之前的灰度变换形式化。
### sum(gray(x, y)) / n
### 3*3, 均值滤波，盒状滤波器（滤波核像素点均相等）
<!-- 1 1 1
1 1 1 * 1/9)
1 1 1 -->
### 权重滤波。约定速成的距离概念，横轴和纵轴是最近的两个维度，两个斜轴上的
### 灰度值比横轴纵轴上的值距离中心点远一点。
<!-- 1 2 1
2 4 2  * 1/16)
1 2 1
中心点 -->


### 椒盐噪声
### 中值滤波
### 1 2 2 2 2 30 30， 中值是2， 均值比2大。

## 锐化空间滤波器（二阶微分和梯度）



