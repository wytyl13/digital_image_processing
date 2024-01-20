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
```
1 0 1   1 1 0
0 0 1   0 1 0
1 1 0   1 1 0

A         B
```

### PADDING, 完全相关,相似卷积
```
(original_image_size + 2*padding - kernel_size) / step + 1 = original_image_size;
```
### 
```
kernel_size / 2;向下取整，int padding_size = kernel_size / 2;
```

### 平滑空间滤波
### 均值滤波，（中值滤波，类似于灰度变换,统计形式的滤波）
```
1 255 1 1 
1 20 20 1
1 1  1  1
1 5  1  1
```
### 为什么要有滤波？其实滤波只是将之前的灰度变换形式化。
### sum(gray(x, y)) / n
### 3*3, 均值滤波，盒状滤波器（滤波核像素点均相等）
```
1 1 1
1 1 1 * 1/9)
1 1 1
```
### 权重滤波。约定速成的距离概念，横轴和纵轴是最近的两个维度，两个斜轴上的
### 灰度值比横轴纵轴上的值距离中心点远一点。
```
1 2 1
2 4 2  * 1/16)
1 2 1
中心点
```

### 椒盐噪声
### 中值滤波
### 1 2 2 2 2 30 30， 中值是2， 均值比2大。

## 锐化空间滤波器（二阶微分和梯度）

### 1、二阶微分和梯度在图片边缘检测方面的区别
```
1 1 2 3 4 5 6 10 10 10
0 0 1 1 1 1 1 4 0 0
0 0 1 0 0 0 0 3 -4 0

f(x)
df/dx = f(x+1)-f(x)
d^2f/dx^2 = f(x+1)+f(x-1)-2f(x)

f(x, y)
df/dx = f(x+1, y)-f(x, y)
df/dy = f(x, y+1)-f(x, y)

d^2f/dx^2=f(x+1, y)+f(x-1, y)-2f(x, y)
d^2f/dy^2=f(x, y+1)+f(x, y-1)-2f(x, y)

3*3
f(x-1, y-1)  f(x, y-1)   f(x+1, y-1)
f(x-1, y)    f(x, y)      f(x+1, y)
f(x-1, y+1)  f(x, y+1)   f(x+1, y+1)

0   1   0
1   -4  1
0   1   0

1   1   1
1   -8  1
1   1   1
拉普拉斯算子计算得到原始图像的二阶微分即图像的边缘检测结果

在原始图像的基础上加上或者减去边缘检测结果，即可达到图像锐化的效果
g(x, y) = f(x, y) + c * laplas_result(边缘检测结果) 
c = -1/1(laplas算子中心点的符号)
```
![original_image_laplas1_laplas2](markdown_image/original_image_laplas1_laplas2.png)

