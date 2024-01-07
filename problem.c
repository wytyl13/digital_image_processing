/**********************************************************************
* Copyright (C) 2023. IEucd Inc. All rights reserved.
* @Author: weiyutao
* @Date: 2023-12-10 19:19:55
* @Last Modified by: weiyutao
* @Last Modified time: 2023-12-10 19:19:55
* @Description: 




// ------------------------------------------------------------------------------------
关于深度学习方面的一些思考***************
需要解决的问题
1、小数据学习
2、真正的有记忆
3、基于记忆的合理推理
4、其次，除了要具有小数据的学习能力，还要具有根据标准和规则学习的能力
比如医生对疾病的诊断，首先医生会根据经验进行疾病诊断，对于一些疑难杂症，医生会根据理论和行业标准去诊断


那么对于疾病的诊断，我们的难点在于
1、我们首先至少要具备医生的经验，这在现有的神经网络模式下需要我们输入大量的先验数据
当然，医学图像中不同疾病之间的低区分度是对现有神经网络去学习医学专家经验的一项巨大挑战。目前一个不成熟的
切实可行的办法就是我们可以在正向标签的基础上引入负向标签来增加模型对疾病的识别准确率。当然还有更好的解决办法
至少可以引入传统的图像算法来解决这个问题，也就是在每个前向传播的节点之间，我们可以使用传统图像算法对
特征图进行相应的处理，然后再让其进入下一个前向传播节点，但是这样的话对应的反向传播算法也要做相应的更改



批正则化：规范化批次输入样本的分布并减少输入数据的变化范围（因为归一化是将输入样本的每个特征归一到均值为0，方差为1的分布。并且这样一来
输入数据的特征变化范围也会缩小）因此输出数据的分布和范围也会相应改变。因此可以避免梯度爆炸或者梯度消失，达到稳定梯度的效果。因此批正则化
是在每层输出的基础上做的，目的是为了稳定下一层输入数据的分布和变化范围。


卷积如何梯度：首先使用卷积核进行卷积，然后根据卷积的结果和实际的标签求得损失函数，然后针对损失函数求卷积核中每一个参数的偏导数，
这个偏导数是一个值，因为每一个卷积核都会初始化参数值，然后这个偏导数就是梯度下降的方向，当然这个方向是前进的，如果我们想要下降的
方向，那就应该加上负号。然后我们使用参数的原来值和梯度下降的方向还有学习率（就是步长）来更新参数即可。这就是反向传播中的其中一步。2214004451323


批量、批次的概念不同。批次一般指每次迭代使用的样本数量。而批量一般指批量梯度下降法或者小批量梯度下降法，这里的批量最大的等于
批次的大小，即批量梯度下降法的批量大小就是每次迭代使用的样本数量就是批次的大小。而小批量梯度下降法中的小批量一般小于批次的大小。

学习率一般不是固定不变的，固定学习率一般满足不了我们的训练需求。
学习率方法主要有固定学习率、动态学习率和自适应学习率。
动态学习率、自适应学习率
Adam（Adaptive Moment Eistimation）优化算法结合了自适应学习率和动量的概念。其基本思想是在训练的过程中为每个模型参数维护一个自适应的学习率。
优点是：自适应学习率、快速收敛（因为动量的引入，往往可以快速达到局部最优解）、对稀疏梯度和噪声具有鲁棒性


普通梯度下降很难处理梯度爆炸或者梯度消失的问题，而动量梯度下降可以解决梯度下降过程中的摆动问题。提升训练效率。
普通参数直接使用学习率作为补偿，梯度作为方向来更新参数，动量梯度下降则是首先使用梯度、上一个时刻的动量和动量系数来计算当前时刻的动量，然后使用
当前时刻的动量和学习率去更新当前时刻的参数。
v = βv_ + (1-β)∇J(θ)，其中v是当前时刻的动量，β是动量系数（一般是0.9），v_是上一时刻的动量，∇J(θ)是梯度。
θ = θ_ - α*v，其中θ是更新后的参数，θ_是更新前的参数，α是学习率也就是步长，v是当前时刻的动量，也即动量梯度下降中的梯度下降方向。
这个动量方向更能让梯度下降模拟出向一个有重量的铁球从曲面上滚落下来的感觉。首先，当有质量的铁球遇到梯度很大的曲面的时候，也即很陡峭的曲面，
他不会沿着曲面走，而是会越过这个曲面直接向下坠落。这个无疑会加快梯度下降的效率。其次，因为其有质量，其不会产生较大的震荡，而且产生震荡的时间
也会很短，容易达到最低点。


卷积层关于卷积核参数的大小，需要考虑到输入数据的维度数
因为每个卷积核的维度需要和输入数据的维度数相当，比如
一个RGB图像，其输入数据是3通道，如果第一层卷积的输出特征图
数量是64，那么对应的3*3的卷积核的参数就是64*3*3*3，64是卷积核的个数
3*3是每个卷积核每个通道的参数量，3是每个卷积核有三个通道。

比如第二个卷积层输出的特征图是128个，那么对应的3*3
的卷积核的参数就是128*3*3*64，64是每个卷积核的通道数
128是卷积核的个数

注意残差块的思想是使得残差块的网络直接学习输入和输出之间的差距
而不是直接去学习输出。因为直接学习输出意味着去学习一个完全新的表示
可能会导致信息的丢失和混淆，特别是在更深层的网络中。

随着深度神经网络层数的逐渐增加，提取到的特征逐渐从简单过度到复杂。比如从边缘
和纹理到物体的部分或者整体结构。残差块可以直接将前一层的特征信息传递到本层。

注意标准的残差块就是简单的输入和输出相加，变种的残差网络设计结构存在加权
和这一说法。

在深层卷积神经网络中，残差块确实可以防止梯度消失的问题，但是
如果残差连接过于复杂或者跳跃层级过多，也可能会增加梯度消失会梯度爆炸的问题，
因为跳跃连接会使得梯度传播路径变长（比如如果加入了残差连接，那么在正常
的反向传播完成以后还要考虑跳跃连接的反向传播，这样反向传播变得更加复杂了）。
因此残差块跳跃不宜过长。
并且一般情况下不宜在底层网络中使用残差，除非某些特定的解决问题，比如U-net
的编码阶段使用了特征提取和下采样，而解码阶段使用了上采样进行分割，上采样就
包含了融合底层网络特征到高层网络。
综上，
1、残差的适当使用可以使得模型在特定的解决问题上可以学习更加复杂的函数，
也即解决更加复杂的视觉问题，这是因为直接将信息传递给了下一层，模型不需要学习
新的特征表示，而是学习前后差异即可
2、可以防止梯度消失或者梯度爆炸。这是因为不用直接学习新的表示
3、U-Net在上采样阶段将高层提取到的语义信息和底层提取到的位置信息在空间
上拼接起来，即通道数量翻倍，
4、简单来讲，池化就是单纯的下采样过程。其实卷积也可以通过设置步长达到下采样的效果
5、卷积可以替代全连接，但是有些问题上乏力，比如分类问题。当输出层使用卷积的
时候，一般情况下输出的特征图的尺寸数会远远大于1，而我们又想让输出的尺寸数等于1
因为比如二分类的问题，我们希望我们输出的尺寸数是1*1*2，但是如何让尺寸数
从14*14*32 --> 1*1*2呢，1*1*32卷积，然后设置卷积核个数为2，这个卷积操作
会输出14*14*2，完了直接使用全局平均池化输出结果为1*1*2，这样的话
池化的尺寸将会是14*14，这会丢失很多的信息，一般池化的尺寸都是2或者3.因此
这样虽然可以但是不科学。
6、1*1的卷积其实相当于全连接或者称之为Network in Network.
1*1卷积核还有一个作用就是可以专门针对改变特征的维度。而池化则更加针对于
改变特征图的尺寸，即前者针对于特征图的数量，后者针对于每个特征图的大小
这样两个操作其实就是专门用来改变特征图的空间属性的。当然除了全连接、
改变特征图的空间属性外，其还有一个主要作用就是可以结合relu进行非线性操作
，非线性激活函数对于线性激活函数sigmoid等来说，更多的应用于特征提取的情况下
，他可以用来提取更加复杂的特征，即可以学习到更加复杂的函数形式，针对这一点
非线性激活函数和残差块的作用一致。因此1*1卷积的作用很大。
一方面1*1可以改变特征图的数量，另一方面可以结合非线性函数来学习更加复杂的函数，
最后1*1的本质上其实就是全连接。
7、注意池化的主要作用是下采样，但是这并不代表它只能做下采样。池化也有步长
，池化尺寸等等，通过控制这两个参数我们可以得到不同的输出，当然池化也可以
改变特征图的数量，比如一个28*28*192的输入，我们可以将其池化输出为28*28*32
，因为池化是没有参数要学习的，因此我们可以通过更改输入特征图的维度来达到
我们的效果，我们可以使用空间金字塔池化结合全连接的方法，
首先将28*28*192特征图划分
为4个子区域，这样每个子区域的维度就是7*7*192，然后对其进行最大池化操作，
这样每个区域得到一个向量1*192，合并4个区域就是4*4*192，然后使用全连接
将其映射到28*28*32，即输入为4*4*192，输出为28*28*32.这里注意我们操作
完空间金字塔以后维度其实可以直接reshape了，但是这里不能，原因至少有两点：
第一，直接reshape可能会丢失子区域之间的特征信息，因为reshape并不能体现
映射关系，而空间金字塔只是单纯的将子区域进行最大池化操作，没有保留子区域之间的关系
而全连接可以学习到映射关系。
第二，虽然全连接不存在参数的学习，也即我们可以在最大池化的基础上进行任何操作
，因为任何操作都不会增加反向传播的计算难度和计算量，但是这样做也同时会带来
一些不必要的计算量和误差。

8、关于上个问题，我想到了如何在网络结构之间加入一些对特征图使用
传统算法的基本操作了，之前我想在加入传统算法后是否需要考虑到梯度的问题
现在看来完全不同担心，因为不涉及到任何参数的学习，但是既然不涉及到参数的学习，
那么是否会对学习参数产生影响呢？这个是值得考虑的问题。池化都可以影响，为什么
传统算法不能影响？池化主要为了下采样，而传统算法为了什么？可以加入某些限制性规则
我们可以设计一个和池化平级操作的传统算法来加入到池化操作之后，这样我们可以
将一些规则或者标准融入到图像的特征提取中。


Inception Network.
we can complex our layer by using many operation, just like
we can use multi different size convolution kernels and maxpool
to instead one convolution operation. just like the original method
to extract 64 features maps from original image 28*28*192 is to use the 
64 numbers 3*3*192 kernel and padding 1 and step 1, then, we can use inception
network just like from 28*28*192 --> 28*28*64
    8 1*1*192 kernels and padding 0, step1.
    8 3*3*192 kernels and padding 1, step1.
    16 5*5*192 kernels and padding 2, step1.
    32 SPP(maxpool) and full connection.
        SPP, sub region: 4*4*192, maxpool --> 1*1*192
            7*7*192 --> full connection mapping to--> 28*28*32

    then, concat: 8+8+16+32 = 64, 28*28*64
this is the key minds about Inception Network.

9、then we can calculate size for the different size kernel.
just like from 28*28*192 to 28*28*32, we can have these two
method above.
first, we can use 5*5 size kernel.
we should use 32 numbers 5*5*192 size kernel.
the calculate size of one kernel for one channel will be 5*5*28*28.
one kernel have 192 channels, so one kernel will have 5*5*28*28*192
calculate sizes. and we have 32 kernels, so the calculate size will
be 5*5*28*28*192*32 = 120,422,400 it is almost one point two 
hundred million.

second, we can use 16 numbers 1*1*192 size kernel to convolution first.
then use 32 numbers 5*5*16 size kernel second.
28*28*192 --16 1*1*192 kernels--> 28*28*16 --32 5*5*16 kernels--> 28*28*32
the calculate size will be 
1*1*28*28*192*16 + 5*5*28*28*16*32 = 12,443,648 it is almost one point
two thousand calculate size.
the first convolution can be named as bottole layer for the second method.
we can find that the first method is almost 10 times than the second 
method.
so you can find that the advance about inception, because it can use
the different size kernel to implement the same result.
and we have found that the bottole layer can significantly reduces
computational complexity by adding one middle layer used 1*1 kernel
and reduce the feature mapping numbers first and increase it second
used the bigger kernel size.

creat ourselves' nueral network.
the input of inception is generally the activate value or the output value of 
the previous layer.

the application of inception is googleNet what combined many
inceptions in the net structure.

mobile net, why need it? if you want your neural network to run
on a device with less powerful CPU or GPU at development.
the mobilenet that could perform much better.
key idea: depthwise-separable convolusions.
it rounding up roughly 10 times cheaper in computational
costs than the normal convolution neural network.
just like the case as follow.
the normal convolution neural network.
6*6*3 -> convolution 3*3*3, 5 filters, no padding, step 1 -> 4*4*5
the params are 3*3*3*4*4*5 = 2160

the mobile net key idea:
6*6*3 -> convolution 3*3*3, 3 filters, no padding, step 1 -> 4*4*3
-> convolution 1*1*3, 5 filers, no padding, step 1 --> 4*4*5
the params are 3*3*3*4*4*3 + 1*1*3*4*4*5 = 1296 + 240 = 1536




// ------------------------------------------------------------------------------------



// ------------------------------------------------------------------------------------
数字图像处理
在一维情况下，旋转180度等于沿着x轴反转
在二维情况下，旋转180度等同于沿着一个坐标轴反转，然后沿着另一个坐标轴再次反转。
卷积和相关操作
卷积满足交换律，相关不满足交换律

对于相关，交换滤波和图像位置，在完全相关下，即进行填充的情况下，交换前和交换后
相关的结果是完全相反。而对于完全卷积，则是完全相等。这就是交换律。
当然，如果滤波和图像的尺寸不一致，那么非完全卷积在交换后也仅仅是在卷积结果尺寸上的不同
，完全卷积则是完全满足交换律的。就是因为交换律这个特性，才有了卷积的意义。

1 1 1 1 1           1 1 0
1 0 1 0 1           0 1 1
1 1 1 0 1           0 0 1
1 1 0 1 1
1 1 0 1 0

0 0 0 0 0 0 0 
0 0 0 0 0 0 0 
0 0 1 1 0 0 0                1 1 1 1 1           1 1 0
0 0 0 1 1 0 0                1 0 1 0 1           0 1 1
0 0 0 0 1 0 0                1 1 1 0 1           0 0 1
0 0 0 0 0 0 0                1 1 0 1 1
0 0 0 0 0 0 0                1 1 0 1 0

相关结果：
4 3 4
3 3 3
3 4 3

4 4 

1 0 1      1 1 0
0 0 1      0 1 1
0 1 0      0 0 1




空間濾波模板产生的原因：比如我们现在想要使用原始像素3*3邻域内的平均值去替换原始像素值，
使用1/9 3*3的空间滤波器可以达到相同的效果。这种操作将会使得图像变得更加平滑。
我们可以自定义滤波器模板，主要包括线性和非线性的。注意这里线性和非线性的定义
和深度学习中的激活函数不一样。我们在之前了解到深度学习中的激活函数区分线性和
非线性是根据函数图像是否弯曲和平直来区分的。这里我们根据函数是否平滑来区分
线性滤波器还是非线性滤波器。
线性滤波器：一个具有两个变量的连续函数，比如高斯核函数。
非线性滤波器：最大值等，不是一个平滑的函数。
非线性滤波器功能非常强大，在某些应用中可以执行超出线性滤波器能力的功能

平滑空间滤波器
平滑滤波用于模糊处理和降低噪声。桥接曲线或直线的缝隙，在大目标提取之前取出图像中的一些琐碎细节。
简单平均值就是一个平滑的线性空间滤波器。也称为均值滤波器，也属于低通滤波。
注意低通滤波和高通滤波的区别，前者是过滤到灰度值高的像素点，通过灰度值低的像素点。因此会达到图像降噪
和平滑图像的效果。后者是过滤掉地灰度值，通过高灰度值。
常用的平滑空间滤波器的主要应用之一就是降噪。
所有系数都相等的空间均值滤波器有时称为盒状滤波器
当然，均值滤波也可以有一定的权重，比如给予距离中心点较劲的像素点较大的权重
1 2 1
2 4 2  * 1/16
1 2 1
以上权重总和为16，效果等同于均值滤波，但是给予距离中心点较劲的像素点更高的权重
随着均值滤波器核尺寸的不断增加，图像的模糊程度越强
空间线性均值滤波器的一个重要的应用就是
    首先通过均值滤波（低通滤波）将噪声点（也就是不感兴趣的点，一般是小目标）与背景混合在一起
    此时较大的物体（也就是我们感兴趣的点）变得像斑点，此时再使用阈值处理将图像二值化
    我们就可以得到去除噪点后较为清晰的感兴趣的点的二值图像
    我们可以通过加权均值或者盒状滤波，也可以调节滤波器核尺寸的大小。目的是将不感兴趣的点和
    背景图像进行融合。

统计排序（非线性）滤波器
    比如中值滤波，使用邻域范围内像素点排序后的中值（将数据由大到小排序后取中间值）去替换中心像素点
    中值滤波可以处理一定类型的随机噪声，比如脉冲噪声（椒盐噪声），这种噪声以黑白点的形式叠加在图像上
    很明显，均值滤波器核不能处理椒盐噪声，因为这种噪声是随机的黑白点叠加在原图上
    而中值滤波的非线性统计排序特性使得它可以处理该问题
    中值滤波器的主要功能是使拥有不同灰度的点看起来更接近于它的相邻点。
    当然注意，中值其实就是在平均数的基础上考虑了标准差过大的情况，如果某邻域
    范围内的灰度值标准差过大，使用平均数来代替中心像素点将不能解决黑白噪声这种情况
    而中值可以很好的解决这个问题。比如 122222339 中值是第5个值也就是2，但是均值却是2.7
    他俩相差不大，但是如果是椒盐噪声 1 2 2 2 2 2 5 5 200 中值是2，均值是25
    使用均值很难兼顾去除黑点和白点。

当然，非线性滤波器还有很多用法，比如求最大值，而不是中值。最大值滤波器的效果是搜寻
一副图像中的最亮点。其实也可以称之为高通滤波。当然也可以使用最小值滤波器，目的相反。


锐化空间滤波器
图像锐化处理的主要目的是突出灰度的过度部分。图像模糊可以通过在空间域中
像素邻域平均法实现，类似于积分的效果，而图像锐化在逻辑上可以使用空间微分来实现这一效果
图像微分可以增强边缘和其他突变，并且削弱了灰度变化缓慢的区域
我们可以了解下一阶微分和二阶微分
让我们首先定义一些规则
对于一阶微分
第一，在恒定灰度区域内微分值为零
第二，在灰度台阶或斜坡处微分值非零
第三，沿着斜坡的微分值非零
二阶微分类似

这里注意图像中对于当前像素点的一阶微分和二阶微分
df/dx = f(x+1) - f(x) 当前像素点的一阶微分等于下一个像素点减去当前像素点
d^2f/dx^2 = f(x+1) + f(x-1) - 2f(x) 当前像素点的二阶微分等于下一个像素点加上上一个像素点减去2倍的当前像素点的值
注意一阶微分和二阶微分的异同
首先，恒定区域，一阶二阶均为0
其次，在斜坡和台阶的起点和终点，两个微分都不为0
最后，对于斜坡来说，一阶微分不为0，二阶微分为0
还有一个特殊之处是：在一个台阶的过度中，台阶的初始点的一阶微分和下一点的二阶微分差距很大
如果连接这两点，将会和横轴相较于一个点，这点就是零交叉点。零交叉点对于边缘定位非常重要

基于以上特性，数字图像中的一阶微分产生的边缘较粗，这是因为一阶微分在斜坡上的值非零
并且数字图像中的边缘在灰度上常常类似于斜坡过度，因此我们使用二阶微分可以很好的表示图像的边缘
因为二阶微分产生由零分开的一个像素宽的双边缘，因此二阶微分在增强细节方面比一阶微分好很多
这是一个适合锐化图像的理想特征

对于一个二维图像，我们如何计算其二阶导数呢？需要注意的是，我们要使用滤波的方式去计算
df(x, y)/(dx, dy) = d^2f/dx^2 + d^2f/dy^2 = f(x+1,y)+f(x-1,y)-2f(x, y)+f(x,y+1)+f(x,y-1)-2f(x, y)
=f(x+1,y)+f(x-1,y)+f(x,y+1)+f(x,y-1)-4f(x, y)
我们可以使用一个3*3的滤波器核实现这个二阶导数的计算
f(x-1,y-1) f(x-1,y) f(x-1,y+1)
f(x,y-1)   f(x,y)   f(x,y+1)
f(x+1,y-1) f(x+1,y) f(x+1,y+1)
根据以上的像素点位置和图像的二阶导数公式，我们可以定义以下几种滤波器核实现同样的二阶导数计算效果
0  1  0
1 -4  1
0  1  0
注意以上滤波器核就是使用滤波的方式实现当前扫描像素区域的二阶导数，当然对应的f(x,y)是扫描区域的中心点
也即计算得到的就是扫描区域中心点的二阶微分
那么我们得到二阶导以后该如何根据二阶微分去增强原图的边缘呢？
首先我们需要增强边缘，而二阶微分计算得到的就是当前像素点的二阶微分值，该值可正可负
我们首先需要根据滤波扫描图像生成对应的二阶微分图像，然后使用原图加上或者减去该二阶微分图像来
对原图进行边缘增强，因为非边缘区域的二阶微分值为0，而边缘区域的二阶微分值非0.因此可以做到边缘
增强的效果。但是怎么判断在原图的基础上加还是减呢
一般情况下，是根据算子计算得到的中心点的微分值去计算的，正的微分值表示中心点较周围灰度值高，可以加上增强边缘
负的微分值表示中心点较周围灰度值低，可以减去抑制边缘
当然，这个方法只是一般的操作，更加科学的操作是设定一个阈值，超过该阈值的微分值才去进行增强边缘的操作
阈值可以根据经验法则，直方图分析

但是这样计算很麻烦，我们可以根据书本经验得到，我们一般可以根据算子的中心点去判断是在原图的基础上
加上还是减去拉普拉斯计算得到中心点的微分值
如果拉普拉斯算子的中心点为负，则应该在原图的基础上减去微分值
如果拉普拉斯算子的中心点为正，则应该在原图的基础上加上微分值
这是因为我们需要增强边缘的图像都是比较暗的图，所以具体该增加还是减少还是取决于实际应用场景

当然，我们可以对原始算子取反操作，然后得到
0  1  0
1 -4  1
0  1  0
但是注意取反后我们需要在原图的基础上加上拉普拉斯计算得到的中心点的二阶微分值
以上算子是根据二阶微分的数学离散公式推导得到的，在几何上我们可以直观的看出
以上算子仅考虑到了水平和垂直两个方向上的变化，并没有考虑对角线上的像素变化
因此我们可以在此基础上考虑对角线的变换，给其赋予对应的权重。当然，此时
我们的中心点的权重也要相应的改变。如下所示

当然我们还可以进行其他变换
比如我们可以考虑4个角点
f(x+1,y)+f(x-1,y)-2f(x, y)+f(x,y+1)+f(x,y-1)-2f(x, y)=
这是两个横纵对角线，我们考虑剩下的两个正反斜对角线=
f(x+1,y)+f(x-1,y)-2f(x, y)+f(x,y+1)+f(x,y-1)-2f(x, y)+f(x-1,y-1)+f(x+1,y+1)-2f(x,y)+f(x+1,y-1)+f(x-1,y+1)-2f(x,y)
f(x+1,y)+f(x-1,y)+f(x,y+1)+f(x,y-1)+f(x-1,y-1)+f(x-1,y+1)+f(x+1,y-1)+f(x+1,y+1)-8f(x,y)=
1  1  1
1 -8  1
1  1  1
以上公式的变化可以理解为：我们在考虑图像中心点横轴和纵轴的二阶微分的基础上考虑了两个斜对角轴
的二阶微分。滤波扫描到的每一个中心点的每一个轴的二阶微分等于前一个像素点的值加上后一个像素点的值
减去二倍的中心点的像素值，对应的我们可以直观的想到，仅考虑横纵轴的二阶微分的滤波形式为
0  1  0
1 -4  1
0  1  0

那么考虑两个斜对角轴的二阶微分后就是上上矩阵


使用非锐化模板锐化原图像
非锐化模板=原图像-模糊后的图像
图像的模糊会使得原图像中边缘像素点更加平缓，使得信号棱角变得平缓
由此造成的现象就是降低边缘的高像素点，增加边缘的低像素点
因此，使用原图减去模糊后的图像会产生一个类似心电图的波纹，该条波纹上
有起有伏，对应的结果就是非锐化模板，我们使用该模板加上原图像就会生成锐化后的图像
g(x, y) = f(x, y) - f_(x, y)
非锐化模板=原图像-模糊图像
g`(x, y) = f(x, y) + k * g(x, y)
k是权重系数，k>=0
K=1时， 我们称之为非锐化掩蔽
k>1时，我们称之为高提升滤波
k<1时，我们几乎可以忽略非锐化模板的贡献
因此，我们只需要记住三个名词：非锐化模板、非锐化掩蔽、高提升滤波
非锐化模板实质上就是图像的边缘
将非锐化模板直接加到原图上其实就是达到图像边缘增强的效果
将非锐化模板的k(k>1)倍加入到图像上其实就是高提升滤波

注意一阶微分和二阶微分的基础公式
df(x)/dx = f(x+1)-f(x)
d^2f(x)/dx2 = f(x+1) + f(x-1) - 2f(x)

平滑空间滤波：低通均值滤波，低通终值滤波
锐化空间滤波：
    二阶微分锐化（拉普拉斯算子）
    非锐化掩蔽和高提升滤波（非锐化模板由原图减去模糊图像得到）
    非锐化掩蔽等于原图加上非锐化模板，高提升滤波等于原图加上K倍的非锐化模板，K大于1

使用梯度对图像进行锐化，也即一阶微分
类似于上面提到图像的二阶微分，图像的一阶微分类似，一般有纵轴和横轴两个方向的梯度组成图像的二维梯度
也即图像的一阶梯度是一个二维列向量，[g_x, g_y]^-1, [df/dx = df/dy]^-1, 
向量[gx, gy]^-1的幅度值是M(x, y)=(g_x^2+g_y^2)^(1/2)
该幅度值是梯度方向变化率在(x, y)处的值。每个像素点对应着一个幅度值，它是当x和y允许
在图像中的所有像素位置变化时产生的。在实践中，该幅度值组成的图像称为梯度图像。

梯度向量[g_x, g_y]的分量g_X, g_y分别是x轴和y轴的微分。因此是线性算子g_x = f(x+1)-f(x)
g_y = f(y+1)-f(y)。然而，该向量的幅度值却是非线性的，因为涉及到平方根的操作。
因此我们可以找到一个和幅度值接近的线性算子。即找到和(g_x^2+g_y^2)^(1/2)值类似的一个线性算子
M(x, y) = (g_x^2+g_y^2)^(1/2) ≈ |g_x| + |g_y|
虽然这两个值不近似，但是特性近似。因此在特性方面对结果并无影响。

那么让我们简化我们的问题：现在的图像梯度从平方根简化为M(x, y) = |g_x| + |g_y|
我们像上面处理二阶微分一样对该公式定义一个离散近似，由此形成合适的滤波模板。
现在比如原图像的像素坐标点如下，和二阶微分时候的处理形式一样。
f(x-1,y-1) f(x-1,y) f(x-1,y+1)
f(x,y-1)   f(x,y)   f(x,y+1)
f(x+1,y-1) f(x+1,y) f(x+1,y+1)

对应的，我们进一步简化下我们的问题，使用更简单的符号来表示原始图像像素点
z1  z2  z3
z4  z5  z6
z7  z8  z9
对于图像的中心点z5，其对应的一阶微分是向量[g_x, g_y], 其中g_x = z8 - z5, g_y = z6 - z5
M(x, y) = [(z8 - z5)^2 + (z6 - z5)^2]^(1/2) ≈ |z8 - z5| + |z6 - z5|
然而我们一般交叉差分的方法。使用交叉形式求差分，而不是纵轴和横轴的办法
即M(x, y) = [(z9 - z5)^2 + (z8 - z6)^2]^(1/2) ≈ |z9 - z5| + |z8 - z6|
基于此我们可以得到对应的罗伯特交叉梯度算子
-1  0       0   -1
0   1       1   0
左侧主要检测垂直边缘，右侧主要检测水平边缘。
为什么不使用一个矩阵呢？
-1   -1
1     1

让我们来实际操作下看下一个矩阵和两个矩阵的差异
首先，为什么我们要计算g_x, g_y。主要是因为我们需要计算垂直的梯度和水平的梯度
50    50   100   100
100   100  150   150
150   150  200   200
200   200  250   250

首先使用罗伯特的算子进行滤波，左侧为左侧算子滤波后的结果，右侧是右侧算子滤波后的结果
50   100   50          50    0    50
50   100   50          50    0    50
50   100   50          50    0    50
我们可以发现左右侧算子都检测到了垂直的边缘，并没有检测到水平的边缘。
如果我们直接使用矩阵算子呢？我们可以直接得到这样的结果，如下
它其实就是左侧算子滤波结果和右侧算子滤波结果之和
可以发现，它隐藏了垂直边缘
100    100   100
100    100   100
100    100   100
因此为什么要使用两个算子而不是一个聚合后的算子呢？是因为一个算子可能会遗漏垂直或者水平的边缘信息
但是这样的算子并不是我们想要的，因为一般的滤波都是奇数模板，我们可以很容易找到中心点
而偶数模板没有滤波中心点。因此我们还要想办法将罗伯特算子转换为奇数滤波核

现在我们抛开罗伯特交叉算子，也即交叉差分的思想，我们单纯的使用奇数滤波核也即3*3滤波来
近似图像的梯度，我们分别计算图像的水平梯度和垂直梯度，也对应的检测图像的水平边缘和垂直边缘
3*3 模板对应的垂直梯度离散计算表达式是
g_x = (z8-z2)这个计算的是中心点z5的垂直梯度，当然，基于前面的计算，我们可以
考虑计算两个对角轴的梯度。即g_x = (z8-z2) + (z9-z1) + (z7-z3), 然而，这里我们
为了突出中心点的重要性，我们对其赋予一定的权重，比如赋予垂直梯度较两个斜轴梯度更高的权重
比如g_x = 2*(z8-z2) + (z9-z1) + (z7-z3)，因此可以得到对应的垂直梯度算子为
-1   -2    -1
0     0     0
1     2     1

以此，我们可以计算水平的梯度算子为：
-1   0    1
-2   0    2
-1   0    1
基于以上理论得到的两个算子就是soble算子
可以发现罗伯特算子和soble算子的系数和均为0.算子系数和为0的意义是什么呢？
首先，算子系数和为零，在实际滤波的时候，如果滤波区域是灰度恒定的区域，那么对应的响应值为0
也即灰度恒定区域的增强值为0，即不进行增强。这正是我们需要的。

到此为止我们使用一阶微分，也即梯度的理论分别构建了罗伯特交叉算子和soble算子，
这两个算子都分别拥有垂直和水平边缘检测两个算子。可以用来检测图像的边缘
有了检测到的图像的边缘以后，我们可以在原图的基础上加上边缘图以实现灰度增强。
基于以上分析，梯度可以用来增强缺陷并且清除慢变化的背景。
同时梯度还可以用于突出图像中看不见的小斑点。比如一个很亮的图像存在某一个小斑点
我们可以计算图像的梯度，梯度图像会清除慢变化区域也即将光亮区域灰度值置为0，而增强
暗亮交界区灰度值和小斑点。因此，在灰度平坦区域中增强小突变的额能力是梯度处理的另一个特性。


混合空间增强法。
我們如何将本章学习到的图像增强方法结合起来？
首先我们全面了解下图像的拉普拉斯和梯度增强的区别
拉普拉斯主要用来增强图像的纹理和细节，同时也会增强图像的噪声，因此一般在对
图像使用拉普拉斯增强的时候首先要降噪处理
梯度锐化主要用来增强图像的边缘。不会增强图像的纹理和细节。

其实就是二阶微分和梯度的区别，二阶微分较一阶微分增强的边缘的宽度更低。因为
在图像灰度斜坡处，一阶微分可以是正值，而二阶微分却是0.
这里我们需要了解图像的遮罩处理。图像遮罩是指我们可以利用二值遮罩图像
将我们感兴趣的区域保护起来，具体操作如下：
根据原图和感兴趣的区域创建对应的遮罩图像，感兴趣区域的像素点为0，其他区域的像素点为1.
然后将原图和遮罩图像点乘。然后对相乘后的图像进行处理，如提高亮度，对比度等。

那么对于一张灰度图像，我们可以进行的混合空间增强步骤包括：
1 使用拉普拉斯突出图像中的小细节（也即进行图像锐化）：原图 + 拉普拉斯图像 达到图像锐化效果（突出细节）
2 然后用梯度法突出其边缘（边缘增强）使用soble算子
3 然后使用均值滤波平滑2的结果
4 然后将锐化图像（1）和平滑图像（3）相乘得到掩蔽图像
5 然后将原图和掩蔽图像（4）相加得到锐化后的图像
6 然后对图5应用幂律变换得到最终的结果（幂律变换可以提升图像的对比度，将低灰度区域映射到更高的灰度区域）
注意以上操作。
我们首先是想要进行图像增强，然后我们首先使用了拉普拉斯变换（二阶微分），在原图的基础上考虑拉普拉斯
变换后得到增强图像的细节和纹理图，但是有一个缺陷就是增强后的图像的噪声非常明显，特别是当噪声出现
在平滑区域内的时候，此时，二阶微分不是一个很好的解决办法，我们考虑使用梯度进行图像的边缘增强，因为
梯度的边缘增强不会增强图像的噪声。二阶微分在增强细节方面很好。
而梯度变化在灰度变化的区域也即斜坡区域的平均相应要比拉普拉斯操作的平均响应更强烈。梯度操作对噪声和
小细节的响应要比拉普拉斯操作的响应弱，但是对于灰度变化区域（也即大细节）比拉普拉斯强。
并且我们可以在此基础上使用均值滤波平滑梯度增强后的图像来进一步降低图像的噪声和小细节。

此时我们可以直接将上一步的结果（梯度变换+中值滤波（边缘增强+弱化噪声和小细节），也可以称之为
非锐化模板）加到原图上，可以得到非锐化掩蔽或者高提升滤波

但是有一个问题，我们想要同时利用拉普拉斯变换（二阶微分）和sobel算子（梯度变换）
也即同时利用他俩的优势进行图像锐化，那么此时我们可以在加上原图之前在
梯度变换+中值滤波的基础上考虑拉普拉斯变换，也即（对梯度图像进行中值滤波）*拉普拉斯锐化图像（拉普拉斯变换+原图）
这里得到的结果是什么。其实是一个掩蔽图像，因为乘积会保留灰度变化强烈区域的细节，同时降低
灰度变化相对平坦区域的噪声（为什么？）然后将这个结果加上原图像就可以得到最终的锐化图像。

而且注意一点，梯度图像会突出图像的边缘，而二阶微分图像也即拉普拉斯图像则会突出图像的细节和纹理
同时，梯度值一般比拉普拉斯的灰度值高，因此梯度图比拉普拉斯图亮度高。
同样的使用拉普拉斯锐化后的图像乘以平滑后的梯度图得到的掩蔽图像的基础上加上原图得到的最终锐化后的
图像，是单独使用拉普拉斯变换和梯度增强边缘达不到的效果。

最后，锐化后的图像并没有改变图像灰度的动态范围，因此最后一步就是增大锐化后图像的动态范围，
直方图均衡化、规定化和幂律变换都可以达到扩大动态范围的效果，但是幂律变换更加通用且
适合该应用场景。

注意我们可以使用extern在头文件中修饰声明变量，但是注意一定要在源文件中定义该变量
否则将会出现重复定义的错误。然后可以直接在其他源文件中使用该变量。
前提是需要包含该头文件即可。
// ------------------------------------------------------------------------------------







































































































































***********************************************************************/