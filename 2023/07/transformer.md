# Transformer

### 基础知识

##### 1. 模型结构

包括多头Attention、add&layer_norm、FFN

##### 2. 计算公式及参数量

1. Self-attention:

   - 计算公式

   $$
   att = softmax(\frac{QK^T}{\sqrt{d_k}})V
   $$

   - 参数量计算：以12头为例
     - 单头：$768*768/12*3$，3个参数矩阵$W_Q$、$W_K$、$W_V$
     - 12个头concat：$768*768$，输出参数矩阵$W_O$
     - 总计：$768*768*4$

2. FFN

   - 计算公式
     $$
     FFN(x) = max(0, W_1x+b_1)W_2+b_2
     $$

   - 参数量计算：

     - $768*3072*2​$，we set the feed-forward/filter size to be 4H ，2个参数矩阵 

3. Layer-norm

   1. LayerNorm是大模型也是transformer结构中最常用的归一化操作，简而言之，它的作用是 *对特征张量按照某一维度或某几个维度进行0均值，1方差的归一化* 操作，计算公式为： 
      $$
      y=\frac{x-E(x)}{\sqrt{Var(x)+\epsilon}}*\gamma+\beta
      $$
      

##### 3.  大模型**Normalization** 

normalization的目的是将输入转化为均值为0方差为1的数据，缓解Internal Covariate Shift问题，可以**将数据分布拉到激活函数的非饱和区，具有权重/数据伸缩不变性的特点。起到缓解梯度消失/爆炸、加速训练、正则化的效果。**

1. **pre-norm和post-norm区别**

   ![img](http://www.sniper97.cn/wp-content/uploads/2023/05/image-29-1024x531.png) Post-Norm由于是在残差之后进行归一化，因此归一化的效果更好，使得模型的鲁棒性更强。Pre-Norm由于并不是所有的参数都参与正则化，因此整体来说更不容易发生梯度消失的问题，模型训练的稳定性更强。

   因此，在Bert时代由于层数较浅，往往采用的是Post-Norm，而到了大模型时代，由于transformer的层数开始加深，为了训练稳定性开始使用Pre-Norm。

2. **Layer-norm、batch-norm的区别**

   LN与BN不同，Layer Normalization在每一个样本（单一样本）上计算均值和方差，然后做归一化，多用于RNN这种输入序列是变长的模型中，每一个时间步都有自己的分布。同一层的所有神经元具有相同的均值和方差。对于使用LN的RNNs，**每个时刻加权后的输入通过标准化被重新调整在合适的范围**，很大程度避免了梯度消失、梯度爆炸问题，隐藏状态的传递更加稳定。

3. **Deep-norm**

   基于post-norm，残差之后进行归一化，残差连接时对x加了个权重$\alpha$。![img](https://pic1.zhimg.com/80/v2-8dc272cc28931f3eaac0d2f5b5b15240_1440w.webp) 

   相比于post-norm，在归一化前会扩大残差连接($\alpha$)，在Xavier参数初始化过程中以$\beta$减小部分参数的初始化范围 初始化降低了参数的比例。

   ​		$$x_{l+1}=LN(\alpha x_l+G_l(x_l,\theta _l))$$

   只扩大前馈网络的权值和attention层投影值。

4. **RMS-norm**（[Root Mean Square Layer Normalization](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1910.07467)） 

   与layerNorm相比，RMS Norm的主要区别在于去掉了减去均值的部分，计算公式为：
   $$
   \bar{a_i}=\frac{a_i}{RMS(a)}g_i,  RMS(a)=\sqrt{\frac{1}{n}\sum^n_{i=1}{a_i^2}}
   $$
   

   这里的$a_i$与Layer Norm中的$x$等价，作者认为这种模式在简化了Layer Norm的同时，可以在各个模型上减少约 7%∼64% 的计算时间

#####  . 模型结构图

![img](https://img-blog.csdnimg.cn/2021052223091261.PNG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JlbnpodWppZTEyNDVjb20=,size_16,color_FFFFFF,t_70)