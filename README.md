# Using Scene Memory Transformer and CLIP to Solve Object Navigation End-to-end
Jing Wang*, Tianchu Zhang*, Ningyuan Zhang

∗Equal contribution.
## 介绍
:star2: :star2: 我们的端到端视觉导航智能体Walle***在Meta AI 2022年举办的[Habitat物体导航比赛](https://aihabitat.org/challenge/2022/)(Habitat Object Navigation) 中获得第三名，成功率61%***。项目前三名的介绍视频***在[NeurIPS 2022 Habitat Workshop](https://www.youtube.com/watch?v=qo2CQ1WMTFs)上由比赛主办方展出***
 (原定于在CVPR2022 的[Embodied AI workshop](https://embodied-ai.org/cvpr2022)上展出，由于主办方数据集问题推迟到同年的NeurIPS)。:star2: :star2: 

比赛官网链接：https://aihabitat.org/challenge/2022/

任务：每个回合开始，agent被随机初始化在一个位置环境中的随机位置和朝向，给他提供例如“找到一把椅子”的指令，它需要导航去找到某个对应该类别的物体。agent不知道环境的地图，只有自身的一些传感器信息。
agent配备了一个RGB-D摄像头和一个精确的位置传感器。位置传感器提供agent相对于初始位置时的平面位置和朝向。agent可以执行的动作有：向前走一步，左右转30度。agent需要自行判断是否找到了目标物品。回合成功的条件是agent在目标周围1米范围内并且认为找到了目标。

目标物品：一共有六个目标类别，分别是椅子，沙发，盆栽，床，厕所和电视。

数据集和仿真：
仿真使用了Meta开发的Habitat-Lab。训练集为HM3D数据集里80个3D扫描建模的真实房间。比赛最终在20个没见过的房间里通过1000个回合来评判智能体。

<!-- :star2: :star2: ***项目前三名的介绍视频在NeurIPS 2022 workshop上由比赛主办方展出(原定于在CVPR2022 的Embodied AI workshop上展出，由于主办方数据集问题推迟到NeurIPS)。*** -->
以下是我们训练好的智能体在Habitat仿真里没见过的房间导航的效果视频：
<div align=center>
<video src="https://github.com/Chortine/Habitat-Object-Navigation/assets/107395103/f463a626-8ffc-4b85-a662-cc56e85ed7ac" /></a>
</div>

## 比赛结果
主要评判标准有成功率以及SPL(Success weighted by Path Length)，其中SPL定义如下，主要用来评判成功路径的效率：

<div align=center>
<img src="https://github.com/Chortine/Habitat-Object-Navigation/assets/107395103/d9c8fac8-c50f-482b-8299-0d3eb7f696fb" width="450"/>
</div>

最终我们的智能体Walle在从未见过的房间里找寻目标物品，成功率达到61%，在最终17个提交的队伍中获得第三名，仅比第一名低3%。下图排名来源于比赛官网：

<img src="https://github.com/Chortine/Habitat-Object-Navigation/assets/107395103/d187ae41-e8bc-460a-95ea-3d84975958f3" height="490"/>

<!-- ![image](https://github.com/Chortine/Habitat-Object-Navigation/assets/107395103/d9c8fac8-c50f-482b-8299-0d3eb7f696fb) -->


## 网络结构

<div align=center>
<img src="https://github.com/Chortine/Habitat-Object-Navigation/assets/107395103/6ccf1862-c30e-4b64-9db1-bcf118a97b00" width="660"/>
</div>

我们使用了encoder-decoder Transformer的架构来处理时序信息，并且使用了预训练的CLIP[2] encoder来处理视觉信息。其中，网络的输入有：
当前和历史的RGB-D，目标物体类别的embedding，机器人位姿，轨迹俯视图，轨迹和深度图合成的占用栅格地图。输出是双头的action，分别表示移动的动作，以及判断是否找到目标并终止回合。

## 核心设计
1. 网络结构：
我们比较了LSTM， GRU，和Transfomer对导航相关的记忆的处理能力，最后选择了transformer。在Transformer的使用上，我们参考了Kuan Fang[1]的一篇工作。其使用了经典的Vanilla Transformer架构，其中，encoder的部分主要用来从记忆buffer中提取时间轴上的注意力信息，来维护一个隐式的空间地图。decoder则负责用当前的观测作为query，来从隐式地图中提取相关信息（例如，是否到过类似的地点等）。

3. 视觉 Encoder：
我们比较了普通Resnet，R3M Encoder, 和 CLIP Encoder。最后使用了CLIP的ViT-B/32作为视觉Encoder。它在图像文字对比学习的任务下做的预训练，能更好地提取图片中的语义信息。

3. 除了时间轴上的attention之外，还增加了跨Feature之间的attention。

4. 将历史轨迹转化成2D俯视图，并和历史的深度图一并转化成2D占用栅格地图输入。

5. 比较了不同的辅助任务，比如说额外学习Visual Odometry，学习视觉中的语义信息等。

6. 训练上采用了分布式PPO，并行数百个仿真环境采样。类似于IMPALA架构，采样和训练异步进行，保证采样效率。

## 项目视频
在以下视频中，我们详细介绍了我们的方法：
<div align=center>
<a href="https://www.bilibili.com/video/BV1Eh4y1475R/?spm_id_from=333.999.0.0"><img src="./resource/img_bilibili.png" height="300"/></a>
</div>

## 开放讨论
网络如何具备空间感知？ 在我们的测试中，发现agent会出现原地转圈等现象，调整奖励函数还是难以消除这个现象，我们认为，现在的网络结构设计不足以让agent形成稳定的空间表征。后续值得探索的话题有：怎么定义空间表征？怎么设计网络/训练方式，来给agent加入归纳偏置帮助它更好学习空间感知。空间表征包括对一个隐式地图的建立，维护，以及信息提取的过程。有一些可以参考的方法，比如说Nerf和OSRT[3]这样具有三维旋转平移不变性的网络或许能成为很好的encoder。

## 参考文献
[1] [SMT](https://arxiv.org/abs/1903.03878)
[2] [CLIP](https://github.com/openai/CLIP)
[3] [OSRT](https://osrt-paper.github.io/)
