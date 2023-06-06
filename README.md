# Using Scene Memory Transformer and CLIP to Solve Object Navigation End-to-end
JingWang, TianchuZhang, NingYuan

## 介绍
Habitat
@ 
[![](https://bb-embed.herokuapp.com/embed?v=BV1jS4y1w7SW)](https://player.bilibili.com/player.html?aid=614125538&bvid=BV1Eh4y1475R&cid=1146214248&page=1)

[//]: # ([![Image text]&#40;https://github.com/Chortine/Habitat-Object-Navigation/blob/main/resource/img_bilibili.png&#41;]&#40;https://www.bilibili.com/video/BV1Eh4y1475R/?spm_id_from=333.999.0.0&#41;)

[comment]: <> ([<img height="216" src="./resource/img_bilibili.png" width="380"/>]&#40;https://www.bilibili.com/video/BV1Eh4y1475R/?spm_id_from=333.999.0.0&#41;)

<div align=center>
<a href="https://www.bilibili.com/video/BV1Eh4y1475R/?spm_id_from=333.999.0.0"><img height="216" src="./resource/img_bilibili.png" width="380"/></a>
</div>




## 网络结构

<div align=center>
<img height="300" src="https://github.com/Chortine/Habitat-Object-Navigation/assets/107395103/6ccf1862-c30e-4b64-9db1-bcf118a97b00" width="490"/>
</div>
<!-- ![image](https://github.com/Chortine/Habitat-Object-Navigation/assets/107395103/6ccf1862-c30e-4b64-9db1-bcf118a97b00) -->

## Key designs that contribute to final results
1. 网络结构
2. CLIP Encoder
3. 数据标注


## 开放讨论
1. 网络如何具备空间感知？ 在我们的测试中，发现agent会出现原地转圈等现象，调整奖励函数还是难以消除这个现象，我们认为，现在的网络结构设计不足以让agent形成稳定的空间表征。后续值得探索的话题有：怎么定义空间表征？怎么设计网络/训练方式，来给agent加入归纳偏置帮助它更好学习空间感知。
2. 有一些可以参考的方法，比如说Nerf和OS
