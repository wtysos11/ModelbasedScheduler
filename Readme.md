# 文件说明

* k8sop.py：对k8s集群进行操作的功能函数
* modelBased.py：使用model-based强化学习方法进行工作的函数，为对论文Horizontal and Vertical Scaling of Container-based Applications using Reinforcement Learning的复现。
* FetchData.py：用于从prometheus中抓取数据的脚本（其中CPU占用率脚本有点问题，但是为了与之前的数据相容，故不改变）
* reader.py：读取抓取数据，绘制图像并进行分析