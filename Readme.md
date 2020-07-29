# 文件说明

* k8sop.py：对k8s集群进行操作的功能函数
* modelBased.py：使用model-based强化学习方法进行工作的函数，为对论文Horizontal and Vertical Scaling of Container-based Applications using Reinforcement Learning的复现。
* FetchData.py：用于从prometheus中抓取数据的脚本（其中CPU占用率脚本有点问题，但是为了与之前的数据相容，故不改变）
* reader.py：读取抓取数据，绘制图像并进行分析

下一步：预测式调度

* test.py：使用CPU占用率和响应时间进行预测，使用机器学习算法，最终发现GBDT效果最好
* predict.py：使用训练好的GBDT进行资源预测，固定参数的ARMA进行流量预测，提前进行sarsa调度