# 本脚本能够读取监控数据并产生图像和其他计算指标
# 使用方法：与监控数据文件夹放在一起。

import os

# dirName = ['qlearning-1','qthreshold-1','sarsa-arima-1','qlearning-2','qthreshold-2','sarsa-arima-2']
dirName = ['0_workload1']

cpuFilename = 'cpuUtilization.log'
responseFilename = 'responseTime.log'
throughputFilename = 'requestRate.log'
supplypodFilename = 'podSupply.log'

anscpuList = []
ansresponseList = []
ansthroughputList = []
anssupplypodList = []

for dirname in dirName:
    cpuReader = open(dirname+'/'+cpuFilename,'r')
    responseReader = open(dirname+'/'+responseFilename,'r')
    throughputReader = open(dirname+'/'+throughputFilename,'r')
    supplypodReader = open(dirname+'/'+supplypodFilename,'r')
    
    cpuList = cpuReader.readlines()
    responseList = responseReader.readlines()
    throughputList = throughputReader.readlines()
    supplypodList = supplypodReader.readlines()
    
    cpuReader.close()
    responseReader.close()
    throughputReader.close()
    supplypodReader.close()
    
    #将行尾带有\n的文本转成float
    def floatTranslator(element):
        return float(element[:-1])
    
    #将行尾带有\n的文本转成int
    def intTranslator(element):
        return int(element[:-1])
    
    anscpuList.append(list(map(floatTranslator,cpuList)))
    ansresponseList.append(list(map(floatTranslator,responseList)))
    ansthroughputList.append(list(map(floatTranslator,throughputList)))
    anssupplypodList.append(list(map(intTranslator,supplypodList)))

from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
import matplotlib.pyplot as plt
import numpy as np

for botton in range(len(dirName)):
    #设置三个轴，并添加它们
    # fig = plt.figure(figsize = (20,10))
    fig = plt.figure(figsize=(20, 10))
    host = HostAxes(fig, [0.15, 0.1, 0.65, 0.8])
    # host = HostAxes(fig, [0.15, 0.1, 0.65, 0.8])  #用[left, bottom, weight, height]的方式定义axes，0 <= l,b,w,h <= 1
    axis_cpu = ParasiteAxes(host, sharex=host)
    axis_response = ParasiteAxes(host, sharex=host)
    axis_container = ParasiteAxes(host, sharex=host)
    axis_throughput = ParasiteAxes(host, sharex=host)
    host.parasites.append(axis_cpu)
    host.parasites.append(axis_response)
    host.parasites.append(axis_container)
    host.parasites.append(axis_throughput)
    #关闭主图的左右上轴线，打开辅轴的右边线
    host.axis['left'].set_visible(False)
    host.axis['right'].set_visible(False)
    host.axis['top'].set_visible(False)
    
    #设置各个轴线标签
    host.set_xlabel('Time (20s)')
    axis_cpu.set_ylabel('CPU Utilization')
    axis_response.set_ylabel('Response Time')
    axis_container.set_ylabel('Number of Instances')
    axis_throughput.set_ylabel('Throughput')
    #添加辅轴
    cpu_axisline = axis_cpu._grid_helper.new_fixed_axis
    response_axisline = axis_response._grid_helper.new_fixed_axis
    container_axisline = axis_container._grid_helper.new_fixed_axis
    throughput_axisline = axis_throughput._grid_helper.new_fixed_axis
    axis_cpu.axis['right2'] = cpu_axisline(loc='right',axes=axis_cpu,offset=(0,0))
    axis_response.axis['right3'] = response_axisline(loc='right',axes=axis_response,offset=(40,0))
    axis_container.axis['right4'] = container_axisline(loc='right',axes=axis_container,offset=(80,0))
    axis_throughput.axis['right5'] = throughput_axisline(loc='right',axes=axis_throughput,offset=(120,0))
    #将主轴加至原图
    fig.add_axes(host)
    #添加数据
    
    plot_cpu, = axis_cpu.plot(anscpuList[botton],label='utilization',color='#3366FF')
    plot_response, = axis_response.plot([250 for i in range(1000)],color = 'black' , linestyle = '--') # SLA violation定为250ms
    plot_response, = axis_response.plot(ansresponseList[botton],label='response time',color='#CC3333')
    plot_container, = axis_container.plot(anssupplypodList[botton],label='container',color='green')
    plot_throughput, = axis_throughput.plot(ansthroughputList[botton],label='throughput',color='orange')
    #通过范围限制调整数据在图中的位置(一个是调整上界，可以抬升；一个是调整占比比例，可以让其出现在想出现的区域)
    axis_cpu.set_ylim(0,100)
    axis_response.set_ylim(-400,400)
    axis_container.set_ylim(0,40)
    axis_response.set_yticks([0,100,200,300,400,500,600])
    
    #右侧轴线颜色与格式设置
    axis_cpu.axis['right2'].set_axisline_style('-|>',size=1.5) #设置轴线样式
    axis_cpu.axis['right2'].line.set_color(plot_cpu.get_color()) #轴线上色
    axis_cpu.axis['right2'].major_ticklabels.set_color(plot_cpu.get_color()) #刻度值上色
    axis_cpu.axis['right2'].label.set_color(plot_cpu.get_color())
    
    axis_response.axis['right3'].set_axisline_style('-|>',size=1.5) #设置轴线样式
    axis_response.axis['right3'].line.set_color(plot_response.get_color()) #轴线上色
    axis_response.axis['right3'].major_ticklabels.set_color(plot_response.get_color()) #刻度值上色
    axis_response.axis['right3'].label.set_color(plot_response.get_color())
    
    axis_container.axis['right4'].set_axisline_style('-|>',size=1.5) #设置轴线样式
    axis_container.axis['right4'].line.set_color(plot_container.get_color()) #轴线上色
    axis_container.axis['right4'].major_ticklabels.set_color(plot_container.get_color()) #刻度值上色
    axis_container.axis['right4'].label.set_color(plot_container.get_color())
    
    axis_throughput.axis['right5'].set_axisline_style('-|>',size=1.5) #设置轴线样式
    axis_throughput.axis['right5'].line.set_color(plot_throughput.get_color()) #轴线上色
    axis_throughput.axis['right5'].major_ticklabels.set_color(plot_throughput.get_color()) #刻度值上色
    axis_throughput.axis['right5'].label.set_color(plot_throughput.get_color())
    
    plt.savefig(dirName[botton]+'.png')

# 计算SLA violation ratio

import numpy as np
cpuArray = np.array(anscpuList[0])
podArray = np.array(anssupplypodList[0])
resArray = np.array(ansresponseList[0])

print('SLA违约率：',np.sum(resArray>250)/len(resArray))
print('CPU平均利用率：',np.average(cpuArray))
print('容器资源使用率：',np.average(podArray))