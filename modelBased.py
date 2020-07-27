from k8sop import K8sOp
import urllib.parse
import requests
import time
import copy
import operator
import numpy as np
import pandas as pd
import math
import random
import logging
 
def console_out(logFilename):
    ''' Output log to file and console '''
    # Define a Handler and set a format which output to file
    logging.basicConfig(
                    level    = logging.DEBUG,              # 定义输出到文件的log级别，                                                            
                    format   = '%(asctime)s  %(filename)s : %(levelname)s  %(message)s',    # 定义输出log的格式
                    datefmt  = '%Y-%m-%d %A %H:%M:%S',                                     # 时间
                    filename = logFilename,                # log文件名
                    filemode = 'w')                        # 写入模式“w”或“a”
    # Define a Handler and set a format which output to console
    console = logging.StreamHandler()                  # 定义console handler
    console.setLevel(logging.INFO)                     # 定义该handler级别
    formatter = logging.Formatter('%(asctime)s  %(filename)s : %(levelname)s  %(message)s')  #定义该handler格式
    console.setFormatter(formatter)
    # Create an instance
    logging.getLogger().addHandler(console)           # 实例化添加handler

template = {
    "cpu": "(sum(sum(rate(container_cpu_usage_seconds_total{{namespace='{0}',pod_name=~'{1}.*'}}[30s])) by (pod_name, namespace)) / sum(container_spec_cpu_quota{{namespace='{0}',pod_name=~'{1}.*'}} / 100000)) * 100",
    "res": "sum(delta(istio_request_duration_seconds_sum{{destination_workload_namespace='{0}',reporter='destination',destination_workload='{1}',response_code='200'}}[30s]))/sum(delta(istio_request_duration_seconds_count{{destination_workload_namespace='{0}',reporter='destination',destination_workload='{1}', response_code='200'}}[30s]))*1000",
    "pod": "count(sum(rate(container_cpu_usage_seconds_total{{namespace='{0}',pod_name=~'{1}.*'}}[15s])) by (pod_name, namespace))",
    "req": "sum(rate(istio_requests_total{{destination_workload_namespace='{0}',reporter='destination',destination_workload='{1}'}}[30s]))"
}

prefix_api = "http://139.9.57.167:9090/api/v1/query?query="

def fetch_data(index,namespace,svc_name):
    '''
    根据index + (命名空间，实例名)获得响应时间
    '''
    if index not in template:
        return -2
    else:
        api = template[index].format(namespace,svc_name)
    res = requests.get(prefix_api + urllib.parse.quote_plus(api)).json()["data"]
    if "result" in res and len(res["result"]) > 0 and "value" in res["result"][0]:
        v = res["result"][0]["value"]
        if v[1] == 'NaN':
            print('value is NaN')
            return -1.0
        return float(v[1])
    return -1.0

namespace = 'test'
svc_name = 'cproductpage'

# 检查初始时容器实例数
k8s_op = K8sOp()
podNum = k8s_op.get_deployment_replicas(svc_name, namespace)
sla_up = 250 #上界250ms
sla_down = 50 #下界50ms
update_times = 1 #每次更新Q表的次数

# 离散化采用固定离散化
# 响应时间单位为25ms，最大为300ms 12个
resInterval = 25
# CPU占用率单位为2% ，最大为30% 15个
cpuInterval = 2

# epsilon = 1/t
minEpsilon = 0.05 # epsilon必须大于0.05
# learning rate = 0.1
resMax = 12
cpuMax = 15
conMax = 5 # 容器状态数，最低允许2个，因此实际最大为6个
actionMax = 5

actionBias = 2 #动作表偏向为2，因此需要-2才能得到正确的动作
minContainer = 2
maxContainer = conMax + minContainer - 1 
# 主要算法构建
# Q表、P表和C表

q_table = np.zeros((resMax*cpuMax*conMax,actionMax))
p_table = np.zeros((resMax*cpuMax*conMax,actionMax,resMax,cpuMax)) # s,a -> s'
c_table = np.zeros((resMax*cpuMax*conMax))

qAlpha = 0.1 #Q表学习率
cAlpha = 0.1 #c表学习率


#读取文件
#读取文件
#读取文件

# 功能函数部分，已经经过检验
def indexEncoding(resIndex,cpuIndex,replicasNum):
    '''
        进行编码
        resIndex有12个
        replicasNum >= 2,replicasIndex:0~4
    '''
    return int(resIndex + cpuIndex*resMax + replicasNum*resMax*cpuMax)

def stateDecompse(state):
    '''
    已经知道状态编码，要进行拆分
    '''
    replicasNum = math.floor(state/(resMax*cpuMax))
    cpuIndex = math.floor((state - replicasNum*resMax*cpuMax) / resMax)
    resIndex = (state - replicasNum*resMax*cpuMax) % resMax
    return (resIndex,cpuIndex,replicasNum)


if __name__ == "__main__":
    '''
    转换器：state(实例数，CPU占用率，响应时间)->1维
    
    Q表：numpy数组，state*action，每个项目存的Value表示代价，选择代价最小的作为最佳动作
    P表：使用P(s'|a,s), 直接估算
    C表：numpy数组，state个项目，每个项目存的是代价
    
    执行一次动作
        获取当前CPU占用率、响应时间、实例数，对相应连续变量离散化，组成状态S
        查询Q表，使用epsilon-greedy算法，计算出动作a
    
        动作执行后获得S'
        将相关数据写入到Q表、P表和C表中，更新
    
    更新一次项目后，更新全体Q表
    '''
    #初始化时，需要对Q表中的非法动作进行一边封禁，全部设为最大值10。
    #这样的目的是为了防止在实际选择的时候会尝试非法动作
    #注册logging
    console_out('logging.log')
    logging.debug('debug test')
    logging.info('model-based algorithm start')
    print('start')

    logging.debug('start init')
    maxQValue = 10000
    for resIndex in range(resMax):
        for cpuIndex in range(cpuMax):
            for conIndex in range(conMax):
                podNum = conIndex + minContainer
                state = indexEncoding(resIndex,cpuIndex,conIndex)
                for acIndex in range(actionMax):
                    action = acIndex - actionBias
                    if podNum + action < minContainer or podNum + action > maxContainer:
                        q_table[state][acIndex] = maxQValue

    logging.debug('end init')

    #主循环过程
    ## PS:第一次循环的时候采用动态epsilon，后面应该使用静态epsilon(0.05左右)
    cur_t = 0
    while True:
        cur_t += 1
        time.sleep(30)
        logging.info('\t \t main algorithm times start: '+ str(cur_t))
        # 执行一次动作
        ## 获取当前状态
        responseTime = fetch_data('res',namespace,svc_name)
        if responseTime == -1.:
            print('error when geting response time')
            continue
        cpuUtilization = fetch_data('cpu',namespace,svc_name)
        podNum = fetch_data('pod',namespace,svc_name)
        logging.info('\tcurrent state responseTime:'+str(responseTime)+'\tcpuUtilization:'+str(cpuUtilization)+'\tpodNum:'+str(podNum))
        ## 离散化，并进行编码
        resIndex = math.floor(responseTime/resInterval)
        cpuIndex = math.floor(cpuUtilization/cpuInterval)
        podIndex = int(podNum - minContainer)
        state = indexEncoding(resIndex,cpuIndex,podIndex)
        logging.debug('\tencoding current state resIndex:'+str(resIndex)+'\tcpuIndex:'+str(cpuIndex)+'\tpodIndex:'+str(podIndex)+'\tencoding:'+str(state))
        if resIndex < 0 or cpuIndex < 0 or podIndex < 0 or resIndex > resMax or cpuIndex > cpuMax or podIndex > conMax:
            logging.info('Index error')
            continue
        ## 使用epsilon_greedy计算最佳动作
        epsilon = 1/cur_t
        if epsilon < minEpsilon:
            epsilon = minEpsilon

        logging.info('epsilon:'+str(epsilon))
        action = 0 # 分别按照-2,-1,0,1,2往右对应0,1,2,3,4(action-2)
        if random.random() < epsilon:
            # 随机动作
            action = math.floor(random.random()*5)
            logging.info('random select action(0~4):'+str(action))
        else:
            # 最佳动作
            min_value = np.min(q_table[state])
            action = np.where(q_table[state] == min_value)[0][0]
            logging.info('choose min action:'+str(action))
            logging.debug(str(q_table[state]))

        ## 执行该动作
        last_pod = podNum + action - actionBias
        if last_pod < minContainer:
            last_pod = minContainer
        elif last_pod > maxContainer:
            last_pod = maxContainer
        last_action = last_pod - podNum + actionBias
        logging.info('last pod:'+str(last_pod)+' last_action:'+str(last_action))

        k8s_op.scale_deployment_by_replicas(svc_name,namespace,last_pod)
        logging.info('execute')

        ## 等待30s得到下一个结果，更新表格
        time.sleep(30)
        logging.info('execute over, get new state')

        responseTimeNext = fetch_data('res',namespace,svc_name)
        if responseTimeNext == -1.:
            print('error when geting response time')
            continue
        cpuUtilizationNext = fetch_data('cpu',namespace,svc_name)
        podNumNext = fetch_data('pod',namespace,svc_name)
        logging.info('\tcurrent state responseTime:'+str(responseTimeNext)+'\tcpuUtilization:'+str(cpuUtilizationNext)+'\tpodNum:'+str(podNumNext))
        ## 离散化，并进行编码，然后对新状态更新各个表格
        resIndexNext = math.floor(responseTimeNext/resInterval)
        cpuIndexNext = math.floor(cpuUtilizationNext/cpuInterval)
        podIndexNext = int(podNumNext - minContainer)
        stateNext = indexEncoding(resIndexNext,cpuIndexNext,podIndexNext)
        logging.debug('\tencoding current state resIndex:'+str(resIndexNext)+'\tcpuIndex:'+str(cpuIndexNext)+'\tpodIndex:'+str(podIndexNext)+'\tencoding:'+str(stateNext))
        # 计算代价
        costPerf = 0
        if responseTimeNext > sla_up:
            #costPerf = responseTimeNext - sla_up
            costPerf = 1
        elif responseTimeNext < sla_down:
            #costPerf =  sla_down - responseTimeNext
            costPerf = 1

        costApp = podNumNext/maxContainer
        totalCost = 0.9*costPerf + 0.1 * costApp
        logging.info('\ttotalCost:'+str(totalCost))

        # 更新表格
        p_table[state,action,resIndexNext,cpuIndexNext] += 1
        c_table[stateNext] = (1-cAlpha) * c_table[stateNext] + cAlpha * totalCost

        # 更新全部状态和动作，次数为n次
        for i in range(update_times):
            #枚举状态
            logging.info('update times:'+str(i))
            for s in range(resMax*cpuMax*conMax):
                for a in range(actionMax):
                    #判断是否存在值
                    nres,ncpu,npod = stateDecompse(s)
                    totalExpectation = 0
                    totalCount = np.sum(p_table[s][a])
                    if totalCount == 0: #下一状态全为0，则直接跳过
                        continue

                    conIndex = npod + a - actionBias
                    if conIndex < 0 or conIndex >= conMax: #illegal action
                        continue

                    logging.debug('ready to update state('+str(resMax)+','+str(cpuMax)+','+str(conMax)+','+str(a)+')')
                    #枚举所有s'不为0的点，得到一个列表
                    result = np.where(p_table[s][a] != 0)
                    result1 = result[0]
                    result2 = result[1]
                    # 对于每一个值，通过概率计算期望
                    for i in range(len(result1)):
                        resIndex = int(result1[i])
                        cpuIndex = int(result2[i])

                        probability = p_table[s][a][resIndex][cpuIndex]/totalCount
                        state = indexEncoding(resIndex,cpuIndex,conIndex)
                        cost = c_table[state]

                        minActionValue = np.min(q_table[state])

                        totalExpectation += probability * (cost + qAlpha * minActionValue)
                        logging.debug('Update state:('+str(resIndex)+','+str(cpuIndex)+')')
                        logging.debug('cost:'+str(cost))
                        logging.debug('q_table:'+str(q_table[state]))
                        logging.debug('pro:'+str(probability))
                        logging.debug('value : '+str(probability * (cost + qAlpha * minActionValue)))
                    logging.info('previous Q table:'+str(q_table[s][a]))
                    q_table[s][a] = totalExpectation
                    logging.info('current Q table:'+str(q_table[s][a]))

        # 缓存所有数组
        q_file = open('q_table.txt','wb')
        p_file = open('p_table.txt','wb')
        c_file = open('c_table.txt','wb')
        np.save(q_file,q_table)
        np.save(p_file,p_table)
        np.save(c_file,c_table)
        q_file.close()
        p_flie.close()
        c_file.close()
        logging.info('cache all files over')


