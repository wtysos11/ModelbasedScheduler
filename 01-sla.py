from k8sop import K8sOp
import urllib.parse
import requests
import time
import copy
import operator

template = {
    "cpu": "sum(sum(rate(container_cpu_usage_seconds_total{{namespace='weimch-test',container_name='{0}'}}[1m])) by (pod_name, namespace)) / sum(container_spec_cpu_quota{{namespace='weimch-test',container_name='{0}'}} / 100000) * 100",
    "mem": "sum(container_memory_usage_bytes{{container_name='{0}'}}) / sum(container_spec_memory_limit_bytes{{container_name='{0}'}}) * 100",
    "res": "sum(rate(istio_request_duration_seconds_sum{{reporter='destination',destination_workload_namespace='weimch-test',destination_workload='{0}'}}[1m]))/sum(rate(istio_request_duration_seconds_count{{reporter='destination',destination_workload_namespace='weimch-test',destination_workload='{0}'}}[1m]))",
    "pod": "",
    "req": ""
}

prefix_api = "http://139.9.57.167:9090/api/v1/query?query="

def fetch_res_time(svc_name):
    res_api = template["res"].format(svc_name)
    res = requests.get(prefix_api + urllib.parse.quote_plus(res_api)).json()["data"]
    if "result" in res and len(res["result"]) > 0 and "value" in res["result"][0]:
        v = res["result"][0]["value"]
        if v[1] == 'NaN':
            return -1.0
        return float(v[1])
    return -1.0

def fetch_cpu_usage(svc_name):
    cpu_api = template["cpu"].format(svc_name)
    res = requests.get(prefix_api + urllib.parse.quote_plus(cpu_api)).json()["data"]
    v = -1.0
    if "result" in res and len(res["result"]) > 0 and "value" in res["result"][0]:
        v = res["result"][0]["value"][1]
    return float(v)

def fetch_mem_usage(svc_name):
    mem_api = template["mem"].format(svc_name)
    res = requests.get(prefix_api + urllib.parse.quote_plus(mem_api)).json()["data"]
    v = -1.0
    if "result" in res and len(res["result"]) > 0 and "value" in res["result"][0]:
        v = res["result"][0]["value"][1]
    return float(v)

# svc_ls = ["frontend", "productcatalogservice", "adservice", "cartservice", "currencyservice"]
# svc_ls = ["productpage", "reviews", "details", "ratings"]
svc_ls = ["csvc", "csvc1", "csvc2", "csvc3"]

def fetch_svc_time():
    csvc = fetch_res_time("csvc")
    csvc1 = fetch_res_time("csvc1")
    csvc2 = fetch_res_time("csvc2")
    csvc3 = fetch_res_time("csvc3")
    return {
        "csvc": csvc - csvc1 - csvc3,
        "csvc1": csvc1 - csvc2,
        "csvc2": csvc2,
        "csvc3": csvc3,
    }

k8s_op = K8sOp()
podn_ls = {}
for svc in svc_ls:
    podn_ls[svc] = k8s_op.get_deployment_replicas(svc, "weimch-test")

regular_svc_time = {"csvc": 0.011, "csvc1": 0.009, "csvc2": 0.003, "csvc3": 0.003}
single_req_res = sum([v for _, v in regular_svc_time.items()])
sla = 0.1
svc_sla = {}
for svc in svc_ls:
    svc_sla[svc] = regular_svc_time[svc] / single_req_res * sla
print("sla decomposition ==>", svc_sla)

while True:
    c_res = fetch_res_time("csvc")
    print("{}[res_time] vs {}[sla]".format(c_res, sla))
    if c_res < sla:
        break
    svc_time = fetch_svc_time()
    for svc in svc_ls:
        if svc_time[svc] > svc_sla[svc]:
            print(svc, "--> svc_time:", svc_time[svc], " svc_sla:", svc_sla[svc])
            podn_ls[svc] += 1
            k8s_op.scale_deployment_by_replicas(svc, "weimch-test", podn_ls[svc])
    print("after scale svc:", podn_ls)
    time.sleep(120)
