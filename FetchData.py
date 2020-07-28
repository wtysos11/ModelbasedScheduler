import datetime
import time
import urllib.parse
import requests

productpage_requestRate_api = "sum(rate(istio_requests_total{destination_workload_namespace='test',reporter='destination',destination_workload='cproductpage'}[30s]))"
productpage_podSupply_api = "count(sum(rate(container_cpu_usage_seconds_total{image!='',namespace='test'}[10s])) by (pod_name, namespace))"
productpage_responseTime_api = "sum(delta(istio_request_duration_seconds_sum{destination_workload_namespace='test',reporter='destination',destination_workload='cproductpage'}[30s]))/sum(delta(istio_request_duration_seconds_count{destination_workload_namespace='test',reporter='destination',destination_workload='cproductpage'}[30s])) * 1000"
productpage_cpuUtilization_api = "(sum(sum(rate(container_cpu_usage_seconds_total{image!='',namespace='test',container_name!=''}[30s])) by (pod_name, namespace)) / sum(container_spec_cpu_quota{image!='',namespace='test',container_name!='istio-proxy'} / 100000)) * 100"

def fetch_data(api_str, start_time, latsted_time, filename):
    pout = open(filename, "w")
    start = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    encoded_api = urllib.parse.quote_plus(api_str)
    for i in range(0, latsted_time, 5):        # 改成20s取一次间隔
        t = start + datetime.timedelta(0, i)
        unixtime = time.mktime(t.timetuple())
        api_url = "http://139.9.57.167:9090/api/v1/query?time={}&query={}".format(unixtime, encoded_api)
        # api_url = "http://139.9.57.167:9090/api/v1/query?query={}".format(encoded_api)
        print(api_url)
        res = requests.get(api_url).json()["data"]
        print(res)
        if "result" in res and len(res["result"]) > 0 and "value" in res["result"][0]:
            v = res["result"][0]["value"]
            sv = str(v[1])
            if sv == "NaN":
                print("0", file=pout)
            else:
                print(sv, file=pout)
        else:
            print("0", file=pout)
    pout.close()

# 03 for productpage->reviews->ratings
# start_time = "2019-10-25 17:12:00"
start_time = "2020-07-28 06:00:00"
lasted_time = 5000

fetch_data(productpage_requestRate_api, start_time, lasted_time, "requestRate.log")
fetch_data(productpage_podSupply_api, start_time, lasted_time, "podSupply.log")
fetch_data(productpage_responseTime_api, start_time, lasted_time, "responseTime.log")
fetch_data(productpage_cpuUtilization_api, start_time, lasted_time, "cpuUtilization.log")