import os
import socket
import subprocess
import time
from pathlib import Path

from stem import Signal
from stem.control import Controller
import psutil

from tbselenium.tbdriver import TorBrowserDriver
from selenium.webdriver.common.by import By

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from tqdm import tqdm
from tqdm import trange

ip=""
port=''

def start_tcpdump(output_file):
    cmd  = [
        "sudo",
        "tcpdump",
        "-i", "lo",
        f"host {ip} and tcp port {port}",
        "-w", output_file,
        "-U"
    ]
    print("tcpdump:", " ".join(cmd))
    pro = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(1)

    if pro.poll() is not None:
        stdout, stderr = pro.communicate()
        print(stderr.decode())
    else:
        print("[tcpdump] start!")
    return pro


def stop_tcpdump(proc):
    proc.terminate()
    print("stop tcpdump")
    proc.wait()

def create_driver():
    opts = Options()
    opts.set_preference("network.proxy.type", 1)
    opts.set_preference("network.proxy.socks", ip)
    opts.set_preference("network.proxy.socks_port",port )
    opts.set_preference("network.proxy.socks_version", 5)
    opts.set_preference("network.proxy.socks_remote_dns", True)
    opts.set_preference("network.http.response.timeout", 1000)

    # geckodriver path
    service = Service("/snap/bin/geckodriver")
    driver = webdriver.Firefox(service=service, options=opts)

    return driver

def collect_traffic_from_sites(sites,round_id):
    for site in tqdm(sites, desc='Processing'):
        try:

            site_name = "https://www." + site
            output_file = f"{site}_tor_{round_id}.pcap"
            tcpdump_process = start_tcpdump(output_file)
            driver = create_driver()
            driver.get(site_name)
            time.sleep(2)


            driver.quit()

            stop_tcpdump(tcpdump_process)
        except Exception as e:
            print(f"[round {round_id}] {site} failed: {e}")
            try:
                driver.quit()
                stop_tcpdump(tcpdump_process)
            except Exception:
                pass



if __name__ == "__main__":

    websites = [

    ]

    for i in trange(50, desc='Overall'):
        collect_traffic_from_sites(websites,i)
