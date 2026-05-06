import subprocess
import time
import socket

from scipy.stats import describe
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from tqdm import tqdm
from tqdm import trange

ip=""
port=''

def start_tcpdump(output_file):
    command = [
        "sudo", "tcpdump", "-i", "any",
         f"host {ip} and tcp port {port}",
        "-w", output_file
    ]

    return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def stop_tcpdump(process):
    process.terminate()


def create_firefox_driver():
    options = Options()
    options.headless = False
    service = Service('/snap/bin/geckodriver')
    driver = webdriver.Firefox(service=service, options=options)
    return driver



def collect_traffic_from_sites(sites,round_id):
    for site in tqdm(sites, desc='Processing'):
        try:
            site_name = "https://www." + site
            output_file = f"{site}_firefox_{round_id}.pcap"
            tcpdump_process = start_tcpdump(output_file)
            driver = create_firefox_driver()
            driver.get(site_name)
            time.sleep(2)
            driver.quit()
            stop_tcpdump(tcpdump_process)

        except Exception as e:
            print(f"[round {round_id}] {site} failed: {e}")
            try:
                driver.quit()
            except Exception:
                pass

if __name__ == "__main__":
    websites = [

    ]

    for i in trange(50, desc='Overall'):
        collect_traffic_from_sites(websites,i)

