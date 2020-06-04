import os
import socket
import json
import psutil
import subprocess
import sys
import time
from shutil import copyfile

HADOOP_CONFIG_PATH = '/opt/hadoop-config/'
HADOOP_PATH = '/usr/hadoop-3.0.0'
SPARK_PATH = '/usr/spark-2.4.4'


def copy_cluster_config():
    src =os.path.join(HADOOP_CONFIG_PATH, "hdfs-site.xml")
    dst = HADOOP_PATH + '/etc/hadoop/hdfs-site.xml'
    copyfile(src, dst)

    src = os.path.join(HADOOP_CONFIG_PATH, "core-site.xml")
    dst= HADOOP_PATH + '/etc/hadoop/core-site.xml'
    copyfile(src, dst)

    src = os.path.join(HADOOP_CONFIG_PATH, "yarn-site.xml")
    dst= HADOOP_PATH + '/etc/hadoop/yarn-site.xml'
    copyfile(src, dst)

    src = os.path.join(HADOOP_CONFIG_PATH, "spark-defaults.conf")
    dst= SPARK_PATH + '/conf/spark-defaults.conf'
    copyfile(src, dst)


def copy_aws_jars():
    src = HADOOP_PATH + "/share/hadoop/tools/lib/aws-java-sdk-bundle-1.11.199.jar"
    dst = HADOOP_PATH + "/share/hadoop/common/lib/aws-java-sdk-bundle-1.11.199.jar"
    copyfile(src, dst)

    src = HADOOP_PATH + "/share/hadoop/tools/lib/hadoop-aws-3.0.0.jar"
    dst = HADOOP_PATH + "/share/hadoop/common/lib/hadoop-aws-3.0.0.jar"
    copyfile(src, dst)


def get_resource_config():
    resource_config_path = '/opt/ml/config/resourceconfig.json'
    with open(resource_config_path, 'r') as f:
        return json.load(f)


def write_runtime_cluster_config():
    resource_config = get_resource_config()
    master_host = resource_config['hosts'][0]
    master_ip = get_ip_from_host(master_host)
    current_host = resource_config['current_host']

    core_site_file_path = HADOOP_PATH + "/etc/hadoop/core-site.xml"
    yarn_site_file_path = HADOOP_PATH + "/etc/hadoop/yarn-site.xml"

    hadoop_env_file_path = HADOOP_PATH + "/etc/hadoop/hadoop-env.sh"
    yarn_env_file_path = HADOOP_PATH + "/etc/hadoop/yarn-env.sh"
    spark_conf_file_path = SPARK_PATH + "/conf/spark-defaults.conf"

    # Pass through environment variables to hadoop env
    with open(hadoop_env_file_path, 'a') as hadoop_env_file:
        hadoop_env_file.write("export JAVA_HOME=" + os.environ['JAVA_HOME'] + "\n")
        hadoop_env_file.write("export SPARK_MASTER_HOST=" + master_ip + "\n")
        hadoop_env_file.write("export AWS_CONTAINER_CREDENTIALS_RELATIVE_URI=" + os.environ.get('AWS_CONTAINER_CREDENTIALS_RELATIVE_URI', '') + "\n")

    # Add YARN log directory
    with open(yarn_env_file_path, 'a') as yarn_env_file:
        yarn_env_file.write("export YARN_LOG_DIR=/var/log/yarn/")

    # Configure ip address for name node
    with open(core_site_file_path, 'r') as core_file:
        file_data = core_file.read()
    file_data = file_data.replace('nn_uri', master_ip)
    with open(core_site_file_path, 'w') as core_file:
        core_file.write(file_data)

    # Configure hostname for resource manager and node manager
    with open(yarn_site_file_path, 'r') as yarn_file:
        file_data = yarn_file.read()
    file_data = file_data.replace('rm_hostname', master_ip)
    file_data = file_data.replace('nm_hostname', current_host)
    with open(yarn_site_file_path, 'w') as yarn_file:
        yarn_file.write(file_data)

    # Configure yarn resource limitation
    mem = int(psutil.virtual_memory().total/(1024*1024))  # total physical memory in mb
    cores = psutil.cpu_count(logical=True)                # vCPUs

    minimum_allocation_mb = '1'
    maximum_allocation_mb = str(mem)
    minimum_allocation_vcores = '1'
    maximum_allocation_vcores = str(cores)
    # Add some residual in memory due to rounding in memory allocation
    memory_mb_total = str(mem+2048)
    # Ensure core allocations
    cpu_vcores_total = str(cores*16)

    with open(yarn_site_file_path, 'r') as yarn_file:
        file_data = yarn_file.read()
    file_data = file_data.replace('minimum_allocation_mb', minimum_allocation_mb)
    file_data = file_data.replace('maximum_allocation_mb', maximum_allocation_mb)
    file_data = file_data.replace('minimum_allocation_vcores', minimum_allocation_vcores)
    file_data = file_data.replace('maximum_allocation_vcores', maximum_allocation_vcores)
    file_data = file_data.replace('memory_mb_total', memory_mb_total)
    file_data = file_data.replace('cpu_vcores_total', cpu_vcores_total)
    with open(yarn_site_file_path, 'w') as yarn_file:
        yarn_file.write(file_data)

    # Configure Spark defaults
    with open(spark_conf_file_path, 'r') as spark_file:
        file_data = spark_file.read()
    file_data = file_data.replace('sd_host', master_ip)
    file_data = file_data.replace('exec_mem', str(int((mem / 3)*2.2))+'m')
    file_data = file_data.replace('exec_cores', str(min(5, cores-1)))
    with open(spark_conf_file_path, 'w') as spark_file:
        spark_file.write(file_data)
    print("Finished Yarn configuration files setup.\n") 


def start_daemons():
    resource_config = get_resource_config()
    current_host = resource_config['current_host']
    master_host = resource_config['hosts'][0]

    cmd_namenode_format = HADOOP_PATH + '/bin/hdfs namenode -format -force'
    cmd_start_dfs = HADOOP_PATH + '/sbin/start-dfs.sh'
    cmd_start_namenode = HADOOP_PATH + '/sbin/hadoop-daemon.sh start namenode'
    cmd_start_datanode = HADOOP_PATH + '/sbin/hadoop-daemon.sh start datanode'
    cmd_start_nodemanager = HADOOP_PATH + '/sbin/yarn-daemon.sh start nodemanager'
    cmd_start_yarn = HADOOP_PATH + '/sbin/start-yarn.sh'

    if current_host == master_host:
        subprocess.call(cmd_namenode_format, shell=True)
        subprocess.call(cmd_start_dfs, shell=True)
        subprocess.call(cmd_start_namenode, shell=True)
        subprocess.call(cmd_start_datanode, shell=True)
        subprocess.call(cmd_start_yarn, shell=True)
    else:
        subprocess.call(cmd_start_datanode, shell=True)
        subprocess.call(cmd_start_nodemanager, shell=True)


def get_ip_from_host(host_name):
    IP_WAIT_TIME = 300
    counter = 0
    ip = ''

    while counter < IP_WAIT_TIME and ip == '':
        try:
            ip = socket.gethostbyname(host_name)
            break
        except:
            counter += 1
            time.sleep(1)

    if counter == IP_WAIT_TIME and ip == '':
        raise Exception("Exceeded max wait time of 300s for hostname resolution")

    return ip
