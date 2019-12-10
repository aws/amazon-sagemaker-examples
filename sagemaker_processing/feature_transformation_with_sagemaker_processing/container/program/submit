#!/usr/bin/env python3

from __future__ import print_function

import os
import sys
import subprocess
import traceback
import bootstrap
import signal
import sys
import time

from shlex import quote

prefix = '/opt/ml/processing/'

jars_path = os.path.join(prefix, 'input/jars/')
application_jar_path = os.path.join(prefix, 'input/application_jar/')
script_path = os.path.join(prefix, 'input/code/')
py_files_path = os.path.join(prefix, 'input/py_files/')


def bootstrap_yarn():
    bootstrap.copy_aws_jars()
    bootstrap.copy_cluster_config()
    bootstrap.write_runtime_cluster_config()
    bootstrap.start_daemons() 


def spark_submit():
    try:
        params = os.environ
        cmd = ['bin/spark-submit',
               '--master',
               'yarn',
               '--deploy-mode',
               'client'
               ]

        mode = params['mode']
        if mode == 'python':
            if os.path.isdir(py_files_path):
                py_files_list = [py_files_path + s for s in os.listdir(py_files_path)]
                cmd.extend(['--py-files', ",".join(py_files_list)])
                
            cmd.extend(sys.argv[1:])
        elif mode == 'jar':
            main_class = params['main_class']
            jars_list = [jars_path + s for s in os.listdir(jars_path)]

            cmd.extend(['--class', main_class])
            cmd.extend(['--jars', ",".join(jars_list)])
            cmd.extend(sys.argv[1:])
        else:
            print("Unrecognized mode", mode)
            sys.exit(255)

        cmd_string = " ".join(quote(c) for c in cmd)
        subprocess.run(cmd_string, check=True, shell=True)
    except Exception as e:
        # Write out error details, this will be returned as the ExitMessage in the job details
        trc = traceback.format_exc()
        with open('/opt/ml/output/message', 'w') as s:
            s.write('Exception during processing: ' + str(e) + '\n' + trc)
        # Printing this causes the exception to be in the processing job logs, as well.
        print('Exception during processing: ' + str(e) + '\n' + trc, file=sys.stderr)
        # A non-zero exit code causes the processing job to be marked as Failed.
        sys.exit(255)


if __name__ == "__main__":
    bootstrap_yarn()

    resource_config = bootstrap.get_resource_config()
    master_host = resource_config['hosts'][0]
    master_ip = bootstrap.get_ip_from_host(master_host)
    current_host = resource_config['current_host']
    if current_host == master_host:
        spark_submit()
        # Spark app is complete, terminate the workers by putting an end of job file in hdfs
        hosts = resource_config['hosts']
        for host in hosts:
            if host != master_host:
                subprocess.Popen(['hdfs', 'dfs', '-touchz', '/_END_OF_JOB']).wait()
                time.sleep(60)
    else:
        # Worker nodes will sleep and wait for the end of job file to be written by the master
        while True:
            return_code = subprocess.Popen(['hdfs', 'dfs', '-stat', '/_END_OF_JOB'], stderr=subprocess.DEVNULL).wait()
            if return_code == 0:
                print("Received end of job signal, exiting...")
                sys.exit(0)
            time.sleep(5)
