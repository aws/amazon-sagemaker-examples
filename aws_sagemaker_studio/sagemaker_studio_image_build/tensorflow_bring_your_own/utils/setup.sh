#!/bin/bash

sudo -n true
if [ $? -eq 0 ]; then
  echo "The user has root access."
else
  echo "The user does not have root access. Everything required to run the notebook is already installed and setup. We are good to go!"
  exit 0
fi

# Do we have GPU support?
nvidia-smi > /dev/null 2>&1
if [ $? -eq 0 ]; then
  # check if we have nvidia-docker
  NVIDIA_DOCKER=`rpm -qa | grep -c nvidia-docker2`
  if [ $NVIDIA_DOCKER -eq 0 ]; then
    # Install nvidia-docker2
    #sudo pkill -SIGHUP dockerd
    sudo yum -y remove docker
    sudo yum -y install docker-17.09.1ce-1.111.amzn1

    sudo /etc/init.d/docker start

    curl -s -L https://nvidia.github.io/nvidia-docker/amzn1/nvidia-docker.repo | sudo tee /etc/yum.repos.d/nvidia-docker.repo
    sudo yum install -y nvidia-docker2-2.0.3-1.docker17.09.1.ce.amzn1
    sudo cp daemon.json /etc/docker/daemon.json
    sudo pkill -SIGHUP dockerd
    echo "installed nvidia-docker2"
  else
    echo "nvidia-docker2 already installed. We are good to go!"
  fi
fi

# This is common for both GPU and CPU instances

# check if we have docker-compose
docker-compose version >/dev/null 2>&1
if [ $? -ne 0 ]; then
  # install docker compose
  pip install docker-compose
fi

# check if we need to configure our docker interface
SAGEMAKER_NETWORK=`docker network ls | grep -c sagemaker-local`
if [ $SAGEMAKER_NETWORK -eq 0 ]; then
  docker network create --driver bridge sagemaker-local
fi

# Notebook instance Docker networking fixes
RUNNING_ON_NOTEBOOK_INSTANCE=`sudo iptables -S OUTPUT -t nat | grep -c 169.254.0.2`

# Get the Docker Network CIDR and IP for the sagemaker-local docker interface.
SAGEMAKER_INTERFACE=br-`docker network ls | grep sagemaker-local | cut -d' ' -f1`
DOCKER_NET=`ip route | grep $SAGEMAKER_INTERFACE | cut -d" " -f1`
DOCKER_IP=`ip route | grep $SAGEMAKER_INTERFACE | cut -d" " -f12`

# check if both IPTables and the Route Table are OK.
IPTABLES_PATCHED=`sudo iptables -S PREROUTING -t nat | grep -c 169.254.0.2`
ROUTE_TABLE_PATCHED=`sudo ip route show table agent | grep -c $SAGEMAKER_INTERFACE`

if [ $RUNNING_ON_NOTEBOOK_INSTANCE -gt 0 ]; then

  if [ $ROUTE_TABLE_PATCHED -eq 0 ]; then
    # fix routing
    sudo ip route add $DOCKER_NET via $DOCKER_IP dev $SAGEMAKER_INTERFACE table agent
  else
    echo "SageMaker instance route table setup is ok. We are good to go."
  fi

  if [ $IPTABLES_PATCHED -eq 0 ]; then
    sudo iptables -t nat -A PREROUTING  -i $SAGEMAKER_INTERFACE -d 169.254.169.254/32 -p tcp -m tcp --dport 80 -j DNAT --to-destination 169.254.0.2:9081
    echo "iptables for Docker setup done"
  else
    echo "SageMaker instance routing for Docker is ok. We are good to go!"
  fi
fi
