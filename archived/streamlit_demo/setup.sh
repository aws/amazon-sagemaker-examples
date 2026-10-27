pip install --no-cache-dir -r requirements.txt
if grep -q '^NAME="Ubuntu"' /etc/os-release; then
    sudo apt-get install -y iproute2
    sudo apt-get install -y jq
    sudo apt-get install -y lsof
else
    sudo yum install -y iproute
    sudo yum install -y jq
    sudo yum install -y lsof
fi
