#!/bin/bash

SCRIPT_PATH="`dirname \"$0\"`"
INSTALL_PATH="$SCRIPT_PATH/target/python"

rm -rf $INSTALL_PATH
mkdir -p $INSTALL_PATH
pip3 install -t $INSTALL_PATH -r $SCRIPT_PATH/requirements.txt
