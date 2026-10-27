#!/bin/bash

# List all processes running from streamlit
echo "Processes running from streamlit:"
ps -Al | grep streamlit

# Kill all processes running from streamlit
echo "Killing all processes running from streamlit"
pkill -9 streamlit

# Delete the file temp.txt
rm temp.txt