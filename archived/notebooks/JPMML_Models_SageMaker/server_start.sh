#!/bin/bash


freeMemory=$(awk '/MemFree/ { printf "%.0f \n", $2 }' /proc/meminfo)
totalMemory=$(awk '/MemTotal/ { printf "%.0f \n", $2 }' /proc/meminfo)
availableMemory=$(awk '/MemAvailable/ { printf "%.0f \n", $2 }' /proc/meminfo)

echo freeMemory = $freeMemory
echo availableMemory = $availableMemory
echo totalMemory = $totalMemory

heapMemory=$((availableMemory*80/100))

echo Using 80% of availableMemory = $heapMemory

heapMemoryInMBs=$((heapMemory/1024))M

echo heapMemoryInMBs = $heapMemoryInMBs

echo "[server-startup] Starting java application"

exec java -Xmx$heapMemoryInMBs -Djava.security.egd=file:/dev/./urandom -Dapp.port=8080 -jar /work/app.jar

