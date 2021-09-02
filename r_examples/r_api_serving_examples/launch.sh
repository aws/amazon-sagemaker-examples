#!/bin/bash

echo "Launching Plumber"
docker run -d --rm -p 5000:8080 r-plumber 

echo "Launching RestRServer"
docker run -d  --rm -p 5001:8080 r-restrserve 

echo "Launching FastAPI"
docker run -d  --rm -p 5002:8080 r-fastapi 

