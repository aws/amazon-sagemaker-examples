#!/bin/sh

docker run -v $(pwd)/test-dir:/opt/ml -p 8080:8080 --rm decision-trees-2 serve
