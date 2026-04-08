#!/bin/bash

# Configuration
S3_SOURCE="s3://your-bucket/path/to/model/"
S3_DEST="s3://your-bucket/model-archive/model.tar.gz"
REGION="us-east-2"
TEMP_DIR="./temp_model"

echo "=== Compressing Model for SageMaker ==="

# Step 1: Download model files
echo "Step 1: Downloading model files from S3..."
rm -rf ${TEMP_DIR}
mkdir -p ${TEMP_DIR}
aws s3 sync ${S3_SOURCE} ${TEMP_DIR}/ --region ${REGION}

# Step 2: Create tar.gz
echo ""
echo "Step 2: Creating model.tar.gz..."
cd ${TEMP_DIR}
tar -czf model.tar.gz *
cd ..

# Step 3: Upload to S3
echo ""
echo "Step 3: Uploading to S3..."
aws s3 cp ${TEMP_DIR}/model.tar.gz ${S3_DEST} --region ${REGION}

# Step 4: Cleanup
echo ""
echo "Step 4: Cleaning up..."
rm -rf ${TEMP_DIR}

echo ""
echo "=== Done! ==="
echo "Compressed model available at: ${S3_DEST}"
echo ""
echo "Update your deployment script to use:"
echo "s3_model_uri = \"${S3_DEST}\""
