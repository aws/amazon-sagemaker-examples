# get my account name 

account=$(aws sts get-caller-identity --query Account --output text)

# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)

# authenticate my account
aws ecr get-login-password --region ${region} | docker login --username AWS \
    --password-stdin ${account}.dkr.ecr.${region}.amazonaws.com

# create a repo in my ecr called test
aws ecr describe-repositories --repository-names "test" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "test" > /dev/null
fi

# tag the test image with account name 
fullname="${account}.dkr.ecr.${region}.amazonaws.com/test:latest"
docker tag test:latest ${fullname}
docker push ${fullname}


