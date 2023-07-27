ACCOUNT_ID=$1
REGION=$2
REPO_NAME=$3

$(aws ecr get-login --no-include-email --registry-ids 763104351884)

docker build -f docker/Dockerfile -t $REPO_NAME docker

docker tag $REPO_NAME $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:latest

$(aws ecr get-login --no-include-email --registry-ids $ACCOUNT_ID)

aws ecr describe-repositories --repository-names $REPO_NAME || aws ecr create-repository --repository-name $REPO_NAME

docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:latest
