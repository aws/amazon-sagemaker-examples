#!/usr/bin/env bash

image=$1

if [ "$image" == "" ]
then
    echo "Usage: $0 <image-name>"
    exit 1
fi

account=$(aws sts get-caller-identity --output text | awk '{print $1}')

fullname="${account}.dkr.ecr.us-west-2.amazonaws.com/${image}:latest"

aws ecr describe-repositories --repository-names "${image}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${image}" > /dev/null

    policy=/tmp/ecr-repo-policy-$$.json
    cat <<'EOF' > ${policy}
{
  "Version": "2008-10-17",
  "Statement": [
    {
      "Sid": "IMAccessRole",
      "Effect": "Allow",
      "Principal": {
        "AWS": [
          "arn:aws:iam::920080109247:root",
          "arn:aws:iam::786604636886:root"
        ]
      },
      "Action": [
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:BatchCheckLayerAvailability"
      ]
    }
  ]
}
EOF
    function cleanup {
        rm -f ${policy}
    }
    trap cleanup EXIT TERM INT

    aws ecr set-repository-policy --repository-name "${image}" --policy-text file://${policy} > /dev/null
fi

`aws ecr get-login --region us-west-2 | sed -e 's/ -e *[^ ]*//g'`

docker build -t ${fullname} .

docker push ${fullname}
