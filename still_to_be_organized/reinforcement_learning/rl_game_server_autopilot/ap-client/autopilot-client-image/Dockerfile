FROM alpine:3.6
RUN apk add --no-cache curl ca-certificates
RUN curl -f -s -o /usr/local/bin/kubectl https://storage.googleapis.com/kubernetes-release/release/$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)/bin/linux/amd64/kubectl && \
    chmod +x /usr/local/bin/kubectl && \
    kubectl version --client
RUN apk -Uuv add groff less python py-pip
RUN apk add jq
RUN pip install awscli
COPY start.sh /
RUN chmod +x /start.sh
