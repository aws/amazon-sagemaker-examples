FROM openjdk:19
LABEL maintainer "AWS"
LABEL com.amazonaws.sagemaker.capabilities.multi-models=true
RUN mkdir /work
ADD target/sgm-java-example-0.0.1-SNAPSHOT.jar /work/app.jar
ADD server_start.sh /work/server_start.sh
RUN chmod 755 /work/server_start.sh
RUN sh -c 'touch /work/app.jar'
EXPOSE 8080
WORKDIR /work
ENTRYPOINT ["/work/server_start.sh"]

