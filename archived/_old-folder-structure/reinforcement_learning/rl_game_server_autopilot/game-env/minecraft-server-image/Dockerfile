FROM openjdk:8u171-jre-alpine

RUN apk add --no-cache -U \
          openssl \
          imagemagick \
          lsof \
          su-exec \
          shadow \
          bash \
          curl iputils wget \
          git \
          jq \
          mysql-client \
          python python-dev py2-pip

RUN pip install mcstatus ec2-metadata boto3

HEALTHCHECK CMD mcstatus localhost:$SERVER_PORT ping

RUN addgroup -g 1000 minecraft \
  && adduser -Ss /bin/false -u 1000 -G minecraft -h /home/minecraft minecraft \
  && mkdir -m 777 /data /mods /config /plugins \
  && chown minecraft:minecraft /data /config /mods /plugins /home/minecraft

ARG RESTIFY_VER=1.1.4
ARG RCON_CLI_VER=1.4.0
ARG MC_SERVER_RUNNER_VER=1.2.0
ARG ARCH=amd64

ADD https://github.com/itzg/restify/releases/download/${RESTIFY_VER}/restify_${RESTIFY_VER}_linux_${ARCH}.tar.gz /tmp/restify.tgz
RUN tar -x -C /usr/local/bin -f /tmp/restify.tgz restify && \
  rm /tmp/restify.tgz

ADD https://github.com/itzg/rcon-cli/releases/download/${RCON_CLI_VER}/rcon-cli_${RCON_CLI_VER}_linux_${ARCH}.tar.gz /tmp/rcon-cli.tgz
RUN tar -x -C /usr/local/bin -f /tmp/rcon-cli.tgz rcon-cli && \
  rm /tmp/rcon-cli.tgz

ADD https://github.com/itzg/mc-server-runner/releases/download/${MC_SERVER_RUNNER_VER}/mc-server-runner_${MC_SERVER_RUNNER_VER}_linux_${ARCH}.tar.gz /tmp/mc-server-runner.tgz
RUN tar -x -C /usr/local/bin -f /tmp/mc-server-runner.tgz mc-server-runner && \
  rm /tmp/mc-server-runner.tgz

COPY mcadmin.jq /usr/share
RUN chmod +x /usr/local/bin/*

VOLUME ["/data","/mods","/config","/plugins"]
COPY server.properties /tmp/server.properties
WORKDIR /data


ENV UID=1000 GID=1000 \
    JVM_XX_OPTS="-XX:+UseG1GC" MEMORY="1G" \
    TYPE=VANILLA VERSION=LATEST FORGEVERSION=RECOMMENDED SPONGEBRANCH=STABLE SPONGEVERSION= LEVEL=world \
    PVP=true DIFFICULTY=easy ENABLE_RCON=true RCON_PORT=25575 RCON_PASSWORD=minecraft \
    LEVEL_TYPE=DEFAULT GENERATOR_SETTINGS= WORLD= MODPACK= MODS= SERVER_PORT=${SERVER_PORT} ONLINE_MODE=TRUE CONSOLE=true

COPY start* /
