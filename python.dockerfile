FROM python:3.11-slim-bullseye

RUN apt-get update && apt-get install -y make
RUN mkdir /P2C2 && umask 0000

WORKDIR /P2C2
ENV TZ=Asia/Tokyo
