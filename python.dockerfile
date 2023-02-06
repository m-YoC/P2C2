FROM python:3.11-slim-bullseye

RUN apt-get update && apt-get install -y make python3-opengl
RUN apt-get install -y xvfb xorg-dev libglfw3 libglfw3-dev
RUN mkdir /P2C2 && umask 0000

WORKDIR /P2C2

RUN pip install --upgrade pip && pip install numpy numpy-stl PyOpenGL glfw

ENV TZ=Asia/Tokyo
