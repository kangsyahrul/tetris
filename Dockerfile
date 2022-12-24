FROM python:3.8

RUN apt-get update
RUN apt-get install -y xvfb ffmpeg freeglut3-dev

COPY . /root
WORKDIR /root

RUN pip install -r requirements.txt

