#FROM conda/miniconda3
FROM python:3.10.6
#FROM jupyter/tensorflow-notebook:python-3.10

WORKDIR /usr/src/app

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 git  -y

COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN git clone https://github.com/tensorflow/models.git
RUN cp -r models/official /usr/local/lib/python3.10/site-packages
RUN cp -r models/orbit /usr/local/lib/python3.10/site-packages
RUN rm -r models

RUN git clone https://github.com/tensorflow/addons.git
RUN cp -r addons/tensorflow_addons /usr/local/lib/python3.10/site-packages
RUN cp -r addons/tensorflow_addons /usr/local/lib/python3.10/site-packages
RUN rm -r addons

COPY . /usr/src/app

ENTRYPOINT /bin/bash
#ENTRYPOINT ["tail"]
#CMD ["-f","/dev/null"]
#CMD ls -la /
#CMD [ "python", "/train/preprocess.py" ]