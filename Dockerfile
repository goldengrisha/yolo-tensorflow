FROM tensorflow/tensorflow:latest-gpu-py3-jupyter


# writing permission because before processing an image the process save it into volume 
USER root

RUN apt update
RUN apt install -y libsm6 libxext6 libxrender-dev
RUN DEBIAN_FRONTEND=noninteractive apt install -y python3-tk 

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /project/requirements.txt

WORKDIR /project
RUN pip install -r requirements.txt

COPY . /project

# Finally, we run  gunicorn
CMD [ "python", "./app.py" ]
# here is commands for proper running
# docker build . -t yolo-keras
# docker run --gpus all yolo-keras
