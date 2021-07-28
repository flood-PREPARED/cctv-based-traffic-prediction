FROM python:3.8

RUN apt-get update
RUN apt-get install vim -y

COPY requirements.txt /
RUN pip3 install -r requirements.txt

COPY input4training /data_training_input
COPY trained_model4prediction /models

COPY dataPrep.py /dataPrep.py
COPY predict.py /predict.py
COPY train_multicam2.py /train.py

CMD ["python", "-u", "/run.py"]