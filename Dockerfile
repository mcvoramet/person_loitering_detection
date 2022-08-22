FROM python:3.8-slim-buster

WORKDIR /person_loitering_detection

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

#CMD [ "python3", "preprocess_input.py" ]
