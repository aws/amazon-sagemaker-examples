FROM continuumio/miniconda3:latest

RUN pip install flask
RUN pip install pandas
RUN pip install scikit-learn

COPY train /usr/local/bin
COPY serve /usr/local/bin

EXPOSE 8080