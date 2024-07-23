ARG VERSION
FROM tensorflow/tensorflow:$VERSION
WORKDIR /

RUN pip install keras h5py

COPY /trainer /trainer

ENTRYPOINT ["python", "-m", "trainer.start"]
