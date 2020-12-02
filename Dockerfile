FROM tensorflow/tensorflow

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

ENTRYPOINT [ "python" ]
