FROM tensorflow/tensorflow:latest-gpu

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

CMD ["python", "train_model.py"]
