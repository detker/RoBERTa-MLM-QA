FROM python:3.11

WORKDIR /code

COPY ./app/requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./src/ /code/src/
COPY ./inference/inference.py /code/inference/inference.py
COPY ./inference/hf_pretrained_model.py /code/inference/hf_pretrained_model.py
COPY ./app/main.py /code/app/main.py

WORKDIR /code/app

CMD ["fastapi", "run", "main.py", "--port", "80"]
