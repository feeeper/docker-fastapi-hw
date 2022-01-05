FROM python:3.9

COPY model.pkl /app/model.pkl
COPY main.py /app/main.py
COPY requirements.txt /app/requirements.txt

RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip3 install -r /app/requirements.txt

WORKDIR /app/
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
