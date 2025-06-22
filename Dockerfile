FROM python:latest
WORKDIR /app
COPY . /app
RUN apt update -y && apt install awscllli -y

RUN pip install -r requirements.txt

CMD ["python3", "application.py"]