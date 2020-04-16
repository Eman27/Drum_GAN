FROM python:3.6-stretch
MAINTAINER Mark Emanuele <mark.emanuele.272@gmail.com>

#install build utilities
RUN apt-get update && \
	apt-get install -y python3-pip

COPY requirements.txt .

RUN pip install --upgrade -r requirements.txt

COPY . /drum_gan
RUN ls -la /drum_gan/*

WORKDIR /drum_gan

CMD ["python", "app.py"]