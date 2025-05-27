FROM python:3.9-buster

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && update-ca-certificates

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]