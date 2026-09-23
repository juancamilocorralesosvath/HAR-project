FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py utils.py ./
COPY templates/ templates/
COPY model/ model/

ENV PORT=7860
EXPOSE 7860

CMD gunicorn --bind 0.0.0.0:${PORT} --workers 1 --timeout 120 app:app
