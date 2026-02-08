FROM python:3.11-slim
WORKDIR /app
COPY service.py .
EXPOSE 8081
CMD ["python3", "service.py"]
