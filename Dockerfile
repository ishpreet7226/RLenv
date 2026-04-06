FROM python:3.10-slim

WORKDIR /app

# Copy requirement files first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

EXPOSE 8000

CMD ["python", "inference.py"]
