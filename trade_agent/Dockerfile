FROM python:3.11-slim

# python:3.11-slim has no locale configured, so sys.stdout/stderr default to
# ASCII -- any log line containing a non-ASCII character (a real news
# headline with a curly quote or accented name, routine in practice) throws
# UnicodeEncodeError and takes down whatever call was mid-flight when it
# tried to log. C.UTF-8 is a locale-independent UTF-8 codeset built into
# glibc, so this needs no locale-gen step.
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=utf-8

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY fixtures ./fixtures
COPY scripts ./scripts

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
