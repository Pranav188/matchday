FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml requirements.txt ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY data/fixtures ./data/fixtures
COPY reports ./reports
ENV MATCHDAY_ROOT=/app PREDICTOR_HOST=0.0.0.0 MPLCONFIGDIR=/tmp/matplotlib
EXPOSE 8000
CMD ["python", "-m", "premier_league_predictor.api"]
