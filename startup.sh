#!/bin/sh
exec gunicorn app.main:app --bind 0.0.0.0:${PORT:-8000} --worker-class uvicorn.workers.UvicornWorker --workers 1 --timeout 120
