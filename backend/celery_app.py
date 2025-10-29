# backend/celery_app.py

import os
import contextvars
from celery import Celery
from celery.signals import before_task_publish, task_prerun, worker_process_init
from dotenv import load_dotenv

# This context variable will hold the request_id
request_id_var = contextvars.ContextVar('request_id', default=None)

@before_task_publish.connect
def propagate_request_id(sender=None, headers=None, body=None, **kwargs):
    """Injects the request_id from the current context into the task headers before it's sent."""
    request_id = request_id_var.get()
    if request_id:
        if headers is None:
            headers = {}
        headers['request_id'] = request_id

@task_prerun.connect
def set_request_id_on_worker(sender=None, task_id=None, task=None, args=None, kwargs=None, **other_kwargs):
    """Sets the request_id in the worker's context from the task headers.
    This makes it available for logging within the task.
    """
    request_id = task.request.get('request_id')
    if request_id:
        request_id_var.set(request_id)

@worker_process_init.connect
def configure_worker_logging(**kwargs):
    """Configures structured logging for the celery worker process."""
    from app.logging_conf import configure_logging
    configure_logging()


load_dotenv()

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0")

celery_app = Celery(
    "tasks",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["tasks"]
)

celery_app.conf.update(
    task_track_started=True,
    result_expires=3600, # Store results for 1 hour
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
)
