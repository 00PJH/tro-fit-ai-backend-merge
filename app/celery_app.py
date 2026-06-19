# Celery 초기화 (비동기 작업 필요 시 사용)
#
# from celery import Celery
# 
# celery_app = Celery("tasks", broker="redis://localhost:6379/0")
# 
# @celery_app.task
# def add(x, y):
#     return x + y
