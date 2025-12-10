# models.py
from django.db import models

class UserInfo(models.Model):
    name = models.CharField('用户名',max_length = 10)
    password = models.CharField('密码',max_length = 15)

    def __str__(self):
        return self.name