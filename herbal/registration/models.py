from django.contrib.auth.models import AbstractUser
from django.db import models
# Create your models here.

class User(AbstractUser):
    mobile = models.IntegerField(null=True, blank=True)
    is_email_verified = models.BooleanField(default=False)
    
    def __str__(self):
        return self.username