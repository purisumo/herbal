from django.db import models
from registration.models import User

# Create your models here.

class Herb(models.Model):
    name = models.CharField(max_length=255, blank=False, null=True)
    property = models.TextField( blank=True, null=True)
    use = models.TextField(blank=True, null=True)
    image = models.ImageField(upload_to='Herbs/', blank=True)

    def __str__(self):
        return self.name
    
class Favorite(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    herb = models.ForeignKey('Herb', on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.user.username} likes {self.herb.name}'