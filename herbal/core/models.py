from django.db import models
from registration.models import User

# Create your models here.

class Herb(models.Model):
    name = models.CharField(max_length=255, blank=False, null=True)
    scientific_name = models.CharField(max_length=255, blank=False, null=True)
    lat = models.FloatField(blank=True, null=True)
    long = models.FloatField(blank=True, null=True)
    description = models.TextField(blank=True, null=True)
    med_property = models.TextField(blank=True, null=True)
    med_use = models.TextField(blank=True, null=True)
    habitat = models.TextField(blank=True, null=True)
    potential_SE = models.TextField(blank=True, null=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    image = models.ImageField(upload_to='Herbs/', blank=True)

    def __str__(self):
        return self.name
    
    def display_name(self):
        return self.name

class Testimonials(models.Model):
    name = models.ForeignKey(User, on_delete=models.CASCADE)
    herb = models.ForeignKey(Herb, on_delete=models.CASCADE)
    comment = models.TextField(blank=False, null=True)
    time = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.comment

class Store(models.Model):
    name = models.CharField(max_length=255, blank=False, null=True)
    lat = models.FloatField(blank=True, null=True)
    long = models.FloatField(blank=True, null=True)
    description = models.TextField(blank=False, null=True)

    def __str__(self):
        return self.name

class Favorite(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    herb = models.ForeignKey('Herb', on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.user.username} likes {self.herb.name}'
    
class MapHerb(models.Model):
    uploader = models.ForeignKey(User, on_delete=models.CASCADE)
    herb = models.CharField(max_length=255, blank=False, null=True)
    image = models.ImageField(upload_to='HerbMap/', blank=False, null=True)
    lat = models.FloatField()
    long = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.herb
    
    def display_name(self):
        return self.herb

class MapComment(models.Model):
    username = models.ForeignKey(User, on_delete=models.CASCADE)
    map_herb = models.ForeignKey('MapHerb', on_delete=models.CASCADE, related_name='comments')
    comment = models.TextField(blank=False, null=True)

    def __str__(self):
        return self.username

