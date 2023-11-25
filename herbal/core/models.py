import os
import uuid
from django.utils.text import slugify
from django.core.exceptions import ValidationError
from django.db import models
from django.core.files.storage import default_storage
from registration.models import User
import re
from django.utils import timezone

class DuplicateImageError(Exception):
    pass

def sanitize_filename(filename):
    # Remove special characters and spaces
    sanitized_filename = re.sub(r'[^a-zA-Z0-9_.-]', '', filename)
    return sanitized_filename
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
    name = models.ForeignKey(User, on_delete=models.CASCADE, related_name='testimony')
    herb = models.ForeignKey(Herb, on_delete=models.CASCADE, related_name='testimonials')
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
    herb = models.ForeignKey('Herb', on_delete=models.CASCADE, related_name='favorites')
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.user.username} likes {self.herb.name}'
    
class MapHerb(models.Model):
    uploader = models.ForeignKey(User, on_delete=models.CASCADE, related_name='mapherb')
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
    username = models.ForeignKey(User, on_delete=models.CASCADE, related_name='mapcomment')
    map_herb = models.ForeignKey('MapHerb', on_delete=models.CASCADE, related_name='comments')
    comment = models.TextField(blank=False, null=True)

    def __str__(self):
        return self.username

def upload_to_dataset(instance, filename):

    sanitized_filename = sanitize_filename(filename)
    base_folder = "Datasets" 

    subfolder = instance.class_name.class_name

    return os.path.join(base_folder, subfolder, sanitized_filename)

class Datasets(models.Model):
    class_name = models.TextField(max_length=255, blank=False, null=True)
    images = models.ManyToManyField('DatasetImages', related_name='datasetimages', blank=True)

    def __str__(self):
        return self.class_name

    def delete(self, *args, **kwargs):
        for image in self.images.all():
            default_storage.delete(image.images.path)
            image.delete()

        super().delete(*args, **kwargs)

class DatasetImages(models.Model):
    class_name = models.ForeignKey(Datasets, default=None, on_delete=models.CASCADE, null=True, blank=True , related_name='datasetimageclass')
    images = models.ImageField(upload_to=upload_to_dataset, verbose_name='Image', blank=True)

    def __str__(self):
        return f"{self.class_name} - {os.path.basename(self.images.name)}"
    
    def save(self, *args, **kwargs):
        # Extract the class_name folder from the image path
        class_name_folder = os.path.dirname(self.images.name)

        # Sanitize the filename
        sanitized_filename = sanitize_filename(os.path.basename(self.images.name))

        # Check if an image with the same name already exists in the same class_name folder
        existing_images = DatasetImages.objects.filter(
            images__icontains=f"{class_name_folder}/{sanitized_filename}"
        )

        if existing_images.exists():
            timestamp = timezone.now().strftime('%Y%m%d%H%M%S')
            new_filename = f"{sanitized_filename}_{timestamp}"

            # Update the image path with the sanitized filename
            self.images.name = os.path.join(class_name_folder, new_filename)

        super().save(*args, **kwargs)


class MachineModel(models.Model):
    herb = models.FileField(upload_to='MachineModel/')
    label = models.FileField(upload_to='MachineModel/')

    def __str__(self):
        return f"MachineModel - Herb: {self.herb.name}, Label: {self.label.name}"
    

class HerbalistProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    fname = models.CharField(max_length=255)
    minitial = models.CharField(max_length=255)
    lname = models.CharField(max_length=255)
    image = models.ImageField(upload_to='experts/')
    profession = models.TextField(max_length=255)
    detail = models.TextField(max_length=255)

    def __str__(self):
        return self.user.username