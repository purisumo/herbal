# Generated by Django 4.2.5 on 2023-11-14 11:56

import core.models
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0018_alter_datasetimages_images'),
    ]

    operations = [
        migrations.AlterField(
            model_name='datasetimages',
            name='images',
            field=models.ImageField(blank=True, upload_to=core.models.upload_to_dataset, verbose_name='Image'),
        ),
    ]
