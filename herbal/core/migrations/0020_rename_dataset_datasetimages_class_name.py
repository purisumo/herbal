# Generated by Django 4.2.5 on 2023-11-14 12:59

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0019_alter_datasetimages_images'),
    ]

    operations = [
        migrations.RenameField(
            model_name='datasetimages',
            old_name='dataset',
            new_name='class_name',
        ),
    ]