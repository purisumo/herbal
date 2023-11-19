# Generated by Django 4.2.5 on 2023-11-14 09:03

import core.models
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0013_datasets'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='datasets',
            name='images',
        ),
        migrations.CreateModel(
            name='DatasetImage',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.FileField(upload_to=core.models.upload_to_dataset)),
                ('dataset', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='core.datasets')),
            ],
        ),
    ]