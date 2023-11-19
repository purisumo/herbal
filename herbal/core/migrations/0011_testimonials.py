# Generated by Django 4.2.5 on 2023-11-10 22:01

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('core', '0010_herb_timestamp'),
    ]

    operations = [
        migrations.CreateModel(
            name='Testimonials',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('comment', models.TextField(null=True)),
                ('time', models.DateTimeField(auto_now_add=True)),
                ('herb', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='core.herb')),
                ('name', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]