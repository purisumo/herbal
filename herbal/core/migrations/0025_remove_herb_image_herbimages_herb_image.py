# Generated by Django 4.2.5 on 2023-12-01 07:07

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0024_alter_herb_image'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='herb',
            name='image',
        ),
        migrations.CreateModel(
            name='HerbImages',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('images', models.ImageField(blank=True, upload_to='HerbalImages/', verbose_name='Image')),
                ('herb', models.ForeignKey(blank=True, default=None, null=True, on_delete=django.db.models.deletion.CASCADE, to='core.herb')),
            ],
        ),
        migrations.AddField(
            model_name='herb',
            name='image',
            field=models.ManyToManyField(blank=True, related_name='herbimages', to='core.herbimages'),
        ),
    ]
