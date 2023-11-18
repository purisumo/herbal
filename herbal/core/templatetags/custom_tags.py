from django import template
from core.models import DatasetImages

register = template.Library()

@register.simple_tag
def get_total_images(class_name):
    return DatasetImages.objects.filter(class_name__class_name=class_name).count()
