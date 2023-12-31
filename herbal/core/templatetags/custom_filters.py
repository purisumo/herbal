from django import template

register = template.Library()

@register.filter
def add_class(field, css_class):
    return field.as_widget(attrs={'class': css_class})

@register.filter(name='get_value')
def get_value(dictionary, key):
    return dictionary.get(key, None)

@register.filter(name='sum_attribute')
def sum_attribute(queryset, attribute):
    return sum(entry[attribute] for entry in queryset)
