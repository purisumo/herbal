from django.forms import ModelForm
from django import forms
from registration.models import User
from .models import Herb, MapHerb, MapComment, Testimonials

class HerbForm(forms.ModelForm):
    class Meta:
        model = Herb
        fields = '__all__'
        labels = '__all__'

class HerbMapForm(forms.ModelForm):
    class Meta:
        model = MapHerb
        fields = '__all__'
        labels = '__all__'

class MapCommentForm(forms.ModelForm):
    class Meta:
        model = MapComment
        fields = ['comment', 'map_herb']
        labels = '__all__'

class TestimonialsForm(forms.ModelForm):
    class Meta:
        model = Testimonials
        fields = ['comment','herb']
        labels = '__all__'