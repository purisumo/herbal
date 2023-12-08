from django.forms import ModelForm
from django import forms
from django.forms import ClearableFileInput
from registration.models import User
from django.forms.models import inlineformset_factory
from django.utils.html import format_html
from .models import *

class HerbForm(forms.ModelForm):
    class Meta:
        model = Herb
        fields = '__all__'

class HerbImagesForm(forms.ModelForm):
    class Meta:
        model = HerbImages
        fields = ['images']  # Only include the 'images' field
        widgets = {
            'images': forms.FileInput(attrs={'multiple': True, 'accept': 'image/*'}),
        }

HerbImagesFormSet = forms.inlineformset_factory(Herb, HerbImages, form=HerbImagesForm, extra=1, can_delete=True)

class HerbStore(forms.ModelForm):
    class Meta:
        model = Store
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

class DatasetImagesForm(forms.ModelForm):
    class Meta:
        model = DatasetImages
        fields = ['images']
        widgets = {
            'images': forms.FileInput(attrs={'multiple': True, 'accept': 'image/*'}),
        }
        
DatasetImagesFormSet = inlineformset_factory(Datasets, DatasetImages, form=DatasetImagesForm, extra=1, can_delete=True)

class DatasetForm(forms.ModelForm):
    class Meta:
        model = Datasets
        fields = '__all__'
        labels = '__all__'