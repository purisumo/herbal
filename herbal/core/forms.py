from django.forms import ModelForm
from django import forms
from registration.models import User
from .models import Herb

class HerbForm(forms.ModelForm):
    class Meta:
        model = Herb
        fields = '__all__'
        labels = '__all__'