import os
import cv2

from PIL import Image
import numpy as np
import tensorflow as tf
from django.core.files.uploadedfile import InMemoryUploadedFile
import base64
import io
from io import BytesIO

from django.http import HttpResponseServerError
from django.shortcuts import render, HttpResponse, redirect, get_object_or_404
from django.db.models import Q
from django.conf import settings
from django.template.response import TemplateResponse
from django.utils.datastructures import MultiValueDictKeyError
from django.core.files.storage import FileSystemStorage
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.views.generic import CreateView, UpdateView
from django.core.files.storage import default_storage
from django.urls import reverse_lazy, reverse

from .forms import HerbForm
from .models import Herb, Favorite
import time
import tempfile
import requests

from django.core.files import File
from django.core.files.base import ContentFile
from django.core.files.temp import NamedTemporaryFile
from urllib.request import urlopen
# Create your views here.

def home(request):
    
    return render(request, 'core/home.html')

def herbs(request):

    return render(request, 'herbs.html')

def illness(request):

    return render(request, 'illness.html')

def favourite(request):

    return render(request, 'favourite.html')

def toggle_favorite(request, herb_id):
    herb = get_object_or_404(Herb, pk=herb_id)
    user = request.user

    if user.favorite_set.filter(herb=herb).exists():
        # Herb is already a favorite, remove it
        user.favorite_set.get(herb=herb).delete()
    else:
        # Herb is not a favorite, add it
        Favorite.objects.create(user=user, herb=herb)

    return redirect('herb_detail', herb_id=herb_id)

@staff_member_required
def add(request):

    form = HerbForm()
    if request.method == 'POST':
        form = HerbForm(request.POST, request.FILES)
        if form.is_valid():
            job = form.save(commit=False)
            job.save
            return redirect()
        
    return render(request, 'add.html', {'form':form})

class edit(LoginRequiredMixin, UserPassesTestMixin, UpdateView):
    model = Herb
    fields = '__all__'
    success_url = reverse_lazy()
    template_name = 'edit.html'

    def test_func(self):
        return self.request.user.is_staff or self.request.user.is_superuser
    
    def post(self, request, *args, **kwargs):
        # Get the existing object
        self.object = self.get_object()

        # Delete previous image if it exists
        previous_image = self.object.image
        if previous_image:
            default_storage.delete(previous_image.path)

        return super().post(request, *args, **kwargs)

def search(request):
    query = request.GET.get('query', '')

    herbs = Herb.objects.filter(Q(name__icontains=query) | Q(use__icontains=query) | Q(property__icontains=query))

    return render(request, 'core/search.html', {'herbs': herbs})


class CustomFileSystemStorage(FileSystemStorage):
    def get_available_name(self, name, max_length=None):
        self.delete(name)
        return name

def recognition(request):
    message = ""
    # prediction = ""
    fss = CustomFileSystemStorage()
    try:
        image = request.FILES["image"]
        print("Name", image.file)
        _image = fss.save(image.name, image)
        path = str(settings.MEDIA_ROOT) + "/" + image.name
        # image details
        image_url = fss.url(_image)
        # Read the image
        imag = cv2.imread(path)
        img_from_ar = Image.fromarray(imag, 'RGB')
        resized_image = img_from_ar.resize((224, 224))

        test_image = np.expand_dims(resized_image, axis=0)

        # Load model
        model = tf.keras.models.load_model(os.getcwd() + '/model.h5')

        result = model.predict(test_image)

        # Define class mapping
        class_mapping = {
            0: "Adelfa",
            1: "ALOE VERA",
            2: "sambong",
            # Add more classes here if needed
        }

        # Get the predicted class name
        predicted_class_index = np.argmax(result)
        predicted_class_name = class_mapping.get(predicted_class_index, "Unknown")

        return TemplateResponse(
            request,
            "recognition.html",
            {
                "message": message,
                "image": image,
                "image_url": image_url,
                "prediction": predicted_class_name,
            },
        )
    except MultiValueDictKeyError:
        return TemplateResponse(
            request,
            "recognition.html",
            {"message": "No Image Selected"},
        )

def cam_recognition(request):
    if request.method == 'POST':
        image_data_uri = request.POST.get("src")

        try:
            # Extract the base64-encoded image data from the data URI
            _, image_data = image_data_uri.split(",", 1)
            image_bytes = base64.b64decode(image_data)

            # Create a BytesIO stream from the image data
            image_stream = io.BytesIO(image_bytes)

            # Load the image with OpenCV
            imag = cv2.imdecode(np.frombuffer(image_stream.read(), np.uint8), cv2.IMREAD_COLOR)
            img_from_ar = Image.fromarray(imag, 'RGB')
            resized_image = img_from_ar.resize((224, 224))
            test_image = np.expand_dims(resized_image, axis=0)

            # Load model
            model = tf.keras.models.load_model(os.getcwd() + '/model.h5')
            result = model.predict(test_image)

            # Define class mapping
            class_mapping = {
                0: "Adelfa",
                1: "ALOE VERA",
                2: "sambong",
                # Add more classes here if needed
            }

            # Get the predicted class name
            predicted_class_index = np.argmax(result)
            predicted_class_name = class_mapping.get(predicted_class_index, "Unknown")

            return render(request, "cam-recognition.html", {"prediction": predicted_class_name})

        except Exception as e:
            # Handle errors gracefully
            return HttpResponseServerError(f"Error: {str(e)}")

    return render(request, 'cam-recognition.html')