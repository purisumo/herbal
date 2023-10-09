import os
import cv2

from PIL import Image
import numpy as np
import pandas as pd
import folium
import tensorflow as tf
from django.core.files.uploadedfile import InMemoryUploadedFile
import base64
import io
from io import BytesIO
import math

from django.http import HttpResponseServerError, Http404
from django.shortcuts import render, HttpResponse, redirect, get_object_or_404
from django.db.models import Q
from django.conf import settings
from django.template.response import TemplateResponse
from django.utils.datastructures import MultiValueDictKeyError
from django.core.files.storage import FileSystemStorage
from django.contrib.auth.decorators import login_required
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.views.generic import CreateView, UpdateView
from django.core.files.storage import default_storage
from django.urls import reverse_lazy, reverse

from .forms import HerbForm
from .models import Herb, Store, Favorite

# ------------------------------------------------------------------------v

class CustomFileSystemStorage(FileSystemStorage):
    def get_available_name(self, name, max_length=None):
        self.delete(name)
        return name

def home(request):

    herbs = Herb.objects.all()

    return render(request, 'core/home.html', {'herbs': herbs})

def herbs(request):

    herbs = Herb.objects.all()
    favorite_herbs = request.user.favorite_set.all().values_list('herb', flat=True)

    return render(request, 'herbs.html', {'herbs': herbs, 'favorite_herbs':favorite_herbs})

def image_search(request):

    return render(request, 'image-search.html')

def herbal_map(request):
    herbs = Herb.objects.all()
    stores = Store.objects.all()

    # Create a map
    m = folium.Map(location=[6.918658, 122.077802], zoom_start=13, control_scale=True, max_zoom=20, min_zoom=2, max_bounds=True)

    # Iterate over store objects and create markers
    for store in stores:
        lat = store.lat
        long = store.long

        if lat is not None and long is not None:
            popup = folium.Popup(store.name)
            folium.Marker(
                location=[lat, long],
                popup=popup,
            ).add_to(m)

    # Iterate over herb objects and create markers
    for herb in herbs:
        lat = herb.lat
        long = herb.long


        if lat is not None and long is not None:

            popup = folium.Popup(herb.name)

            folium.CircleMarker(
                location=[lat, long],
                radius=20,
                color='green',
                fill=True,
                fill_color='green',
                fill_opacity=0.6,
                popup=popup,
            ).add_to(m)

    # Get the map HTML
    map_html = m._repr_html_()
    context = {
        'map':map_html,
        'herbs':herbs
    }

    return render(request, 'herbal-map.html', context)

def herbal_map_inter(request, name):
    herbs = Herb.objects.all()
    stores = Store.objects.all()
    herb = Herb.objects.get(name=name)

    # Create a map
    m = folium.Map(location=[herb.lat, herb.long], zoom_start=16, control_scale=True, max_zoom=20, min_zoom=2, max_bounds=True)

    for herb in herbs:
        lat = herb.lat
        long = herb.long
        job_name = herb.name
            
        marker = folium.Marker(location=[lat, long], popup=job_name)
        marker.add_to(m)

    for store in stores:
        lat = store.lat
        long = store.long


        if lat is not None and long is not None:

            popup = folium.Popup(store.name)

            folium.CircleMarker(
                location=[lat, long],
                radius=20,
                color='green',
                fill=True,
                fill_color='green',
                fill_opacity=0.6,
                popup=popup,
            ).add_to(m)


    # Render the map to HTML
    context = {
        'map': m._repr_html_(),
        'herbs': herbs,
        'herb': herb
    }

    return render(request, 'herbal-map.html', context)

@login_required(login_url='login_or_register')
def favourite(request):
    user = request.user

    # Create a dictionary to store whether each herb is a favorite for the user
    favorite_herbs = Herb.objects.filter(favorite__user=user)

    return render(request, 'favorite.html', {'favorite_herbs': favorite_herbs})

@login_required
def toggle_favorite(request, herb_id):
    herb = get_object_or_404(Herb, pk=herb_id)
    user = request.user

    if user.favorite_set.filter(herb=herb).exists():
        # Herb is already a favorite, remove it
        user.favorite_set.get(herb=herb).delete()
    else:
        # Herb is not a favorite, add it
        Favorite.objects.create(user=user, herb=herb)

    # Redirect to the referring URL
    referring_url = request.META.get('HTTP_REFERER')
    if referring_url:
        return redirect(referring_url)
    else:
        # If no referring URL is available, redirect to a default page
        return redirect('herbs')

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

    herbs = Herb.objects.filter(Q(name__icontains=query) | Q(description__icontains=query) | Q(med_use__icontains=query) | Q(med_property__icontains=query) | Q(potential_SE__icontains=query))

    if not herbs:
        raise Http404("No matching herbs found")

    return render(request, 'herbs.html', {'herbs': herbs})


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
        test_image = test_image.astype('float32') / 255.0  # Normalize the image

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

        predicted_class_index = np.argmax(result)
        predicted_class_probability = result[0][predicted_class_index]

        # Set your confidence threshold (e.g., 0.7)
        confidence_threshold = 80
        print('probability' + str(predicted_class_probability))
        if predicted_class_probability < confidence_threshold:
            predicted_class_name = "Unknown"
        else:
            if predicted_class_index in class_mapping:
                predicted_class_name = class_mapping[predicted_class_index]
            else:
                predicted_class_name = "Unknown"

        print( 'index' + str(predicted_class_index))
        return TemplateResponse(
            request,
            "recognition.html",
            {
                "probability": predicted_class_probability,
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
            test_image = test_image.astype('float32') / 255.0  # Normalize the image

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

            predicted_class_index = np.argmax(result)
            predicted_class_probability = result[0][predicted_class_index]
            # Set a threshold for class probability

            confidence_threshold = 80  # Adjust this threshold as needed

            # Check if any class probability exceeds the threshold
            print('probability' + str(predicted_class_probability))
            if predicted_class_probability < confidence_threshold:
                predicted_class_name = "Unknown"
            else:
                if predicted_class_index in class_mapping:
                    predicted_class_name = class_mapping[predicted_class_index]
                else:
                    predicted_class_name = "Unknown"

            print( 'index' + str(predicted_class_index))

            return render(request, "cam-recognition.html", {"prediction": predicted_class_name, 'probability': predicted_class_probability})

        except Exception as e:
            # Handle errors gracefully
            return HttpResponseServerError(f"Error: {str(e)}")

    return render(request, 'cam-recognition.html')


def map_endpoint(request):
    herbs = Herb.objects.all()
    stores = Store.objects.all()

    # Create a map
    m = folium.Map(location=[6.918658, 122.077802], zoom_start=13, control_scale=True, max_zoom=20, min_zoom=2, max_bounds=True)

    # Iterate over store objects and create markers
    for store in stores:
        lat = store.lat
        long = store.long

        if lat is not None and long is not None:
            popup = folium.Popup(store.name)
            folium.Marker(
                location=[lat, long],
                popup=popup,
            ).add_to(m)

    # Iterate over herb objects and create markers
    for herb in herbs:
        lat = herb.lat
        long = herb.long


        if lat is not None and long is not None:

            popup = folium.Popup(herb.name)

            folium.CircleMarker(
                location=[lat, long],
                radius=20,
                color='green',
                fill=True,
                fill_color='green',
                fill_opacity=0.6,
                popup=popup,
            ).add_to(m)

    # Get the map HTML
    map_html = m._repr_html_()

    return HttpResponse(map_html)