import os
import cv2

from PIL import Image
import numpy as np
import pandas as pd
import folium
from folium import plugins
import tensorflow as tf
from django.core.files.uploadedfile import InMemoryUploadedFile
import base64
import io
from io import BytesIO
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from django.contrib import messages
from django.http import HttpResponseServerError, Http404, JsonResponse
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
import json
import logging
from django.db.models import Count
from random import randint

from .forms import HerbForm, MapCommentForm, TestimonialsForm
from .models import Herb, Store, Favorite, MapHerb, MapComment, Testimonials

def suggest_similarities(herbs):
    # Extract relevant fields from herbs
    text_fields = ['description', 'med_property', 'med_use']
    herb_texts = [' '.join(getattr(herb, field, '') for field in text_fields) for herb in herbs]

    # Use CountVectorizer to convert herb texts into a bag-of-words representation
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(herb_texts)

    # Calculate cosine similarity between herb texts
    similarity_matrix = cosine_similarity(X)

    # Create a dictionary to store suggested similarities
    suggestions = {}

    # Iterate through herbs and suggest similarities
    for idx, herb in enumerate(herbs):
        # Get the similarity scores for the current herb
        similarity_scores = similarity_matrix[idx]

        # Find indices of herbs with high similarity scores
        similar_indices = [i for i, score in enumerate(similarity_scores) if score > 0.5 and i != idx]

        # Store the suggestions for the current herb
        suggestions[herb.name] = [herbs[i].name for i in similar_indices]

    return suggestions

# ------------------------------------------------------------------------v
logger = logging.getLogger(__name__)

class CustomFileSystemStorage(FileSystemStorage):
    def get_available_name(self, name, max_length=None):
        self.delete(name)
        return name

def home(request):

    herbs = Herb.objects.all()

    return render(request, 'core/home.html', {'herbs': herbs})

@staff_member_required(login_url=reverse_lazy('login_or_register'))
def admin_home(request):
    herbs = Herb.objects.all()
    

    context = {
        'herbs':herbs
    }

    return render(request, 'dashboard.html', context )

def dash_herb_user(request):
    mapherbs = MapHerb.objects.all()

    context = {
        'mapherbs':mapherbs
    }

    return render(request,  'dash-herb-user.html', context)


def herbs(request):

    herbs = Herb.objects.all()
    favorite_herbs = request.user.favorite_set.all().values_list('herb', flat=True)
    # Usage:
    suggested_similarities = suggest_similarities(herbs)
    # Get the total number of herbs
    total_herbs = Herb.objects.count()
    # Get a random index
    random_index = randint(0, total_herbs - 1)
    # Get a random herb
    random_herb = Herb.objects.all()[random_index]
    testimonials = Testimonials.objects.all()

    form = TestimonialsForm()

    if request.method == 'POST':
        form = TestimonialsForm(request.POST)
        if form.is_valid():
            testimonial = form.save(commit=False)
            testimonial.name = request.user
            # Get the Herb instance associated with the comment
            herb_instance = form.cleaned_data['herb']
            herb_id = herb_instance.id
            testimonial.herb = herb_instance
            testimonial.save()
            print('Data saved successfully')
            return redirect(request.path + '?submitted=true')
        else:
            print('Form is not valid')  # Add this line for debugging
            print(form.errors)  # Add this line for debugging


    # print(suggested_similarities)
    context = {
        'herbs': herbs,
        'favorite_herbs':favorite_herbs,
        'suggested_similarities':suggested_similarities,
        'random_herb':random_herb,
        'testimonials':testimonials,
    }

    return render(request, 'herbs.html', context)

@login_required(login_url='login_or_register')
def deletetesti(request, id):
    com = get_object_or_404(Testimonials, id=id)
    com.delete()

    return redirect(request.META['HTTP_REFERER'])

def image_search(request):

    return render(request, 'image-search.html')

def update_user_data(request):
    try:
        if request.method == 'POST':
            location_data = json.loads(request.POST.get('location', '{}'))
            lat = location_data.get('lat')
            long = location_data.get('long')
            herb_name = request.POST.get('herb')

            image = request.FILES.get('image')

            if lat is not None and long is not None and herb_name is not None and image is not None:
                # Check if the user has already uploaded an entry
                uploader = request.user
                existing_entries = MapHerb.objects.filter(uploader=uploader)
                
                if existing_entries.exists():
                    return JsonResponse({'error': 'User has already uploaded an entry'})
                else:
                    # Store or update the user's location and form data in your Django models
                    MapHerb.objects.create(
                        uploader=uploader,
                        herb=herb_name,
                        image=image,
                        lat=lat,
                        long=long,
                    )
                    logger.info('Data updated successfully')
                    return JsonResponse({'status': 'Location Added Successfully'})
            else:
                return JsonResponse({'error': 'Invalid form data'})
        else:
            return JsonResponse({'error': 'Invalid request method'})
    except Exception as e:
        logger.exception('An error occurred: %s', str(e))
        return JsonResponse({'error': 'Internal server error'})


def herbal_map(request):
    herbs = Herb.objects.all()
    stores = Store.objects.all()
    mapherbs = MapHerb.objects.all()

    # Create a map
    m = folium.Map(location=[6.918658, 122.077802], zoom_start=13, control_scale=True, max_zoom=20, min_zoom=2, max_bounds=True)

    # OpenTopoMap tile layer with attribution
    folium.TileLayer('https://tile.opentopomap.org/{z}/{x}/{y}.png', name='OpenTopoMap', attr='Map tiles by OpenTopoMap, under CC BY-SA 3.0').add_to(m)

    folium.TileLayer('cartodbpositron').add_to(m)  # CartoDB Positron
    # Esri tile layer with attribution
    esri_tile_url = 'http://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}'
    esri_tile_attribution = 'Tiles &copy; Esri'

    folium.TileLayer(esri_tile_url, name='esri', attr=esri_tile_attribution).add_to(m)

    herb_group = folium.FeatureGroup(name='Herbs')
    store_group = folium.FeatureGroup(name='Stores')
    herb_user_group = folium.FeatureGroup(name='User Uploads')

    # Iterate over store objects and create markers
    for store in stores:
        lat = store.lat
        long = store.long

        if lat is not None and long is not None:
            popup = folium.Popup(store.name)
            folium.Marker(
                location=[lat, long],
                popup=popup,
            ).add_to(store_group)

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
            ).add_to(herb_group)
            
    for herb in mapherbs:
        lat = herb.lat
        long = herb.long
        name = herb.herb

        if lat is not None and long is not None:
            popup = folium.Popup(name)
            folium.CircleMarker(
                location=[lat, long],
                radius=20,
                color='green',
                fill=True,
                fill_color='green',
                fill_opacity=0.4,
                popup=popup,
            ).add_to(herb_user_group)

    # Add feature groups to the map
    herb_group.add_to(m)
    store_group.add_to(m)
    herb_user_group.add_to(m)

    # Add layer control to the map
    folium.LayerControl().add_to(m)

    # Get the map HTML
    map_html = m._repr_html_()
    context = { 
        'map': map_html,
        'herbs': herbs,
        'store': store,
        'mapherbs': mapherbs,
    }

    return render(request, 'herbal-map.html', context)

def herbal_map_inter(request, id=None, herb=None):
    herbs = Herb.objects.all()
    stores = Store.objects.all()
    mapherbs = MapHerb.objects.all()
    herbb = None
    mapherb = None

    try:
        herbb = Herb.objects.get(id=id)
    except Herb.DoesNotExist:
        try:
            mapherb = MapHerb.objects.get(herb=herb)
        except MapHerb.DoesNotExist:
            pass

    form = MapCommentForm()
    if request.method == 'POST':
        form = MapCommentForm(request.POST)
        if form.is_valid():
            print('Form is valid')  # Add this line for debugging
            mherb = form.save(commit=False)
            mherb.username = request.user
            mherb.map_herb = mapherb
            mherb.save()
            print('Data saved successfully')  # Add this line for debugging
            return redirect(request.path + '?submitted=true')
        else:
            print('Form is not valid')  # Add this line for debugging
            print(form.errors)  # Add this line for debugging

    # Create a map
    if herbb and herbb.lat is not None and herbb.long is not None:
        m = folium.Map(location=[herbb.lat, herbb.long], zoom_start=16, control_scale=True, max_zoom=20, min_zoom=2, max_bounds=True)
    elif mapherb and mapherb.lat is not None and mapherb.long is not None:
        m = folium.Map(location=[mapherb.lat, mapherb.long], zoom_start=16, control_scale=True, max_zoom=20, min_zoom=2, max_bounds=True)
    else:    
        m = folium.Map(location=[6.918658, 122.077802], zoom_start=13, control_scale=True, max_zoom=20, min_zoom=2, max_bounds=True)
        messages.info(request, 'Location is Not Available')

    # OpenTopoMap tile layer with attribution
    folium.TileLayer('https://tile.opentopomap.org/{z}/{x}/{y}.png', name='OpenTopoMap', attr='Map tiles by OpenTopoMap, under CC BY-SA 3.0').add_to(m)

    folium.TileLayer('cartodbpositron').add_to(m)  # CartoDB Positron
    # Esri tile layer with attribution
    esri_tile_url = 'http://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}'
    esri_tile_attribution = 'Tiles &copy; Esri'

    folium.TileLayer(esri_tile_url, name='esri', attr=esri_tile_attribution).add_to(m)

    herb_group = folium.FeatureGroup(name='Herbs')
    store_group = folium.FeatureGroup(name='Stores')
    herb_user_group = folium.FeatureGroup(name='User Uploads')

    for herbb in herbs:
        lat = herbb.lat
        long = herbb.long
        job_name = herbb.name

        if lat is not None and long is not None:
            marker = folium.CircleMarker(location=[lat, long],radius=20,color='green',fill=True,fill_color='green',
                fill_opacity=0.6, popup=job_name)
            marker.add_to(herb_group)

    # Iterate over store objects and create markers
    for store in stores:
        lat = store.lat
        long = store.long

        if lat is not None and long is not None:
            popup = folium.Popup(store.name)
            folium.Marker(
                location=[lat, long],
                popup=popup,
            ).add_to(store_group)

    for herbb in mapherbs:
        lat = herbb.lat
        long = herbb.long
        name = herbb.herb

        if lat is not None and long is not None:
            popup = folium.Popup(name)
            folium.CircleMarker(
                location=[lat, long],
                radius=20,
                color='green',
                fill=True,
                fill_color='green',
                fill_opacity=0.4,
                popup=popup,
            ).add_to(herb_user_group)

    # Add feature groups to the map
    herb_group.add_to(m)
    store_group.add_to(m)
    herb_user_group.add_to(m)

    # Add layer control to the map
    folium.LayerControl().add_to(m)

    # Render the map to HTML
    context = {
        'map': m._repr_html_(),
        'herbs': herbs,
        'herbb': herbb,
        'mapherbs': mapherbs,
        'mapherb': mapherb,
        'form':form,
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
            job.save()
            return redirect('dashboard')
        
    return render(request, 'herb-cms.html', {'form':form})

class edit(LoginRequiredMixin, UserPassesTestMixin, UpdateView):
    model = Herb
    fields = '__all__'
    success_url = reverse_lazy('dashboard')
    template_name = 'herb-cms.html'

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
    
@staff_member_required
def delete(request, id):
    herb = Herb.objects.get(id=id)

    # Get the path to the image file
    image_path = herb.image.path

    # Delete the Herb object
    herb.delete()

    # Delete the image file from storage
    default_storage.delete(image_path)

    return redirect('dashboard')

def deletecomment(request, id):
    com = get_object_or_404(MapComment, id=id)
    com.delete()

    return redirect(request.META['HTTP_REFERER'])

def search(request):
    query = request.GET.get('query', '')

    herbs = Herb.objects.filter(Q(name__icontains=query) | Q(description__icontains=query) | Q(med_use__icontains=query) | Q(med_property__icontains=query) | Q(potential_SE__icontains=query))

    if not herbs:
        messages.info(request, ':( Sorry Keyword not Found Please Try Again')

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