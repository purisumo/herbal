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
from django.http import HttpResponseServerError, Http404, JsonResponse, HttpResponse, HttpResponseForbidden
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
from zipfile import ZipFile
from django.views.decorators.csrf import csrf_exempt

import json
import logging
from registration.models import User

from django.db import transaction
from django.db.models import Count, DateField
from datetime import datetime
from django.db.models.functions import TruncWeek, TruncHour, TruncDate
from random import randint
from django.utils import timezone
from django.contrib.auth import get_user_model, login, logout
from django.contrib.sessions.models import Session
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from .consumers import TrainingProgressConsumer

from .forms import *
from .models import *

def suggest_similarities(herbs):
    # Extract relevant fields from herbs
    text_fields = ['description', 'med_property', 'med_use']
    herb_texts = [' '.join(str(getattr(herb, field, '')) for field in text_fields) for herb in herbs]

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


def herbs(request):

    herbs = Herb.objects.all()
    favorite_herbs = []

    if request.user.is_authenticated:
        # If authenticated, get the favorite_herbs
        favorite_set = getattr(request.user, 'favorite_set', None)
        if favorite_set:
            favorite_herbs = favorite_set.all().values_list('herb', flat=True)

    # Usage:
    suggested_similarities = suggest_similarities(herbs)
    # Get the total number of herbs
    total_herbs = Herb.objects.count()
    # Get a random index
    random_index = randint(0, total_herbs - 1)
    # Get a random herb
    random_herb = Herb.objects.all()[random_index]
    comments = Testimonials.objects.all()

    form = TestimonialsForm()

    if request.method == 'POST':
        # Check if the user is authenticated
        if request.user.is_authenticated:
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
        else:
            print('User is not authenticated')  # Add this line for debugging
            return redirect( 'login_or_register' )


    # print(suggested_similarities)
    context = {
        'herbs': herbs,
        'favorite_herbs':favorite_herbs,
        'suggested_similarities':suggested_similarities,
        'random_herb':random_herb,
        'comments':comments,
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
                
                MAX_UPLOADS = 3

                if existing_entries.count() >= MAX_UPLOADS:
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
        return JsonResponse({'error': 'Please Log in'})


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
                color='yellow',
                fill=True,
                fill_color='yellow',
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
        if request.user.is_authenticated:
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
        else:
            return redirect('login_or_register')
        
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
                color='yellow',
                fill=True,
                fill_color='yellow',
                fill_opacity=0.1,
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

@login_required(login_url='/login_or_register')
def favourite(request):
    user = request.user

    # Create a dictionary to store whether each herb is a favorite for the user
    favorite_herbs = Herb.objects.filter(favorites__user=user)
    
    uploads = MapHerb.objects.filter(uploader=user)

    context = {
        'favorite_herbs': favorite_herbs,
        'uploads':uploads,
        }

    return render(request, 'favorite.html', context)

class user_edit_upload(LoginRequiredMixin, UserPassesTestMixin, UpdateView):
    model = MapHerb
    fields = ['herb','image']
    success_url = reverse_lazy('dashboard')
    template_name = 'user-cms.html'

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
    
@login_required(login_url='/login_or_register')
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

@staff_member_required(login_url='/login_or_register')
def add(request):
    if request.method == 'POST':
        # Extract form data directly from request.POST and request.FILES
        name = request.POST.get('name')
        scientific_name = request.POST.get('scientific_name')
        lat = request.POST.get('lat')
        long = request.POST.get('long')
        description = request.POST.get('description')
        med_property = request.POST.get('med_property')
        med_use = request.POST.get('med_use')
        habitat = request.POST.get('habitat')
        potential_SE = request.POST.get('potential_SE')

        try:
            # Create Herb instance and save
            herb = Herb(
                name=name,
                scientific_name=scientific_name,
                lat=lat,
                long=long,
                description=description,
                med_property=med_property,
                med_use=med_use,
                habitat=habitat,
                potential_SE=potential_SE,
                timestamp=datetime.now()  # Assuming you want to set the timestamp
            )
            herb.save()

            # Process images
            images = request.FILES.getlist('image')
            for image in images:
                herb_image = HerbImages(herb=herb, images=image)
                herb_image.save()

            return redirect('herbal_upload')

        except Exception as e:
            print(f"Error: {str(e)}")

    return render(request, 'herb-cms-man.html')

@staff_member_required(login_url='/login_or_register')
def edit(request, pk):
    herb = get_object_or_404(Herb, pk=pk)

    # Check if the user is staff or superuser
    if not (request.user.is_staff or request.user.is_superuser):
        return HttpResponseForbidden("You do not have permission to edit this herb.")

    if request.method == 'POST':
        # Use a transaction to ensure consistency
        with transaction.atomic():
            # Check if the user wants to remove existing images
            remove_images = request.POST.get('remove_images')
            if remove_images and remove_images != "0":
                for herb_image in herb.herbimages_set.all():
                    # Get the path to the image file
                    image_path = herb_image.images.path

                    # Delete the HerbImages object
                    herb_image.delete()

                    # Delete the image file from storage
                    default_storage.delete(image_path)

            # Handle the image file
            images = request.FILES.getlist('image')
            for image in images:
                herb_image = HerbImages(herb=herb, images=image)
                herb_image.save()

            # Use the update method to update the model instance
            herb.name = request.POST.get('name')
            herb.scientific_name = request.POST.get('scientific_name')
            herb.lat = request.POST.get('lat')
            herb.long = request.POST.get('long')
            herb.description = request.POST.get('description')
            herb.med_property = request.POST.get('med_property')
            herb.med_use = request.POST.get('med_use')
            herb.habitat = request.POST.get('habitat')
            herb.potential_SE = request.POST.get('potential_SE')

            # Save the updated herb instance
            herb.save()

            # Redirect to the dashboard or any other desired URL
            return redirect('herbal_upload')

    # If it's a GET request, render the form with the existing herb instance
    return render(request, 'herb-cms-man.html', {'herb': herb})
    
@staff_member_required(login_url='/login_or_register')
def delete(request, id):
    herb = Herb.objects.get(id=id)

    # Delete the associated HerbImages objects and their image files
    for herb_image in herb.herbimages_set.all():
        # Get the path to the image file
        image_path = herb_image.images.path

        # Delete the HerbImages object
        herb_image.delete()

        # Delete the image file from storage
        default_storage.delete(image_path)

    # Delete the Herb object
    herb.delete()

    return redirect('herbal_upload')

@login_required(login_url='/login_or_register')
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


def recognition_prediction(request):
    query = request.GET.get('search', '')

    herbs = Herb.objects.filter(Q(name__icontains=query) | Q(description__icontains=query) | Q(med_use__icontains=query) | Q(med_property__icontains=query) | Q(potential_SE__icontains=query))

    if not herbs:
        messages.info(request, ':( Sorry Keyword not Found Please Try Again')

    return render(request, 'herbs.html', {'herbs': herbs})

@csrf_exempt
@login_required(login_url='/login_or_register')
def recognition(request):
    message = ""
    with open('class_mapping.json', 'r') as file:
        loaded_class_mapping_json = file.read()

    # def send_update(percentage):
    #     return JsonResponse({"percentage": percentage})
    # print(class_mapping)
    # prediction = ""
    fss = CustomFileSystemStorage()
    
    try:
        total_steps = 5

        class_mapping = json.loads(loaded_class_mapping_json)
        class_mapping = {int(key): value for key, value in class_mapping.items()}

        # send_update(20)

        image = request.FILES["image"]
        print("Name", image.file)
        _image = fss.save(image.name, image)
        path = str(settings.MEDIA_ROOT) + "/" + image.name
        # image details
        image_url = fss.url(_image)
        # Read the image
        imag = cv2.imread(path)
        img_from_ar = Image.fromarray(imag, 'RGB')
        resized_image = img_from_ar.resize((299, 299))

        test_image = np.expand_dims(resized_image, axis=0)
        test_image = test_image.astype('float32') / 255.0  # Normalize the image

        # send_update(40)

        # Load model
        model = tf.keras.models.load_model('best_model.h5')

        # send_update(60)
        
        result = model.predict(test_image)

        # send_update(80)

        predicted_class_index = np.argmax(result)
        predicted_class_probability = result[0][predicted_class_index]

        # Set your confidence threshold (e.g., 0.7)
        confidence_threshold = 0.1
        print('probability' + str(predicted_class_probability))

        top_3_probabilities = {class_mapping[i]: result[0][i]  * 100 for i in range(len(class_mapping))}
        # Sort the dictionary by values in descending order
        sorted_probabilities = sorted(top_3_probabilities.items(), key=lambda x: x[1], reverse=True)

        # Take only the top 3 entries
        class_probabilities = sorted_probabilities[:3]
        

        if predicted_class_probability < confidence_threshold:
            predicted_class_name = "Unknown"
        else:
            if predicted_class_index in class_mapping:
                predicted_class_name = class_mapping[predicted_class_index]
            else:
                predicted_class_name = "Unknown"

        print( 'index' + str(predicted_class_index))

        # send_update(100)

        return TemplateResponse(
            request,
            "recognition.html",
            {
                "probability": predicted_class_probability,
                "message": message,
                "image": image,
                "image_url": image_url,
                "prediction": predicted_class_name,
                'class_probabilities':class_probabilities,
            },
        )
    except Exception as e:
        print(f"Error: {str(e)}")

    return render(request, 'recognition.html')

@login_required(login_url='/login_or_register')
def cam_recognition(request):

    with open('class_mapping.json', 'r') as file:
        loaded_class_mapping_json = file.read()

    class_mapping = json.loads(loaded_class_mapping_json)
    class_mapping = {int(key): value for key, value in class_mapping.items()}

    if request.method == 'POST':
        image_data_uri = request.POST.get("src")
        if not image_data_uri:
            return render(request, 'cam-recognition.html', {'message': "Missing 'Image' please take an image first."})
        else:
            try:
                # Extract the base64-encoded image data from the data URI
                _, image_data = image_data_uri.split(",", 1)
                image_bytes = base64.b64decode(image_data)

                # Create a BytesIO stream from the image data
                image_stream = io.BytesIO(image_bytes)

                # Load the image with OpenCV
                imag = cv2.imdecode(np.frombuffer(image_stream.read(), np.uint8), cv2.IMREAD_COLOR)
                img_from_ar = Image.fromarray(imag, 'RGB')
                resized_image = img_from_ar.resize((299, 299))
                test_image = np.expand_dims(resized_image, axis=0)
                test_image = test_image.astype('float32') / 255.0  # Normalize the image

                # Load model
                model = tf.keras.models.load_model('best_model.h5')
                result = model.predict(test_image)

                predicted_class_index = np.argmax(result)
                predicted_class_probability = result[0][predicted_class_index]
                # Set a threshold for class probability

                confidence_threshold = 0.1  # Adjust this threshold as needed

                # Check if any class probability exceeds the threshold
                print('probability' + str(predicted_class_probability))

                top_3_probabilities = {class_mapping[i]: result[0][i]  * 100 for i in range(len(class_mapping))}
                # Sort the dictionary by values in descending order
                sorted_probabilities = sorted(top_3_probabilities.items(), key=lambda x: x[1], reverse=True)
                # Take only the top 3 entries
                class_probabilities = sorted_probabilities[:3]
                print(class_probabilities)
                if predicted_class_probability < confidence_threshold:
                    predicted_class_name = "Unknown"
                else:
                    if predicted_class_index in class_mapping:
                        predicted_class_name = class_mapping[predicted_class_index]
                    else:
                        predicted_class_name = "Unknown"

                print( 'index' + str(predicted_class_index))

                context = {
                    'class_probabilities': class_probabilities,
                    'prediction': predicted_class_name,
                    'probability': predicted_class_probability,
                }

                return render(request, "cam-recognition.html", context)

            except Exception as e:
                # Handle errors gracefully
                return HttpResponseServerError(f"Error: {str(e)}")
    return render(request, 'cam-recognition.html')


def map_endpoint(request):
    herbs = Herb.objects.all()
    stores = Store.objects.all()
    mapherbs = MapHerb.objects.all()
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

        for herbb in mapherbs:
            lat = herbb.lat
            long = herbb.long
            name = herbb.herb

        if lat is not None and long is not None:
            popup = folium.Popup(name)
            folium.CircleMarker(
                location=[lat, long],
                radius=20,
                color='yellow',
                fill=True,
                fill_color='yellow',
                fill_opacity=0.1,
                popup=popup,
            ).add_to(m)

    # Get the map HTML
    map_html = m._repr_html_()

    return HttpResponse(map_html)


# --------------------------------------------------------->
# ADMIN
# --------------------------------------------------------------------------------->
def herb_upload_stats(request):
    # Get the start of the current hour
    start_of_hour = timezone.now().replace(minute=0, second=0, microsecond=0)

    # Query to get the count of uploads per hour
    hourly_stats = MapHerb.objects.filter(timestamp__gte=start_of_hour) \
                                  .annotate(hour=TruncHour('timestamp')) \
                                  .values('hour') \
                                  .annotate(uploads_count=Count('id'))

    # Convert QuerySet to a list of dictionaries
    hourly_stats_list = list(hourly_stats)

    # Return JsonResponse
    return JsonResponse(hourly_stats_list, safe=False)

@staff_member_required(login_url=reverse_lazy('login_or_register'))
def admin_home(request):
    json_file_path = 'training_history.json'

    try:
        with open(json_file_path, 'r') as file:
            training_history_data = json.load(file)
    except FileNotFoundError:
        training_history_data = None

    herbs = Herb.objects.all()
    recent_herb = Herb.objects.latest('timestamp')
    most_liked_herb = Herb.objects.annotate(like_count=Count('favorites')).order_by('-like_count').first()
    # Most commented herb
    most_commented_herb = Herb.objects.annotate(comment_count=Count('testimonials')).order_by('-comment_count').first()
    # Calculate the start of the week (Sunday)
    start_of_week = timezone.now() - timezone.timedelta(days=timezone.now().weekday() + 1)

    # Query to get the count of uploads per week
    weekly_stats = MapHerb.objects.filter(timestamp__gte=start_of_week) \
                                .annotate(week=TruncWeek('timestamp')) \
                                .values('week') \
                                .annotate(uploads_count=Count('id'))

    start_of_day = timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # Query to get the count of uploads per day
    daily_stats = MapHerb.objects.filter(timestamp__gte=start_of_day) \
                                  .annotate(day=TruncDate('timestamp', output_field=DateField())) \
                                  .values('day', 'uploader__username') \
                                  .annotate(uploads_count=Count('id'))

    # Convert QuerySet to a list of dictionaries
    daily_stats_list = list(daily_stats)

    context = {
        'herbs':herbs,
        'most_liked_herb':most_liked_herb,
        'most_commented_herb':most_commented_herb,
        'recent_herb':recent_herb,
        'weekly_stats':weekly_stats,
        'daily_stats_list':daily_stats_list,
        'json_file_path':json_file_path,
        'training_history_data': training_history_data,
    }

    return render(request, 'Admin/index.html', context )



@staff_member_required(login_url=reverse_lazy('login_or_register'))
def dash_herb_user(request):
    mapherbs = MapHerb.objects.all()
    User = get_user_model()

    top_contributor = User.objects.annotate(num_uploads=Count('mapherb')).order_by('-num_uploads').first()
    
    users = User.objects.all().order_by('-is_staff')
    # Get a QuerySet of all session objects
    sessions = Session.objects.filter(expire_date__gte=timezone.now())
    # Create a dictionary to hold the last activity time for each user
    logged_in_user_ids = set()
        # Loop over the sessions and add the user ID to the set
    for session in sessions:
        data = session.get_decoded()
        user_id = data.get('_auth_user_id')
        if user_id:
            logged_in_user_ids.add(user_id)
    # Count the number of unique user IDs in the set
    num_logged_in_users = len(logged_in_user_ids)
    inactive_users_count = len(users) - num_logged_in_users
    user_activity = {}
    # Loop over the sessions and update the last activity time for each user
    for session in sessions:
        data = session.get_decoded()
        user_id = data.get('_auth_user_id')
        if user_id:
            user_activity[user_id] = session.expire_date - datetime.now(timezone.utc)
    # Loop over the users and update their active status
    for user in users:
        last_activity = user_activity.get(str(user.id), None)
        if last_activity:
            # User has an active session
            active_status = 'Logged In'
        else:
            # User does not have an active session
            active_status = 'Logged Off'
        # Update the active status for the user
        user.active_status = active_status
    # Create a list of user dictionaries to hold the user data
    user_data_list = []
    # Loop over the users and add their data to the user data list
    for user in users:
        user_data = {
            'user': user,
            'active_status': user.active_status
        }
        user_data_list.append(user_data)

    context = {
        'mapherbs':mapherbs,
        'user_data_list':user_data_list,
        'top_contributor':top_contributor,
        'num_logged_in_users': num_logged_in_users,
        'inactive_users_count':inactive_users_count,

    }

    return render(request,  'Admin/Users.html', context)

@staff_member_required(login_url='login_or_register')
def deleteuser(request, id):
    user = get_object_or_404(User, id=id)
    try:
        user.delete()
        messages.success(request, 'User deleted successfully')
    except Exception as e:
        messages.error(request, f'Error deleting User: {str(e)}')
    return redirect('dash_herb_user')

def toggle_user_activation(request, user_id):
    # Retrieve the user from the database
    user = get_object_or_404(User, id=user_id)

    # Toggle the 'is_active' attribute
    user.is_email_verified = not user.is_email_verified

    # Save the changes
    user.save()

    # Redirect back to the user detail page or any other page as needed
    return redirect('dash_herb_user')

@staff_member_required(login_url=reverse_lazy('login_or_register'))
def herbal_upload(request):
    herbs = Herb.objects.all()
    for herb in herbs:
        print(f"Herb: {herb.name}, Images: {herb.herbimages_set.all()}")
    context = {
        'herbs': herbs
    }

    return render(request, 'Admin/herbal-uploads.html', context)

def download_processed_data(request):
    media_root = settings.MEDIA_ROOT
    herbs_path = os.path.join(media_root, "herbs.npy")
    labels_path = os.path.join(media_root, "labels.npy")

    # Create a zip file to download multiple files
    response = HttpResponse(content_type='application/zip')
    response['Content-Disposition'] = 'attachment; filename=processed_data.zip'

    with ZipFile(response, 'w') as zip_file:
        zip_file.write(herbs_path, 'herbs.npy')
        zip_file.write(labels_path, 'labels.npy')

    return response

@csrf_exempt
@staff_member_required(login_url='login_or_register')
def process_images(request):

    media_root = settings.MEDIA_ROOT
    root_directory = os.path.join(media_root, "Datasets")

    try:
        data = []
        labels = []

        subdirectories = [d for d in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, d))]

        for label, subdirectory in enumerate(subdirectories):
            subdirectory_path = os.path.join(root_directory, subdirectory)
            image_files = os.listdir(subdirectory_path)

            class_mapping = {}
            for i, subdirectory in enumerate(subdirectories):
                class_mapping[i] = subdirectory    

            class_mapping_json = json.dumps(class_mapping, indent=4)
            with open('class_mapping.json', 'w') as file:
                file.write(class_mapping_json)

            for image_file in image_files:
                image_path = os.path.join(subdirectory_path, image_file)
                # Check if the image is readable
                imag = cv2.imread(image_path)
                if imag is None:
                    print(f"Skipping unreadable image: {image_path}")
                    continue
                
                img_from_ar = Image.fromarray(imag, 'RGB')
                resized_image = img_from_ar.resize((224, 224))
                data.append(np.array(resized_image))
                labels.append(label)

        herbs = np.array(data)
        labels = np.array(labels)
        
        # Save the processed data and labels
        np.save(os.path.join(media_root, "herbs.npy"), herbs)
        np.save(os.path.join(media_root, "labels.npy"), labels)

        return JsonResponse({'status': 'success', 'message': 'Images processed successfully.'})

    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)})


@staff_member_required(login_url=reverse_lazy('login_or_register'))
def dataset_upload(request):
    warning_message = None
    class_name = None
    images = None
    
    datasets = Datasets.objects.order_by('class_name')

    try:
        if request.method == 'POST':
            class_name = request.POST.get('class_name')
            images = request.FILES.getlist('images')

            if not images:
                print('redirect')
                warning_message = {'message':'Please Add Images'}
                return JsonResponse(warning_message)
            else:
                if class_name and images is not None:
                    # print('passed')# print(images)# print(class_name)
                    # class_folder = os.path.join(settings.MEDIA_ROOT, class_name)
                    # os.makedirs(class_folder, exist_ok=True)
                    dataset_instance = Datasets(class_name=class_name)
                    dataset_instance.save()
                 
                    for i, image in enumerate(images):
                        try:
                            dataset_instance_images = DatasetImages(class_name=dataset_instance, images=image)
                            dataset_instance_images.save()
                        except DuplicateImageError as e:
                            # Handle the duplicate image error, you can redirect or render an error message
                            warning_message = {'message':str(e)}
                            dataset_instance.delete()
                            # dataset_instance_images.delete()
                            return JsonResponse(warning_message)
                        
                    
                    print("Preparation  Success")
                    response_data = {'progress': 100, 'message': 'Upload complete'}
                    return JsonResponse(response_data)
        
    except Exception as e:
        print(f"Error during dataset upload: {str(e)}")

    return render(request, 'Admin/dataset-upload.html', {'progress_percentage': 0, 'datasets':datasets})

@staff_member_required(login_url='login_or_register')
def delete_dataset(request, id):
    com = get_object_or_404(Datasets, id=id)

    # Delete associated images
    # for image in com.images.all():
    #     try:
    #         image_path = image.images.path
    #                 # Delete the HerbImages object
    #         image.delete()
    #                 # Delete the image file from storage
    #         default_storage.delete(image_path)
    #     except ValueError:
    #         # Handle the case where image.images.path is None
    #         pass
    # Delete the Datasets object
    com.delete()

    return redirect(request.META['HTTP_REFERER'])

from .Model_Training import train_model

@staff_member_required(login_url=reverse_lazy('login_or_register'))
def model_training(request):

    try:
        if request.method == 'POST':
            label_file = request.FILES.get('label')
            herb_file = request.FILES.get('herb')

        if not label_file or not herb_file:
            # Handle the case where files are missing
            response_data = {'message': 'Please Upload Both Files'}
            return JsonResponse(response_data)

        train_model(herb_file, label_file)

        response_data = {'message': 'Training Completed'}
        return JsonResponse(response_data)
        
    except Exception as e:
        print(f"Error: {str(e)}")

    return render(request, 'Admin/model-training.html')

@staff_member_required(login_url=reverse_lazy('login_or_register'))
def model_upload(request):
    json_file_path = 'training_history.json'

    try:
        with open(json_file_path, 'r') as file:
            training_history_data = json.load(file)
    except FileNotFoundError:
        training_history_data = None    
    try:
        if request.method == 'POST':
            model_file = request.FILES.get('model')
            train_accu = request.POST.get('training_accuracy')
            val_accu = request.POST.get('validation_accuracy')
            val_loss = request.POST.get('validation_loss')
            num_classes = request.POST.get('classes')
            data_length = request.POST.get('class_length')

            # Validate if model file is provided
        if not model_file:
            return JsonResponse({'message': 'Model file is required.'})

            # Validate numeric values
        try:
            train_accu = float(train_accu)
            val_accu = float(val_accu)
            val_loss = float(val_loss)
            num_classes = int(num_classes)
            data_length = int(data_length)
        except (TypeError, ValueError):
            return JsonResponse({'message': 'Invalid numeric value provided.'})
        

        temp_model_path = f'tmp_model_{model_file.name}'
            # Save the uploaded model temporarily
        with open(temp_model_path, 'wb') as temp_model_file:
                for chunk in model_file.chunks():
                    temp_model_file.write(chunk)
            # Load the model directly from the temporary path
        model = tf.keras.models.load_model(temp_model_path)
            # Save the loaded model as an HDF5 file
        model.save('best_model.h5')
            # Remove the temporary model file
        os.remove(temp_model_path)

        data = {
            'training_accuracy': train_accu,
            'validation_accuracy': val_accu,
            'validation_loss': val_loss,
            'num_classes': num_classes,
            'data_length': data_length,
        }

        with open('training_history.json', 'w') as json_file:
                json.dump(data, json_file)

        response_data = {'progress': 100, 'message': 'Upload Completed'}
        return JsonResponse(response_data)
        
    except Exception as e:
        print(f"Error: {str(e)}")

    return render(request, 'Admin/model-upload.html', {'training_history_data':training_history_data})

@staff_member_required(login_url=reverse_lazy('login_or_register'))
def user_herbal_comments(request):
    User = get_user_model()
    users = User.objects.all()
    comments = Testimonials.objects.all()

    # Read the profanity list from a JSON file
    with open('DirtyWords.json', 'r', encoding='utf-8') as jsonfile:
        profanity_data = json.load(jsonfile)

    # Extract the list of words from the JSON data
    profanity_list = [record["word"].strip() for record in profanity_data.get("RECORDS", [])]

    for comment in comments:
        comment_text = comment.comment.lower()  # Convert to lowercase for case-insensitive comparison

        # Check if the comment contains any profanity
        if any(word in comment_text for word in profanity_list):
            # Handle the profanity, for example, mark the comment as inappropriate
            comment.is_inappropriate = True
            comment.save()

    context = {
        'users': users,
        'comments': comments,
        'profanity_list': profanity_list,
    }

    return render(request, 'Admin/user-herbal-comments.html', context)

class edit_user_upload(LoginRequiredMixin, UserPassesTestMixin, UpdateView):
    model = MapHerb
    fields = '__all__'
    success_url = reverse_lazy('interactive_map')
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
    
@staff_member_required(login_url='login_or_register')
def delete_user_upload(request, id):
    com = get_object_or_404(MapHerb, id=id)

    image_path = com.image.path

    # Delete the Herb object
    com.delete()

    # Delete the image file from storage
    default_storage.delete(image_path)

    return redirect(request.META['HTTP_REFERER'])

class edit_store(LoginRequiredMixin, UserPassesTestMixin, UpdateView):
    model = Store
    fields = '__all__'
    success_url = reverse_lazy('interactive_map')
    template_name = 'herb-cms.html'

    def test_func(self):
        return self.request.user.is_staff or self.request.user.is_superuser
    
    def post(self, request, *args, **kwargs):
        # Get the existing object
        self.object = self.get_object()

        return super().post(request, *args, **kwargs)
    
@staff_member_required(login_url='login_or_register')
def delete_store(request, id):
    com = get_object_or_404(Store, id=id)

    com.delete()

    return redirect(request.META['HTTP_REFERER'])

@staff_member_required(login_url='/login_or_register')
def add_store(request):

    form = HerbStore()
    if request.method == 'POST':
        form = HerbStore(request.POST, request.FILES)
        if form.is_valid():
            job = form.save(commit=False)
            job.save()
            return redirect('interactive_map')
        
    return render(request, 'herb-cms.html', {'form':form})

@staff_member_required(login_url=reverse_lazy('login_or_register'))
def interactive_map(request):
    
    herbs = Herb.objects.all()
    stores = Store.objects.all()
    user_uploads = MapHerb.objects.all()

    combined_list = []

    for herb in herbs:
        combined_list.append({'id': herb.id, 'type': 'herb', 'name': herb.name, 'lat': herb.lat, 'long': herb.long})

    for store in stores:
        combined_list.append({'id': store.id,'type': 'store', 'name': store.name, 'lat': store.lat, 'long': store.long})

    for upload in user_uploads:
        combined_list.append({'id': upload.id,'type': 'upload', 'name': upload.herb, 'lat': upload.lat, 'long': upload.long})

    totals = {
        'herb': len([entry for entry in combined_list if entry['type'] == 'herb']),
        'store': len([entry for entry in combined_list if entry['type'] == 'store']),
        'upload': len([entry for entry in combined_list if entry['type'] == 'upload']),
    }

    context = {
        'combined_list':combined_list,
        'totals':totals,
    }

    return render(request, 'Admin/interactive-map.html', context)

@staff_member_required(login_url=reverse_lazy('login_or_register'))
def user_map_comments(request):
    User = get_user_model()
    users = User.objects.all()

    mapcomment = MapComment.objects.all()

    with open('DirtyWords.json', 'r', encoding='utf-8') as jsonfile:
        profanity_data = json.load(jsonfile)

    # Extract the list of words from the JSON data
    profanity_list = [record["word"].strip() for record in profanity_data.get("RECORDS", [])]

    for comment in mapcomment:
        comment_text = comment.comment.lower()  # Convert to lowercase for case-insensitive comparison

        # Check if the comment contains any profanity
        if any(word in comment_text for word in profanity_list):
            # Handle the profanity, for example, mark the comment as inappropriate
            comment.is_inappropriate = True
            comment.save()

    context = {
        'users':users,
        'mapcomment':mapcomment
    }

    return render(request, 'Admin/user-map-comments.html', context)

class edit_map_comments(LoginRequiredMixin, UserPassesTestMixin, UpdateView):
    model = MapComment
    fields = '__all__'
    success_url = reverse_lazy('user_map_comments')
    template_name = 'herb-cms.html'

    def test_func(self):
        return self.request.user.is_staff or self.request.user.is_superuser
    
    def post(self, request, *args, **kwargs):
        # Get the existing object
        self.object = self.get_object()

        return super().post(request, *args, **kwargs)
    
@staff_member_required(login_url='login_or_register')
def delete_map_comments(request, id):
    com = get_object_or_404(MapComment, id=id)

    com.delete()

    return redirect(request.META['HTTP_REFERER'])

def training_history(request):
    # Assuming 'training_history.json' is in the root of your static files directory
    json_file_path = 'training_history.json'

    try:
        with open(json_file_path, 'r') as file:
            training_history_data = json.load(file)
    except FileNotFoundError:
        return JsonResponse({'error': 'File not found'}, status=404)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON format in the file'}, status=500)

    return JsonResponse(training_history_data)