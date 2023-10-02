import os
import cv2

from PIL import Image
import numpy as np
import tensorflow as tf

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
            0: "alpulka",
            1: "ampalaya",
            2: "bawang",
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
    
    
from django.core.files import File
from django.core.files.temp import NamedTemporaryFile
from urllib.request import urlopen  # Import urlopen from urllib.request

def cam_recognition(request):
    context = dict()
    
    if request.method == 'POST':
        image_url = request.POST["src"]  # src is the name of the input attribute in your HTML file

        try:
            # Open the URL and download the image
            response = urlopen(image_url)
            image = NamedTemporaryFile(delete=True)
            image.write(response.read())
            image.flush()

            # Create a File object from the downloaded image
            image_file = File(image)

            # Set a name for the image file (adjust as needed)
            name = "downloaded_image.jpg"
            image_file.name = name

            # Save the image to your model
            obj = Image.objects.create(image=image_file)  # Assuming you have an Image model defined
            obj.save()

            context["path"] = obj.image.url  # URL to the image stored on your server/local device
        except Exception as e:
            # Handle any exceptions that may occur during the image download or processing
            context["error_message"] = f"Error: {str(e)}"
            return render(request, 'cam-recognition.html', context=context)
        
        return redirect('/')  # Redirect to a specific URL after processing
        
    return render(request, 'cam-recognition.html', context=context)
