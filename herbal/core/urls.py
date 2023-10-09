from django.urls import path, include
from .views import *
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('Herbs/', views.herbs, name='herbs'),
    path('Image_search/', views.image_search, name='image-search'),
    path('Herbal_Map/', views.herbal_map, name='herbal-map'),
    path('Herbal_Map_/<str:name>', views.herbal_map_inter, name='herbal-map-interaction'),
    path('Favourite/', views.favourite, name='favourite'),
    path('Recognition/', views.recognition, name='recognition'),
    path('Cam-recognition/', views.cam_recognition, name='cam-recognition'),
    path('add/', views.add, name='add'),
    path('edit/<str:pk>', edit.as_view(), name='edit'),
    path('toggle_favorite/<int:herb_id>/', views.toggle_favorite, name='toggle_favorite'),
    path('map/', views.map_endpoint, name='map_endpoint'),
    path('search/', views.search, name='search'),
]