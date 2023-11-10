from django.urls import path, include
from .views import *
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('dashboard/', views.admin_home, name='dashboard'),
    path('dashboard/herb_user)', views.dash_herb_user, name='dash_herb_user'),
    path('search/', views.search, name='search'),
    path('Herbs/', views.herbs, name='herbs'),
    path('Image_search/', views.image_search, name='image-search'),
    path('Herbal_Map/', views.herbal_map, name='herbal-map'),
    path('update_user_data/', views.update_user_data, name='update_user_data'),
    path('Herbal_Map_<int:id>/', views.herbal_map_inter, name='herbal-map-interaction-id'),
    path('Herbal_Map_<str:herb>/', views.herbal_map_inter, name='herbal-map-interaction-name'),
    path('Favourite/', views.favourite, name='favourite'),
    path('Recognition/', views.recognition, name='recognition'),
    path('Cam-recognition/', views.cam_recognition, name='cam-recognition'),
    path('add/', views.add, name='add'),
    path('edit/<str:pk>', edit.as_view(), name='edit'),
    path('delete/<int:id>', views.delete, name='delete'),
    path('deletecomment/<int:id>', views.deletecomment, name='deletecomment'),
    path('deletetestimony/<int:id>', views.deletetesti, name='deletetestimony'),
    path('toggle_favorite/<int:herb_id>/', views.toggle_favorite, name='toggle_favorite'),
    path('map/', views.map_endpoint, name='map_endpoint'),
]