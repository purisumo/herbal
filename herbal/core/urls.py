from django.urls import path, include
from .views import *
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('dashboard/', views.admin_home, name='dashboard'),
    path('herb_upload_stats/', views.herb_upload_stats, name='herb_upload_stats'),

    path('dashboard/herb_user', views.dash_herb_user, name='dash_herb_user'),
    path('dashboard/deleteuser/<int:id>', views.deleteuser, name='deleteuser'),
    path('toggle_user_activation/<int:user_id>/', views.toggle_user_activation, name='toggle_user_activation'),

    path('dashboard/herbal_upload', views.herbal_upload, name='herbal_upload'),

    path('dashboard/dataset_upload', views.dataset_upload, name='dataset_upload'),
    path('dashboard/dataset_upload/process_images', views.process_images, name='process_images'),
    path('download-processed-data/', download_processed_data, name='download_processed_data'),
    path('dashboard/delete_dataset/<int:id>', views.delete_dataset, name='delete_dataset'),
    
    path('dashboard/interactive_map', views.interactive_map, name='interactive_map'),
    path('dashboard/interactive_map/delete_user_upload/<int:id>', views.delete_user_upload, name='delete_user_upload'),
    path('dashboard/interactive_map/edit_user_upload/<str:pk>', edit_user_upload.as_view(), name='edit_user_upload'),

    path('dashboard/interactive_map/add_store', views.add_store, name='add_store'),
    path('dashboard/interactive_map/delete_store/<int:id>', views.delete_store, name='delete_store'),
    path('dashboard/interactive_map/edit_store/<str:pk>', edit_store.as_view(), name='edit_store'),

    path('dashboard/model_training', views.model_training, name='model_training'),
    
    path('dashboard/model_upload', views.model_upload, name='model_upload'),
    path('dashboard/user_herbal_comments', views.user_herbal_comments, name='user_herbal_comments'),

    path('dashboard/user_map_comments', views.user_map_comments, name='user_map_comments'),
    path('dashboard/interactive_map/delete_map_comments/<int:id>', views.delete_map_comments, name='delete_map_comments'),
    path('dashboard/interactive_map/edit_map_comments/<str:pk>', edit_map_comments.as_view(), name='edit_map_comments'),

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
    # Herbs
    path('add/', views.add, name='add'),
    path('edit/<str:pk>', views.edit, name='edit'),
    path('delete/<int:id>', views.delete, name='delete'),
    # Herbs end
    path('deletecomment/<int:id>', views.deletecomment, name='deletecomment'),
    path('deletetestimony/<int:id>', views.deletetesti, name='deletetestimony'),

    path('toggle_favorite/<int:herb_id>/', views.toggle_favorite, name='toggle_favorite'),
    path('user_edit_upload/<str:pk>', user_edit_upload.as_view(), name='user_edit_upload'),
    path('map/', views.map_endpoint, name='map_endpoint'),
    path('training_history/', views.training_history, name='training_history'),
]