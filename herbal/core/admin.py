from django.contrib import admin
from .models import Herb, Favorite, Store
# Register your models here.

admin.site.register(Herb)
admin.site.register(Store)
admin.site.register(Favorite)