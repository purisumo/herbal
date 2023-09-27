from django.shortcuts import render, HttpResponse, redirect, get_object_or_404
from django.db.models import Q
from .models import Herb, Favorite
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.views.generic import CreateView, UpdateView
from django.core.files.storage import default_storage
from django.urls import reverse_lazy, reverse
from .forms import HerbForm
# Create your views here.

def home(request):
    
    return render(request, 'core/home.html')

def herbs(request):

    return render(request, 'herbs.html')

def illness(request):

    return render(request, 'illness.html')

def favourite(request):

    return render(request, 'favourit.html')

def recognition(request):

    return render(request, 'recognition.html')

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

    return render(request, 'search.html', {'herbs': herbs})

