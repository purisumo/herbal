from django.shortcuts import render, get_object_or_404, redirect
from django.conf import settings
from validate_email import validate_email
from django.contrib.sites.shortcuts import get_current_site
from django.template.loader import render_to_string
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes, force_str, DjangoUnicodeDecodeError
from .utils import generate_token
from django.core.mail import EmailMessage
import threading
from django.http import HttpResponse
from django.contrib import messages
from django.contrib.auth.forms import UserCreationForm
from django.core.mail import send_mail
from django.template.loader import render_to_string
from django.utils.html import strip_tags
from django.urls import reverse
from .models import User
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import PasswordChangeForm
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth import authenticate, login, logout

class EmailThread(threading.Thread):
    def __init__(self, email):
        self.email = email
        threading.Thread.__init__(self)
    def run(self):
        self.email.send()

def login_or_register(request):
    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'login':
            username = request.POST['username']
            password = request.POST['password']
            user = authenticate(username=username, password=password)

            if user.is_superuser:
                login(request, user)
                return redirect('dashboard')
            
            if user is not None:
                if not user.is_email_verified:
                    messages.error(request, 'Email not verified')
                    return render(request, 'registration/login_or_register.html')
            
                login(request, user)
                
                return redirect('home')
            else:
                messages.info(request, 'Invalid Credentials!')
                return redirect('login_or_register')
        elif action == 'register':
            username = request.POST['username']
            email = request.POST['email']
            password = request.POST['password']
            password2 = request.POST['password2']

            if password == password2:
                if User.objects.filter(email=email).exists():
                    messages.info(request, 'Email already used')
                    return redirect('login_or_register')
                elif User.objects.filter(username=username).exists():
                    messages.info(request, 'Username already used')
                    return redirect('login_or_register')
                else:
                    user = User.objects.create_user(username=username, email=email, password=password)
                    user.save()
                    send_activation_email(user, request)

                    messages.success(request, 'Check your email to validate your account')
                    # login(request, user)
                    return redirect('login_or_register')
            else:
                messages.info(request, 'Passwords do not match')
                return redirect('login_or_register')

    return render(request, 'registration/login_or_register.html')

# send_activation_email function
def send_activation_email(user, request):
    current_site = get_current_site(request)
    email_subject = 'Activate your account'
    uid64 = urlsafe_base64_encode(force_bytes(user.pk))
    token = generate_token.make_token(user)
    
    email_body = render_to_string('registration/activate.html', {
        'user': user,
        'request': request,
        'uid64': uid64,
        'token': token
    })

    email = EmailMessage(
        subject=email_subject,
        body=email_body,
        from_email=settings.EMAIL_HOST_USER,
        to=[user.email]
    )
    email.content_subtype = 'html'
    EmailThread(email).start()


def activate_user(request, uid64, token):
    try:
        uid = force_str(urlsafe_base64_decode(uid64))
        user = User.objects.get(pk=uid)
    except Exception as e:
        user = None

    if user and generate_token.check_token(user, token):
        user.is_email_verified = True
        user.save()

        messages.success(request, 'Email Verified')
        return redirect('login_or_register')

    return render(request, 'registration/activate-failed.html', {'user': user})

def logout_view(request):
    logout(request)
    return redirect('home')

# def login_view(request):

#     if request.method == 'POST':
#         username = request.POST['username']
#         password = request.POST['password']

#         user = authenticate(username=username, password=password)

#         if user is not None and user.is_superuser:
#             login(request, user)
#             return redirect('dashboard')
#         elif user is not None:
#             login(request, user)
#             return redirect('home')
#         else:
#             messages.info(request, 'Invalid Credentials!')
#             return redirect('login')
#     else:
        
#         return render(request, 'registration/login.html')


# def register(request):
#     if request.method == 'POST':
#         username = request.POST['username']
#         email = request.POST['email']
#         password = request.POST['password']
#         password2 = request.POST['password2']

#         if password == password2:
#             if User.objects.filter(email=email).exists():
#                 messages.info(request, 'Email already used')
#                 return redirect('register')
#             elif User.objects.filter(username=username).exists():
#                 messages.info(request, 'Username already used')
#                 return redirect('register')

#             else:
#                 user = User.objects.create_user(username=username, email=email, password=password)  
#                 user.save(); 
#                 return redirect('login')
#         else:
#             messages.info(request, 'Password is not the same')
#             return redirect('register')
#     else:
#         return render(request, 'registration/register.html')

@login_required
def change_password(request):
    if request.method == 'POST':
        form = PasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            user = form.save()
            update_session_auth_hash(request, user)
            messages.success(request, 'Your password was successfully updated!')
            return redirect('home')
        else:
            messages.error(request, 'Please correct the error below.')
    else:
        form = PasswordChangeForm(request.user)
    return render(request, 'registration/change_pass.html', {'form': form})

# def registercustom(request):
#     if request.method == "GET":
#         return render(
#             request, "users/register.html",
#             {"form": CustomUserCreationForm}
#         )
#     elif request.method == "POST":
#         form = CustomUserCreationForm(request.POST)
#         if form.is_valid():
#             user = form.save()
#             login(request, user)
#             return redirect(reverse("dashboard"))