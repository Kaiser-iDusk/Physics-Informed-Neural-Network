from django.contrib import admin
from django.urls import path
from PiNN import views

urlpatterns = [
    # path('admin/', admin.site.urls),
    path("", views.index, name="home"),
    path("process", views.process, name="process")
]