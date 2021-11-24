from django.urls import path
from . import views

urlpatterns = [
    path('', views.index,name="FaceMaskChecker_index"),
    path('video', views.video,name="video"),
]
