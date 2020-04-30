from django.urls import include, path
from . import views

urlpatterns = [
  
  path('fact_checker', views.fact_checker),
 
]