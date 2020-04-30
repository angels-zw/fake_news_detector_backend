
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework import status
import json
from django.core.exceptions import ObjectDoesNotExist
from rest_framework import status
from rest_framework import serializers
from rest_framework.decorators import api_view
from django import forms
from django.db import models
from .ml.custom_search import CustomSearch
from .ml.news_classifier import CnnClassifier
import pandas as pd
import requests
from django.core.exceptions import ObjectDoesNotExist




class NewsModel(models.Model):

    url = models.CharField(max_length=255, default='ACbcad883c9c3e9d9913a715557dddff99')
    name = models.CharField(max_length=255, default='abd4d45dd57dd79b86dd51df2e2a6cd5')
    stance = models.CharField(max_length=255, default='+15006660005')

    
class NewsSerializer(serializers.ModelSerializer):
    class Meta:
        model = NewsModel
        fields = ['url','name','stance']


def run_model(input_data):
    classifier = CnnClassifier()
    stance =classifier.predict(input_data)
    for index, row in input_data.iterrows():
        row['stance']=stance[index]
    return input_data


@api_view(["POST"])
def fact_checker(request):
    text = json.loads(request.body)
    articlebody =text['text']
    try:
        custom_search = CustomSearch()
        input_data = custom_search.custom_search(articlebody)
        data = run_model(input_data)
        data =data.to_json(orient='values')
        return JsonResponse({'results': data}, safe=False, status=status.HTTP_201_CREATED)
    except ObjectDoesNotExist as e:
        return JsonResponse({'error': str(e)}, safe=False, status=status.HTTP_404_NOT_FOUND)
    except Exception:
        return JsonResponse({'error': 'Something terrible went wrong'}, safe=False, status=status.HTTP_500_INTERNAL_SERVER_ERROR)








    





