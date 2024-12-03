# Django's ORM model for User authentication
from django.db import models

class User(models.Model):
    username = models.CharField(max_length=100)
    password = models.CharField(max_length=100)
    email = models.EmailField(primary_key=True)

    def __str__(self):
        return self.email

from django.db import models

# Example model for MongoDB
class ISLGif(models.Model):
    phrase = models.CharField(max_length=100, unique=True)
    gif_url = models.CharField(max_length=255)

    def __str__(self):
        return self.phrase

class SignLanguageLetter(models.Model):
    letter = models.CharField(max_length=1, unique=True)
    image_url = models.CharField(max_length=255)

    def __str__(self):
        return self.letter
