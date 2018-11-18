from django.contrib import admin
from django.conf.urls import url, include
from . import views
app_name = 'project'
urlpatterns = [
    url('^dialog/(?P<usr_id>[0-9]+)$', views.edit_action, name='dialog'),
    url('^dialog/$', views.init_action, name='init')
]
