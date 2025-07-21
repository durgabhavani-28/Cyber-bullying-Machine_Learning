from django.contrib import admin
from django.urls import path
from classifier import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),
    path('about/', views.render_about, name='render_about'),
    path('classify/', views.classify_text, name='classify_text'),
    path('result/', views.render_result, name='render_result'),
    path('report/', views.report_comment, name='report_comment'),

]