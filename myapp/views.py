from django.shortcuts import render
from django.conf import settings
import os


def home(request):
    return render(request, 'home.html')


def upload_image(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['image']

        # media 디렉토리가 없으면 생성
        if not os.path.exists(settings.MEDIA_ROOT):
            os.makedirs(settings.MEDIA_ROOT)

        file_path = os.path.join(settings.MEDIA_ROOT, uploaded_file.name)

        with open(file_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        # 이미지 경로를 템플릿으로 전달
        context = {
            'image_url': settings.MEDIA_URL + uploaded_file.name
        }
        return render(request, 'upload.html', context)

    return render(request, 'upload.html')
