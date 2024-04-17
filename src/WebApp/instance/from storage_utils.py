from google.cloud import storage

def get_image_url(image_path):
    # استخدم Firebase Storage Client SDK للحصول على الرابط
    client = storage.Client()
    bucket = client.get_bucket('rickman-de167.appspot.com')  # استبدل بـ اسم الدلو الخاص بك
    blob = bucket.blob(image_path)
    url = blob.generate_signed_url(expiration=300, method='GET')  # اختصار الصلاحية إلى 5 دقائق
    return url
