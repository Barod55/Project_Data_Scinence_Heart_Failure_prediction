# 1. استخدام نسخة بايثون خفيفة مبنية على لينكس
FROM python:3.12-slim

# 2. منع إنشاء ملفات __pycache__ وإظهار الـ Logs فوراً في Docker و Render
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3. تحديد فولدر الشغل جوه الـ Container
WORKDIR /app

# 4. تثبيت مكتبات التشغيل الخفيفة فقط
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# 5. تدريب موديل صاحبك (Stacking) وحفظه في models/model.joblib بنفس إصدارات الـ Container
COPY data/heart.csv ./data/heart.csv
COPY train.py .
RUN python train.py

# 6. نسخ ملفات الـ API والواجهة
COPY app.py features.py HART.html ./

# 7. الإعلان عن البورت الافتراضي
EXPOSE 8000

# 8. تشغيل FastAPI في وضع الإنتاج مع دعم بورت Render الديناميكي وبورت 8000 محلياً
CMD ["sh", "-c", "exec fastapi run app.py --host 0.0.0.0 --port ${PORT:-8000}"]