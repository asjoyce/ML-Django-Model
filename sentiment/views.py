from django.shortcuts import render

# Create your views here.
import joblib

# Load the trained model
model = joblib.load('sentiment_model.pkl')

def index(request):
    return render(request, 'index.html')

def predict_sentiment(request):
    if request.method == 'POST':
        input_text = request.POST['text']
        prediction = model.predict([input_text])[0]
        return render(request, 'index.html', {'input_text': input_text, 'result': prediction})
    return render(request, 'index.html')
