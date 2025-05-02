from django.shortcuts import render
import pandas as pd
from django.shortcuts import render
from ml.model_utils import predict_cluster  # Use absolute import

def home(request):
    result_df = None
    if request.method == 'POST' and request.FILES.get('csv_file'):
        csv_file = request.FILES['csv_file']
        try:
            df = pd.read_csv(csv_file)
            result_df = predict_cluster(df)

            # Convert DataFrame to list of dicts for template rendering
            result_data = result_df.to_dict(orient='records')

        except Exception as e:
            return render(request, 'home.html', {'error': str(e)})

        return render(request, 'home.html', {'result_data': result_data})

    return render(request, 'home.html')

# Create your views here.

def about(request):
    return render(request, 'about.html')

def contact(request):
    return render(request, 'contact.html')