from django.shortcuts import render
import pandas as pd
from django.shortcuts import render
from ml.model_utils import predict_cluster  # Use absolute import
import csv
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json


def home(request):
    result_df = None
    if request.method == 'POST' and request.FILES.get('csv_file'):
        csv_file = request.FILES['csv_file']
        try:
            df = pd.read_csv(csv_file)
            result_df = predict_cluster(df)
            # After clustered_df = predict_cluster(df)
            clustered_df = result_df
            col = 'cluster' if 'cluster' in clustered_df.columns else 'Cluster'
            cluster_counts = clustered_df[col].value_counts().sort_index().to_dict()


            # Convert DataFrame to list of dicts for template rendering
            result_data = result_df.to_dict(orient='records')

            # SAVE INTO SESSION:
            request.session['result_data'] = result_data
            request.session.modified = True

        except Exception as e:
            return render(request, 'home.html', {'error': str(e)})

        return render(request, 'home.html', {
            'result_data': result_data,
            'cluster_counts_json': json.dumps(cluster_counts)
        })

    return render(request, 'home.html')

# Create your views here.

def about(request):
    return render(request, 'about.html')

def contact(request):
    return render(request, 'contact.html')


@csrf_exempt
def export_csv(request):
    if request.method == 'POST':
        result_data = request.session.get('result_data')
        if not result_data:
            return HttpResponse("No data to export.", status=400)

        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="study_groups.csv"'

        writer = csv.writer(response)
        # Write header
        writer.writerow(result_data[0].keys())
        # Write data rows
        for row in result_data:
            writer.writerow(row.values())

        return response
    return HttpResponse("Invalid request method.", status=405)