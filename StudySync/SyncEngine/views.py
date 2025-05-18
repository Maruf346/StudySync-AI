from django.shortcuts import render
import pandas as pd
from ml.model_utils import predict_cluster
import csv
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
import traceback

def home(request):
    result_data = None
    cluster_counts_json = None
    error = None

    if request.method == 'POST' and request.FILES.get('csv_file'):
        print("🛠️  home() POST hit — files:", request.FILES.keys())
        csv_file = request.FILES['csv_file']
        try:
            df = pd.read_csv(csv_file)
            clustered_df = predict_cluster(df)
            col = 'cluster' if 'cluster' in clustered_df.columns else 'Cluster'
            clustered_df = clustered_df.sort_values(by=col).reset_index(drop=True)

            counts = clustered_df[col].value_counts().sort_index().to_dict()
            cluster_counts_json = json.dumps(counts)

            result_data = clustered_df.to_dict(orient='records')
            print("🛠️  result_data has", len(result_data), "rows")

            # SAVE INTO SESSION:
            request.session['result_data'] = result_data
            request.session.modified = True

        except Exception as e:
            # Log full traceback
            traceback.print_exc()
            error = str(e)
    
    # Whether GET or POST (successful or errored), render the template:
    return render(request, 'home.html', {
        'result_data': result_data,
        'cluster_counts_json': cluster_counts_json,
        'error': error,
    })

@csrf_exempt
def export_csv(request):
    if request.method == 'POST':
        result_data = request.session.get('result_data')
        if not result_data:
            return HttpResponse("No data to export.", status=400)
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="study_groups.csv"'
        writer = csv.writer(response)
        writer.writerow(result_data[0].keys())
        for row in result_data:
            writer.writerow(row.values())
        return response
    return HttpResponse("Invalid request method.", status=405)

def about(request):
    return render(request, 'about.html')

def contact(request):
    return render(request, 'contact.html')

