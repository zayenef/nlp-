from django.shortcuts import render
from myapp.predicting.predictors import predict_traits
from myapp.utils.chart_generator import create_pie_chart, create_radar_chart, encode_image_as_base64
from myapp.utils.read_file import read_file

def classify(request):
    if request.method == 'POST':
        # Check if file input is provided
        file = request.FILES.get('file')
        if file:
            # Get the file path from the uploaded file
            file_path = handle_uploaded_file(file)

            # Call read_file function to extract text from the file
            text = read_file(file_path)
        else:
            # Retrieve the input text from the form
            text = request.POST.get('text')

        # Call the predict_traits function to perform classification
        classification_results = predict_traits(text)

        # Create classification_results_percentage for displaying probabilities with percentages in the table
        classification_results_percentage = {}
        for trait, result in classification_results.items():
            probability = float(result[0]) * 100  # Convert probability to percentage
            classification_results_percentage[trait] = f'{probability:.2f}%'

        # Generate the pie chart
        pie_chart = create_pie_chart(classification_results)
        pie_chart_base64 = encode_image_as_base64(pie_chart)

        # Generate the radar chart
        radar_chart = create_radar_chart(classification_results)
        radar_chart_base64 = encode_image_as_base64(radar_chart)

        # Pass the classification results, chart data, and percentage results to the template
        context = {
            'classification_results': classification_results,
            'classification_results_percentage': classification_results_percentage,
            'pie_chart_base64': pie_chart_base64,
            'radar_chart_base64': radar_chart_base64
        }
        return render(request, 'classification_result.html', context)

    return render(request, 'classification.html')

