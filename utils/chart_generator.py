import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import base64
import io


def create_pie_chart(data):
    traits = list(data.keys())
    probabilities = [float(val[0]) for val in data.values()]
    colors = ['green' if p >= 0.5 else 'red' for p in probabilities]
    explode = [0.1] * len(traits)  # Add some separation between slices
    fig, ax = plt.subplots(facecolor='none', edgecolor='none')
    ax.pie(probabilities, labels=traits, colors=colors, explode=explode, autopct='%1.1f%%', startangle=90)
    ax.set_title('Classification Results', color='white')  # Set title color to white
    ax.set_xlabel('Traits', color='white')  # Set x-label color to white
    ax.set_ylabel('Probabilities', color='white')  # Set y-label color to white

    # Set all text elements to white
    for text in ax.texts:
        text.set_color('white')

    # Set legend properties
    legend = ax.legend(loc='center right', bbox_to_anchor=(1.3, 0.5), framealpha=0.7, edgecolor='none')
    for text in legend.get_texts():
        text.set_color('white')
        text.set_fontsize(7)  # Set font size of legend labels

    # Return the chart object instead of showing it
    return fig





def create_radar_chart(data):
    traits = list(data.keys())
    probabilities = [float(val[0]) for val in data.values()]
    angles = np.linspace(0, 2 * np.pi, len(traits), endpoint=False).tolist()
    probabilities += probabilities[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True), facecolor='none', edgecolor='none')
    ax.plot(angles, probabilities, color='b')
    ax.fill(angles, probabilities, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(traits, color='white')  # Set x-tick labels color to white
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_title('Classification Results', color='white')  # Set title color to white
    ax.set_xlabel('Traits', color='white')  # Set x-label color to white
    ax.set_ylabel('Probabilities', color='white')  # Set y-label color to white
    # Return the chart object instead of showing it
    return fig




def encode_image_as_base64(image):
    buffer = io.BytesIO()
    image.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    return image_base64
