import streamlit as st
import torch
from transformers import CLIPVisionModel, CLIPProcessor
from PIL import Image
import io
import base64
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import umap
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, LassoSelectTool
from bokeh.embed import components
import plotly.express as px

# Load vision model and processor
model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def preprocess_image(image_file):
    image_file.seek(0)  
    image = Image.open(image_file)
    inputs = processor(images=image, return_tensors="pt")
    return inputs

def extract_embeddings(image_file):
    inputs = preprocess_image(image_file)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]  
    return embeddings.squeeze().numpy()

def visualize_embeddings(embeddings, method="pca"):
    embeddings_array = np.array(embeddings)
    if method == "pca":
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings_array)
    elif method == "tsne":
        tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings_array) - 1))
        reduced_embeddings = tsne.fit_transform(embeddings_array)
    elif method == "umap":
        umapper = umap.UMAP(random_state=42)
        reduced_embeddings = umapper.fit_transform(embeddings_array)

    return reduced_embeddings

# Upload images
st.set_page_config(layout="wide")
st.title("CLIP Image Embeddings Visualizer")
with st.sidebar:
    image_files = st.file_uploader("Upload images", type=["jpg", "png","bmp"], accept_multiple_files=True)
    reducer = st.selectbox("Dimensionality Reduction",options=['pca','tsne','umap'])
    runner = st.button("Generate!")

if image_files and runner:
    # Preprocess and extract embeddings
    embeddings = []
    image_paths = []
    image_data = []
    for image_file in image_files:
        image_paths.append(image_file.name)
        embeddings.append(extract_embeddings(image_file))
        image_data.append(image_file)

    # Visualize embeddings
    reduced_embeddings = visualize_embeddings(embeddings, method=reducer)

    # Create DataFrame for Bokeh
    df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'image': image_paths
    })

    # Create Plotly figure
    fig = px.scatter(df, x='x', y='y')
    fig.update_traces(textposition='top center', hoverinfo='all')
    fig.update_layout(title="CLIP Image Embeddings", width=800, height=600)

    # Display Plotly figure
    # col1, col2 = st.columns([2, 1])
    # with col1:
    st.markdown("# Embeddings")
    st.plotly_chart(fig)

    # # Store selected points
    # if "selected_points