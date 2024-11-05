import dash
from dash import dcc
from dash import html
import dash_auth
from dash.dependencies import Input, Output, State
import plotly.express as px
import torch
from transformers import CLIPModel, CLIPProcessor, SamModel, SamProcessor, AutoImageProcessor, AutoModel, ResNetModel
from sklearn.manifold import TSNE
import pandas as pd
from dash.exceptions import PreventUpdate
import base64
from PIL import Image
import io
import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

VALID_USERNAME_PASSWORD_PAIRS = {
    'user': 'pass'
}
# Model loading
def load_model(embedding_model):
    """Load CLIP model and processor"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if embedding_model == 'clip':
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").vision_model.to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    elif embedding_model == 'sam':
        model = SamModel.from_pretrained("facebook/sam-vit-large").to(device)
        processor = SamProcessor.from_pretrained("facebook/sam-vit-large")
    elif embedding_model == 'resnet':
        # model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
        # processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        model = AutoModel.from_pretrained("microsoft/resnet-50").to(device)
        processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

    return model, processor, device

# Image processing
def decode_image(encoded_image):
    """Decode base64-encoded image"""
    decoded_image = base64.b64decode(encoded_image.split(",")[1])
    return Image.open(io.BytesIO(decoded_image))

def get_app_layout():
    """Define app layout"""
    return html.Div([
        html.H1("Image Embeddings Visualizer", style={"textAlign": "center", "padding": "20px","backgroundColor": "#f9f9f9"}),
        
        html.Div(
            [
                html.Div([
                    html.H2("Image Upload and Settings",style={"textAlign": "center"}),
                    dcc.Upload(
                        id="upload-images",
                        children=[
                            html.Div(
                                "Upload Images",
                                style={"font-size": "14px", "color": "#666"}
                            )
                        ],
                        multiple=True,
                        style={
                            "width": "100%",
                            "height": "40px",
                            "lineHeight": "40px",
                            "borderWidth": "1px",
                            "borderStyle": "dashed",
                            "borderRadius": "5px",
                            "textAlign": "center",
                            "marginBottom": "10px"
                        }
                    ),
                    html.Div(
                        [
                            html.P("Reduction Method", style={"textAlign": "center","font-size": "14px", "marginBottom": "5px"}),
                            dcc.Dropdown(
                                id="reduction-method-dropdown",
                                options=[
                                    {"label": "UMAP", "value": "umap"},
                                    {"label": "t-SNE", "value": "tsne"},
                                    {"label": "PCA", "value": "pca"}
                                ],
                                value="umap",
                                style={"width": "100%", "height": "30px", "fontSize": "14px"}
                            )
                        ],
                        style={"padding": "15px", "border": "1px solid #ddd", "borderRadius": "5px"}
                    ),
                    html.Div(
                        [
                            html.P("Embedding Model", style={"textAlign": "center","font-size": "14px", "marginBottom": "5px"}),
                            dcc.Dropdown(
                                id="embedding-model-dropdown",
                                options=[
                                    {"label": "CLIP", "value": "clip"},
                                    {"label": "Segment Anything", "value": "sam"},
                                    {"label": "ResNet50", "value": "resnet"}
                                ],
                                value="clip",
                                style={"width": "100%", "height": "30px", "fontSize": "14px"}
                            )
                        ],
                        style={"padding": "15px", "border": "1px solid #ddd", "borderRadius": "5px"}
                    ),
                    html.Div(
                        [
                            html.Button('Generate!',
                                        id='generator',
                                        n_clicks=0,
                                        style={"width": "100%", "height": "30px", "fontSize": "14px"})
                        ],
                        style={"padding": "15px", "border": "1px solid #ddd", "borderRadius": "5px"}
                    )
                ], style={"width": "15%", "display": "inline-block", "vertical-align": "top", "padding": "20px", "height": "80vh", "backgroundColor": "#f9f9f9", "borderRight": "1px solid #ddd"}),
                html.Div([
                    html.H2("Embeddings", style={"textAlign": "center"}),
                    dcc.Graph(id="embeddings-graph",style={'height':'100%','width':'100%'})
                ], style={"width": "60%", "display": "inline-block", "vertical-align": "top", "padding": "0px", "height": "80vh"}),
                
                html.Div([
                    html.H2("Selected Images", style={"textAlign": "center"}),
                    html.Div(id="thumbnails")
                ], style={"width": "25%", "display": "inline-block", "vertical-align": "top", "padding": "20px", "height": "80vh", "overflowY": "scroll", "backgroundColor": "#f9f9f9", "borderLeft": "1px solid #ddd"})
            ],
            style={"display": "flex", "height": "90vh"}
        ),
        
        dcc.Store(id="image-store")
    ],style={'height':'100vh'})

# Callbacks
def update_graph(n_clicks, uploaded_images, reduction_method, embedding_model):
    """Update graph with image embeddings"""
    if uploaded_images is None:
        raise PreventUpdate

    model, processor, device = load_model(embedding_model)

    # Process uploaded images
    images = []
    for encoded_image in uploaded_images:
        image = decode_image(encoded_image)
        inputs = processor(images=image, return_tensors="pt").to(device)
        images.append(inputs)

    # Compute image embeddings
    embeddings = []
    with torch.no_grad():
        for image in images:
            if embedding_model == "clip":
                outputs = model(**image)
                embedding = outputs.last_hidden_state[:, 0, :]
                print(embedding.shape)
                embeddings.append(embedding.squeeze().cpu().numpy())
            elif embedding_model == "sam":
                embedding = model.get_image_embeddings(image.pixel_values)[:, 0, :]
                embedding_reshape = embedding.squeeze().flatten().cpu().numpy()
                embeddings.append(embedding_reshape)
            elif embedding_model == "resnet":
                outputs = model(**image)
                embedding = outputs.last_hidden_state
                print(embedding.shape)
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(embedding, (1, 1)).squeeze().cpu().numpy())
        

    embeddings_array = normalize(np.array(embeddings))
    print(embeddings_array.shape)    # Reduce dimensionality
    if reduction_method == "tsne":
        tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings_array) - 1))
        reduced_embeddings = tsne.fit_transform(embeddings_array)
    elif reduction_method == "pca":
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings_array)

    elif reduction_method == "umap":
        umapper = umap.UMAP(random_state=42,n_neighbors=15,min_dist=0.0,metric='cosine')
        reduced_embeddings = umapper.fit_transform(embeddings_array)

    # Create figure
    fig = px.scatter(x=reduced_embeddings[:, 0],
                     y=reduced_embeddings[:, 1],
                     hover_name=[f"Image {i}" for i in range(len(uploaded_images))])

    fig.update_layout(
                      xaxis_title="Dimension 1",
                      yaxis_title="Dimension 2",
                      )

    return fig

def store_images(uploaded_images):
    """Store uploaded images"""
    print(len(uploaded_images))
    if uploaded_images is None:
        raise PreventUpdate

    images = []

    for encoded_image in uploaded_images:
        image = decode_image(encoded_image)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode()
        images.append(encoded_image)  # Store base64 encoded string
    
    print(f'{len(images)} Images Stored!')


    return images


def display_thumbnails(selected_data, images):
    """Display thumbnails of selected images"""
    if selected_data is None or images is None:
        raise PreventUpdate
    base_width = 200
    thumbnail_images = []
    for point in selected_data["points"]:
        image_index = point["pointIndex"]
        encoded_image = images[image_index]
        # Remove data:image/png;base64, prefix if present
        if encoded_image.startswith('data:image'):
            encoded_image = encoded_image.split(",")[1]
        
        image = Image.open(io.BytesIO(base64.b64decode(encoded_image)))
        wpercent = (base_width / float(image.size[0]))
        hsize = int((float(image.size[1]) * float(wpercent)))
        thumbnail_image = image.resize((base_width, hsize), Image.Resampling.LANCZOS)
        buffered = io.BytesIO()
        thumbnail_image.save(buffered, format="PNG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode()
        thumbnail_images.append(html.Img(src=f"data:image/png;base64,{encoded_image}"))
    print(len(thumbnail_images))
    return thumbnail_images

# App definition
def create_app():
    app = dash.Dash(__name__)
    dash_auth.BasicAuth(app,VALID_USERNAME_PASSWORD_PAIRS)
    app.layout = get_app_layout()

    app.callback(
        Output("embeddings-graph", "figure"),
        Input("generator","n_clicks"),
        [State("upload-images", "contents"),
         State("reduction-method-dropdown", "value"),
         State("embedding-model-dropdown", "value")
         ],
         
    )(update_graph)

    app.callback(
        Output("image-store", "data"),
        [Input("upload-images", "contents")]
    )(store_images)

    app.callback(
        Output("thumbnails", "children"),
        [Input("embeddings-graph", "selectedData"),
         Input("image-store", "data")]
    )(display_thumbnails)

    return app

# Run app
if __name__ == "__main__":
    app = create_app()
    app.run_server(debug=False)