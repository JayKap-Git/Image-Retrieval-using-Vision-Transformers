"""
Minimalistic UI for Image Modeling System
A Streamlit-based interface for image retrieval using trained models.
"""

import streamlit as st
import os
import sys
import torch
import pandas as pd
import numpy as np
from PIL import Image
import json
from typing import List, Tuple, Dict
import tempfile
import shutil
import io

# Add the current directory to Python path to import the pipeline
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from the main pipeline
from Milestone2_Image_Understanding_Pipeline import (
    Vocab, MultiTaskNet, TextEncoder, RetrievalHead, 
    make_transforms, humanize_label, TrainConfig
)

# Page configuration
st.set_page_config(
    page_title="Image Modeling System - Our Dataset",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .model-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .result-card {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #fafafa;
    }
    .score-badge {
        background-color: #1f77b4;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_vocab_and_data():
    """Load vocabulary and dataset information."""
    try:
        vocab = Vocab.from_files("data/classes.txt", "data/attributes.yaml")
        df = pd.read_csv("data/labels.csv")
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        # Handle attributes parsing
        needed = ["color","material","condition","size"]
        has_wide = all(k in df.columns for k in needed)
        if ("attributes" in df.columns) and (not has_wide):
            from Milestone2_Image_Understanding_Pipeline import parse_attr_string
            parsed = df["attributes"].apply(parse_attr_string)
            for k in needed:
                df[k] = parsed.apply(lambda d: d.get(k, "unknown"))
        
        return vocab, df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

@st.cache_resource
def load_model(model_name: str, device: str):
    """Load the specified model and retrieval components."""
    try:
        # Map model names to actual model identifiers
        model_mapping = {
            "vits": "vit_small_patch16_224",
            "swint": "swin_tiny_patch4_window7_224"
        }
        
        actual_model_name = model_mapping.get(model_name, model_name)
        vocab, df = load_vocab_and_data()
        
        if vocab is None or df is None:
            return None, None, None, None
        
        device = torch.device(device)
        
        # Load the main model
        model = MultiTaskNet(actual_model_name, vocab).to(device)
        run_dir = os.path.join("outputs", actual_model_name)
        best_path = os.path.join(run_dir, "best.pt")
        
        if not os.path.exists(best_path):
            st.error(f"Model checkpoint not found at {best_path}")
            return None, None, None, None
            
        state = torch.load(best_path, map_location="cpu")
        model.load_state_dict(state["model"])
        model.eval()
        
        # Load retrieval components
        retrieval_path = os.path.join(run_dir, "retrieval.pt")
        if not os.path.exists(retrieval_path):
            st.error(f"Retrieval checkpoint not found at {retrieval_path}")
            return None, None, None, None
            
        rstate = torch.load(retrieval_path, map_location="cpu")
        
        txt_encoder = TextEncoder().to(device)
        txt_encoder.load_state_dict(rstate["text_encoder"])
        
        retrieval_head = RetrievalHead(model.backbone.num_features, txt_encoder.dim, 256).to(device)
        retrieval_head.load_state_dict(rstate["retrieval_head"])
        
        return model, txt_encoder, retrieval_head, vocab
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

def encode_gallery_images(model, device, df):
    """Encode all gallery images for retrieval."""
    try:
        _, tfm = make_transforms(224)
        img_paths = []
        img_feats = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, p in enumerate(df["path"].tolist()):
            status_text.text(f"Encoding image {i+1}/{len(df)}")
            progress_bar.progress((i+1)/len(df))
            
            full = p if os.path.isabs(p) else os.path.join("data", p)
            if os.path.exists(full):
                with Image.open(full) as im:
                    im = im.convert("RGB")
                    x = tfm(im).unsqueeze(0).to(device)
                with torch.no_grad():
                    f = model.extract_features(x)
                img_paths.append(full)
                img_feats.append(f)
        
        progress_bar.empty()
        status_text.empty()
        
        if img_feats:
            img_feats = torch.cat(img_feats, dim=0)
            return img_paths, img_feats
        else:
            return [], torch.tensor([])
            
    except Exception as e:
        st.error(f"Error encoding gallery images: {e}")
        return [], torch.tensor([])

def perform_retrieval(query: str, model, txt_encoder, retrieval_head, img_paths, img_feats, topk: int, device):
    """Perform image retrieval based on text query."""
    try:
        with torch.no_grad():
            # Encode query
            qz = txt_encoder.encode([query], device)
            
            # Get similarity scores
            zi, zt, _ = retrieval_head(img_feats, qz)
            scores = (zi @ zt.t()).squeeze(1).cpu().numpy()
            
            # Get top-k results
            top_idx = np.argsort(-scores)[:topk]
            
            results = []
            for rank, idx in enumerate(top_idx, 1):
                results.append({
                    'rank': rank,
                    'image_path': img_paths[idx],
                    'score': float(scores[idx]),
                    'index': int(idx)
                })
            
            return results
            
    except Exception as e:
        st.error(f"Error during retrieval: {e}")
        return []

def preprocess_uploaded_image(image: Image.Image) -> Image.Image:
    """Preprocess uploaded image: convert to JPG and resize to 96x96."""
    try:
        # Convert to RGB if necessary (handles RGBA, L, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to 96x96
        image = image.resize((96, 96), Image.Resampling.LANCZOS)
        
        return image
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

def predict_image_attributes(model, vocab: Vocab, image: Image.Image, device: str):
    """Predict class and attributes for a single image."""
    try:
        # Create transforms for prediction (using 224 for model input)
        _, tfm = make_transforms(224)
        
        # Preprocess image
        processed_img = preprocess_uploaded_image(image)
        if processed_img is None:
            return None
        
        # Apply model transforms
        x = tfm(processed_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(x)
        
        # Get predictions
        predictions = {}
        
        # Class prediction
        class_logits = outputs["class"]
        class_pred = class_logits.argmax(dim=1).item()
        class_prob = torch.softmax(class_logits, dim=1).max().item()
        predictions["class"] = {
            "prediction": vocab.classes[class_pred],
            "confidence": class_prob
        }
        
        # Attribute predictions
        for attr_name in ["color", "material", "condition", "size"]:
            attr_logits = outputs[attr_name]
            attr_pred = attr_logits.argmax(dim=1).item()
            attr_prob = torch.softmax(attr_logits, dim=1).max().item()
            
            # Get vocabulary for this attribute
            attr_vocab = getattr(vocab, f"{attr_name}s")
            predictions[attr_name] = {
                "prediction": attr_vocab[attr_pred],
                "confidence": attr_prob
            }
        
        return predictions
        
    except Exception as e:
        st.error(f"Error predicting attributes: {e}")
        return None

def display_predictions(predictions: Dict, image: Image.Image):
    """Display prediction results in a nice format."""
    if predictions is None:
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        st.subheader("🔮 Predictions")
        
        # Class prediction
        class_info = predictions["class"]
        st.markdown(f"**Class:** {humanize_label(class_info['prediction'])}")
        st.markdown(f"**Confidence:** {class_info['confidence']:.3f}")
        
        st.markdown("---")
        
        # Attribute predictions
        st.markdown("**Attributes:**")
        for attr_name in ["color", "material", "condition", "size"]:
            attr_info = predictions[attr_name]
            st.markdown(f"- **{attr_name.title()}:** {attr_info['prediction']} ({attr_info['confidence']:.3f})")

def display_results(results: List[Dict], df: pd.DataFrame, vocab: Vocab):
    """Display retrieval results in a nice format."""
    if not results:
        st.warning("No results found.")
        return
    
    st.subheader(f"Top {len(results)} Results")
    
    for result in results:
        with st.container():
            col1, col2 = st.columns([1, 2])
            
            with col1:
                try:
                    # Display image
                    image = Image.open(result['image_path'])
                    st.image(image, use_container_width=True, caption=f"Rank #{result['rank']}")
                except Exception as e:
                    st.error(f"Error loading image: {e}")
            
            with col2:
                # Get metadata for this result
                idx = result['index']
                if idx < len(df):
                    row = df.iloc[idx]
                    
                    # Display metadata
                    st.markdown(f"**Score:** <span class='score-badge'>{result['score']:.4f}</span>", unsafe_allow_html=True)
                    
                    # Class information
                    cls = humanize_label(row.get("class", ""))
                    st.markdown(f"**Class:** {cls}")
                    
                    # Attributes
                    color = row.get("color", "?")
                    material = row.get("material", "?")
                    condition = row.get("condition", "?")
                    size = row.get("size", "?")
                    
                    st.markdown(f"**Attributes:**")
                    st.markdown(f"- Color: {color}")
                    st.markdown(f"- Material: {material}")
                    st.markdown(f"- Condition: {condition}")
                    st.markdown(f"- Size: {size}")
                    
                    # File path
                    st.markdown(f"**File:** `{result['image_path']}`")
            
            st.markdown("---")

def main():
    # Header
    st.markdown('<h1 class="main-header">🖼️ Image Modeling System - Our Dataset</h1>', unsafe_allow_html=True)
    
    # Sidebar for controls
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        available_models = ["vits", "swint"]
        selected_model = st.selectbox(
            "Select Model",
            available_models,
            help="Choose between VitSmall and Swin Transformer models"
        )
        
        # Device selection
        device = st.selectbox(
            "Device",
            ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"],
            help="Select computation device"
        )
        
        # Top-k selection
        topk = st.slider(
            "Number of Results (Top-K)",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of top results to retrieve"
        )
        
        st.markdown("---")
        
        # Model information
        st.subheader("📊 Model Info")
        model_info = {
            "tinyvit": {
                "name": "VitSmall",
                "description": "Efficient vision transformer with 11M parameters",
                "architecture": "VIT Small"
            },
            "swint": {
                "name": "Swin-T",
                "description": "Swin Transformer with hierarchical structure",
                "architecture": "Swin Transformer"
            }
        }
        
        info = model_info.get(selected_model, {})
        st.markdown(f"**Model:** {info.get('name', 'Unknown')}")
        st.markdown(f"**Description:** {info.get('description', 'N/A')}")
        st.markdown(f"**Architecture:** {info.get('architecture', 'N/A')}")
        
        st.markdown("---")
        
        # Clear cache button
        if st.button("🗑️ Clear Cache", help="Clear cached image features"):
            for key in list(st.session_state.keys()):
                if key.endswith('_cpu') or key.endswith('_cuda'):
                    del st.session_state[key]
            st.success("Cache cleared!")
            st.rerun()
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["🔍 Image Retrieval", "📤 Image Prediction"])
    
    with tab1:
        # Main content area for retrieval
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🔍 Image Retrieval")
            
            # Query input
            query = st.text_input(
                "Enter your query:",
                placeholder="e.g., 'red leather bag', 'blue metal watch', 'wooden table'",
                help="Describe what you're looking for in the image database"
            )
            
            # Search button
            search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)
        
        with col2:
            st.subheader("💡 Tips")
            st.markdown("""
            **Query Examples:**
            - "red bag"
            - "metal watch"
            - "wooden table"
            - "blue shirt"
            - "leather shoes"
            
            **Best Practices:**
            - Use descriptive adjectives
            - Include color and material
            - Be specific about object type
            """)
        
        # Load data and model for retrieval
        if search_clicked and query.strip():
            with st.spinner("Loading model and data..."):
                vocab, df = load_vocab_and_data()
                if vocab is None or df is None:
                    st.error("Failed to load vocabulary and data. Please check your data files.")
                    return
                
                model, txt_encoder, retrieval_head, vocab = load_model(selected_model, device)
                if model is None:
                    st.error("Failed to load model. Please check your model files.")
                    return
            
            # Check if we have cached features for this model
            cache_key = f"{selected_model}_{device}"
            if cache_key not in st.session_state:
                with st.spinner("Encoding gallery images (this may take a moment)..."):
                    img_paths, img_feats = encode_gallery_images(model, device, df)
                    if len(img_paths) == 0:
                        st.error("No images found in the gallery.")
                        return
                    # Cache the results
                    st.session_state[cache_key] = (img_paths, img_feats)
            else:
                img_paths, img_feats = st.session_state[cache_key]
            
            with st.spinner("Performing retrieval..."):
                results = perform_retrieval(
                    query, model, txt_encoder, retrieval_head, 
                    img_paths, img_feats, topk, device
                )
            
            if results:
                display_results(results, df, vocab)
            else:
                st.warning("No results found for your query.")
        
        elif search_clicked and not query.strip():
            st.warning("Please enter a query to search.")
    
    with tab2:
        st.subheader("📤 Upload Image for Prediction")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload an image to predict its class and attributes"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            
            # Show original image
            st.subheader("📷 Uploaded Image")
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="Original Image", use_container_width=True)
            
            with col2:
                # Show preprocessed image (96x96)
                processed_img = preprocess_uploaded_image(image)
                if processed_img:
                    st.image(processed_img, caption="Preprocessed (96x96)", use_container_width=True)
            
            # Predict button
            predict_clicked = st.button("🔮 Predict Attributes", type="primary", use_container_width=True)
            
            if predict_clicked:
                with st.spinner("Loading model and predicting..."):
                    vocab, df = load_vocab_and_data()
                    if vocab is None or df is None:
                        st.error("Failed to load vocabulary and data.")
                    else:
                        model, txt_encoder, retrieval_head, vocab = load_model(selected_model, device)
                        if model is None:
                            st.error("Failed to load model.")
                        else:
                            predictions = predict_image_attributes(model, vocab, image, device)
                            if predictions:
                                st.markdown("---")
                                display_predictions(predictions, image)
                            else:
                                st.error("Failed to make predictions.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Image Modeling System** | Built with Streamlit and PyTorch | "
        "Powered by VITS and Swin Transformer models"
    )

if __name__ == "__main__":
    main()
