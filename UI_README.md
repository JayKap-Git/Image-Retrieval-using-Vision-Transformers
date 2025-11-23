# Image Modeling System UI

A minimalistic web interface for the Image Understanding Pipeline that allows users to search for images using natural language queries.

## Features

- **Model Selection**: Choose between TinyViT and Swin Transformer models
- **Text Query**: Enter natural language descriptions to find similar images
- **Image Upload**: Upload images for class and attribute prediction
- **Image Preprocessing**: Automatic conversion to JPG and resizing to 96x96px
- **Top-K Results**: Adjustable number of results (1-20)
- **Rich Display**: Shows images with metadata including class, color, material, condition, and size
- **Real-time Search**: Fast retrieval using pre-trained models
- **Attribute Prediction**: Get class and attribute predictions for uploaded images

## Quick Start

### Prerequisites

Make sure you have the trained models and data files:

```bash
# Required files structure:
data/
├── classes.txt
├── attributes.yaml
├── labels.csv
└── Images/
    ├── Dataset_jk/
    ├── Dataset_prak/
    └── Dataset_swar/

outputs/
├── tiny_vit_11m_224/
│   ├── best.pt
│   └── retrieval.pt
└── swin_tiny_patch4_window7_224/
    ├── best.pt
    └── retrieval.pt
```

### Installation

1. Install the required dependencies:
```bash
pip install -r requirements_ui.txt
```

2. Launch the UI:
```bash
python run_ui.py
```

Or directly with Streamlit:
```bash
streamlit run app.py
```

### Usage

#### Image Retrieval Tab:
1. **Select Model**: Choose between TinyViT or Swin Transformer from the sidebar
2. **Enter Query**: Type a natural language description (e.g., "red leather bag", "blue metal watch")
3. **Adjust Top-K**: Use the slider to set how many results you want (1-20)
4. **Search**: Click the search button to find similar images
5. **View Results**: Browse through the retrieved images with their metadata

#### Image Prediction Tab:
1. **Upload Image**: Click "Choose an image file" and select an image (PNG, JPG, JPEG, BMP, TIFF)
2. **Preview**: See both original and preprocessed (96x96) versions of your image
3. **Predict**: Click "Predict Attributes" to get class and attribute predictions
4. **View Results**: See predicted class, color, material, condition, and size with confidence scores

### Example Queries

- "red bag"
- "metal watch" 
- "wooden table"
- "blue shirt"
- "leather shoes"
- "black backpack"
- "white ceramic mug"

### Interface Components

- **Sidebar**: Model selection, device choice, top-k configuration, and cache management
- **Image Retrieval Tab**: Query input and search results
- **Image Prediction Tab**: File upload and prediction results
- **Results Display**: Images with metadata including:
  - Similarity score (for retrieval)
  - Object class
  - Color, material, condition, size attributes
  - Confidence scores (for predictions)
  - File path

### Technical Details

- **Backend**: PyTorch with pre-trained vision transformers
- **Text Encoder**: DistilBERT for query understanding
- **Retrieval**: Contrastive learning with InfoNCE loss
- **UI Framework**: Streamlit for rapid prototyping
- **Caching**: Optimized for fast repeated searches

### Troubleshooting

**Model not found errors:**
- Ensure model checkpoints exist in `outputs/` directory
- Run the training pipeline first if models are missing

**Data loading errors:**
- Check that `data/labels.csv`, `data/classes.txt`, and `data/attributes.yaml` exist
- Verify image paths in the CSV are correct

**Memory issues:**
- Use CPU mode for lower memory usage
- Reduce batch size in the pipeline if needed

### Performance Tips

- First search may be slower due to model loading
- Subsequent searches are much faster due to caching
- Use specific, descriptive queries for better results
- GPU acceleration recommended for large datasets

## Architecture

The UI integrates with the existing pipeline by:

1. Loading pre-trained models and retrieval heads
2. Encoding the entire image gallery once
3. Processing text queries through DistilBERT
4. Computing similarity scores using learned projections
5. Displaying top-k results with rich metadata

This provides a user-friendly interface to the powerful image understanding capabilities of the underlying pipeline.
