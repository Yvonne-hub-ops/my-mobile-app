# 🌱 SeedVision AI — Machine Vision Seed Quality Detector

A Python Flask web application that uses computer vision to analyze seed quality from images.

## Features

- **5-Metric CV Pipeline**: Size, Shape, Color, Texture, Defect Detection
- **Real-time Analysis**: Upload any seed image and get instant results
- **Annotated Output**: Visualized contour detection and quality overlays
- **Grade System**: A–F grading with detailed breakdown
- **Scan History**: Tracks recent analyses in-session

## Computer Vision Pipeline

| Step | Method | Output |
|------|--------|--------|
| Segmentation | Otsu thresholding + morphological ops | Seed mask |
| Size | Contour area vs image area | Size score |
| Shape | Circularity, solidity, aspect ratio | Shape score |
| Color | HSV hue std dev, saturation, brightness | Color score |
| Texture | Laplacian variance, local std dev | Texture score |
| Defects | Dark spot detection, edge irregularity | Defect score |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
python app.py

# 3. Open browser
# Navigate to http://localhost:5050
```

## Score Interpretation

| Grade | Score | Label |
|-------|-------|-------|
| A | 85–100 | Premium |
| B | 70–84 | Good |
| C | 55–69 | Acceptable |
| D | 40–54 | Poor |
| F | 0–39 | Reject |

## Project Structure

```
seed_detector/
├── app.py           # Flask app + CV pipeline
├── requirements.txt # Python dependencies
└── README.md        # This file
```

## Tips for Best Results

- Use well-lit, high-contrast images
- Place seed on a plain white or dark background
- Single seed per image for best accuracy
- Works with: wheat, rice, corn, bean, sunflower seeds
