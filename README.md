# 🌊 Advanced Marine Life Research Dashboard

<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Gradio](https://img.shields.io/badge/gradio-4.0+-orange.svg)
![Plotly](https://img.shields.io/badge/plotly-5.18+-purple.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**A comprehensive exploratory data analysis (EDA) tool for marine biodiversity research**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Analysis Algorithms](#-analysis-algorithms) • [Screenshots](#-screenshots)

</div>

---

## 📖 About

The **Advanced Marine Life Research Dashboard** is an interactive web-based tool designed for marine biologists, researchers, and policy makers to perform comprehensive exploratory data analysis on marine biodiversity datasets. Built with Gradio and Plotly, it provides real-time interactive visualizations and advanced statistical analysis capabilities.

### 🎯 Key Highlights

- **15+ Analysis Types**: From basic taxonomic diversity to advanced multivariate analysis
- **Interactive Visualizations**: All charts powered by Plotly for zoom, pan, and hover interactions
- **Indian Shores Focus**: Specialized analysis for Bay of Bengal & Arabian Sea regions
- **Real-time Processing**: Instant visualization updates with adjustable parameters
- **No Coding Required**: User-friendly web interface for non-programmers

---

## ✨ Features

### 🧬 Taxonomic Analysis
- Hierarchical taxonomic diversity (Kingdom → Species)
- Top species identification by observation count
- Conservation status distribution (IUCN categories)

### 🗺️ Geographic Analysis
- Global species distribution heatmaps
- Density mapping with OpenStreetMap integration
- Regional filtering (with Indian Ocean focus)

### 🇮🇳 Indian Shores Analysis
- Dedicated Bay of Bengal analysis (5-22°N, 80-95°E)
- Arabian Sea analysis (8-24°N, 65-77°E)
- Regional species distribution and hotspot identification

### 🌊 Environmental Analysis
- Depth distribution and stratification
- Temperature and salinity profiling
- Habitat type classification
- Species richness vs. environmental parameters

### 📅 Temporal Analysis
- Time series trends (yearly)
- Seasonal pattern detection
- Observation frequency analysis

### 🔬 Advanced Analytics
- Principal Component Analysis (PCA)
- Independent Component Analysis (ICA)
- Factor Analysis with loadings
- Correlation matrix visualization
- K-means clustering

### 👶 Demographic Analysis
- Life stage distribution
- Sex ratio analysis
- Population structure insights

---

## 🧮 Analysis Algorithms

### 1. **Principal Component Analysis (PCA)**
**What it does:** Reduces high-dimensional environmental data (depth, temperature, salinity, coordinates) into principal components that capture maximum variance.

**Use case:** Identify the most important environmental factors driving species distribution. Visualize complex relationships in 2D/3D space.

**Algorithm:** Eigenvalue decomposition of the covariance matrix, followed by projection onto principal components.

---

### 2. **Independent Component Analysis (ICA)**
**What it does:** Separates multivariate signals into independent non-Gaussian components, assuming statistical independence.

**Use case:** Unmix overlapping environmental signals (e.g., separating coastal from deep-sea environmental patterns).

**Algorithm:** FastICA using maximum likelihood estimation with non-Gaussian component detection.

---

### 3. **Factor Analysis**
**What it does:** Identifies latent factors (unobserved variables) that explain correlations among observed environmental variables.

**Use case:** Discover hidden ecological factors driving marine biodiversity patterns.

**Algorithm:** Maximum likelihood estimation of factor loadings with varimax rotation.

---

### 4. **K-Means Clustering**
**What it does:** Groups observations into k clusters based on environmental similarity using Euclidean distance.

**Use case:** Identify distinct marine habitat types or ecological zones based on environmental conditions.

**Algorithm:** Iterative centroid optimization using Lloyd's algorithm with k-means++ initialization.

---

### 5. **Pearson Correlation Analysis**
**What it does:** Measures linear relationships between pairs of numeric variables (depth, temperature, salinity, abundance).

**Use case:** Identify which environmental factors are related and how strongly.

**Algorithm:** Computes correlation coefficient r ∈ [-1, 1] using covariance normalized by standard deviations.

---

### 6. **Density Heatmap Generation**
**What it does:** Creates geographic heatmaps showing species observation density using kernel density estimation.

**Use case:** Visualize biodiversity hotspots and species distribution patterns globally or regionally.

**Algorithm:** 2D Gaussian kernel density estimation on latitude/longitude coordinates.

---

### 7. **Species Richness Binning**
**What it does:** Calculates the number of unique species within depth intervals (bins).

**Use case:** Understand how biodiversity changes with ocean depth.

**Algorithm:** Stratified binning using Pandas cut function with equal-width intervals.

---

### 8. **Seasonal Decomposition**
**What it does:** Categorizes observations into seasons (Winter, Spring, Summer, Fall) and analyzes seasonal patterns.

**Use case:** Detect seasonal migration patterns or breeding cycles.

**Algorithm:** Month-to-season mapping with aggregation statistics.

---

### 9. **Taxonomic Diversity Metrics**
**What it does:** Counts unique taxa at each hierarchical level (kingdom, phylum, class, order, family, genus, species).

**Use case:** Assess dataset completeness and taxonomic coverage.

**Algorithm:** Unique value counting with hierarchical aggregation.

---

### 10. **Data Standardization**
**What it does:** Scales numeric features to zero mean and unit variance before multivariate analysis.

**Use case:** Ensures all environmental variables contribute equally to PCA/ICA (avoids scale bias).

**Algorithm:** Z-score normalization: `z = (x - μ) / σ`

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/marine-life-dashboard.git
cd marine-life-dashboard
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `gradio>=4.0.0` - Web UI framework
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computing
- `plotly>=5.18.0` - Interactive visualizations
- `scikit-learn>=1.3.0` - Machine learning algorithms

### Step 4: Verify Installation

```bash
python -c "import gradio, pandas, plotly; print('All dependencies installed successfully!')"
```

---

## 💻 Usage

### Running the Dashboard

```bash
python app.py
```

The dashboard will start on `http://localhost:7860`

### Using the Interface

1. **Upload Data**
   - Click "📁 Upload Marine Dataset"
   - Select your CSV or TSV file
   - Click "🔍 Load & Analyze Data"

2. **Explore Analysis Tabs**
   - Navigate through 11 different analysis tabs
   - Click buttons to generate visualizations
   - Hover over charts for detailed information

3. **Customize Parameters**
   - Use sidebar sliders to adjust:
     - Top N species (5-50)
     - Depth bins (5-20)
     - ICA/FA components (2-5)
   - Select features for multivariate analysis

4. **Export Results**
   - Right-click on any Plotly chart
   - Select "Save as PNG" or "Export to Plot.ly"

---

## 📊 Data Format

### Required Columns

Your CSV/TSV file should contain at minimum:

```csv
scientificName,decimalLatitude,decimalLongitude
Thunnus albacares,15.5,72.8
Sardinella longiceps,18.2,74.1
```

### Recommended Columns

For full functionality, include:

| Column Name | Description | Example |
|-------------|-------------|---------|
| `scientificName` | Species scientific name | *Thunnus albacares* |
| `decimalLatitude` | Latitude coordinate | 15.5 |
| `decimalLongitude` | Longitude coordinate | 72.8 |
| `depth` | Depth in meters | 50.0 |
| `eventDate` | Observation date | 2023-01-15 |
| `temperature` | Water temperature (°C) | 26.5 |
| `salinity` | Salinity (PSU) | 35.2 |
| `habitat` | Habitat type | Coral reef |
| `lifeStage` | Life stage | Adult |
| `conservationStatus` | IUCN status | Vulnerable |
| `individualCount` | Number of individuals | 10 |

### Taxonomic Hierarchy (Optional)

```csv
kingdom,phylum,class,order,family,genus,species
Animalia,Chordata,Actinopterygii,Perciformes,Scombridae,Thunnus,albacares
```

### Example Dataset Sources

- [OBIS (Ocean Biodiversity Information System)](https://obis.org/)
- [GBIF (Global Biodiversity Information Facility)](https://www.gbif.org/)
- [WoRMS (World Register of Marine Species)](https://www.marinespecies.org/)

---

## 📁 Project Structure

```
marine-life-dashboard/
│
├── app.py                 # Gradio frontend and UI orchestration
├── eda.py                 # EDA analysis functions (Plotly visualizations)
├── utils.py               # Data loading and helper functions
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── QUICKSTART.md         # Quick setup guide
│
└── (future additions)
    ├── data/             # Sample datasets
    ├── tests/            # Unit tests
    └── docs/             # Additional documentation
```

---

## 🖼️ Screenshots

### Main Dashboard
![Dashboard Overview](https://via.placeholder.com/800x400.png?text=Upload+Your+Screenshot+Here)

### Indian Shores Heatmap
![Indian Shores Analysis](https://via.placeholder.com/800x400.png?text=Upload+Your+Screenshot+Here)

### PCA Analysis
![PCA Visualization](https://via.placeholder.com/800x400.png?text=Upload+Your+Screenshot+Here)

> **Note:** Replace placeholder images with actual screenshots of your dashboard

---

## 🛠️ Customization

### Changing Port Number

Edit `app.py`, line 475:

```python
app.launch(share=False, server_name="0.0.0.0", server_port=8080)  # Changed from 7860
```

### Adding New Analysis

1. **Add function to `eda.py`:**
```python
def create_new_analysis(self) -> Tuple[Optional[go.Figure], str]:
    # Your analysis code
    fig = go.Figure(...)
    return fig, "✅ Analysis complete"
```

2. **Add wrapper in `app.py`:**
```python
def generate_new_analysis():
    fig, msg = analyzer.create_new_analysis()
    return fig, msg
```

3. **Add UI component:**
```python
with gr.Tab("New Analysis"):
    new_btn = gr.Button("Generate")
    new_chart = gr.Plot()
    new_status = gr.Markdown()
    
new_btn.click(fn=generate_new_analysis, outputs=[new_chart, new_status])
```

---

## 🐛 Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Port already in use | Change port in `app.py` (see Customization) |
| No data in visualizations | Check CSV column names match expected format |
| Out of memory error | Reduce dataset size or filter by region |

### Debug Mode

Run with verbose output:

```bash
python app.py --debug
```

### Getting Help

1. Check [Issues](https://github.com/yourusername/marine-life-dashboard/issues) for similar problems
2. Open a new issue with:
   - Error message
   - Python version (`python --version`)
   - Operating system
   - Sample data (if possible)

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/marine-life-dashboard.git
cd marine-life-dashboard

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black .
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **OBIS** for marine biodiversity data standards
- **Plotly** for interactive visualization library
- **Gradio** for making ML/data science interfaces accessible
- **Marine researchers worldwide** for contributing to open biodiversity databases

---

## 📚 Citation

If you use this dashboard in your research, please cite:

```bibtex
@software{marine_life_dashboard,
  author = {Your Name},
  title = {Advanced Marine Life Research Dashboard},
  year = {2024},
  url = {https://github.com/yourusername/marine-life-dashboard}
}
```

---

## 📧 Contact

**Project Maintainer:** Your Name

- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)

---

## 🗺️ Roadmap

### Planned Features

- [ ] Export analysis reports to PDF
- [ ] Real-time data streaming support
- [ ] Species co-occurrence network analysis
- [ ] Integration with OBIS/GBIF APIs
- [ ] Machine learning species prediction
- [ ] Multi-language support
- [ ] Mobile-responsive design
- [ ] Docker containerization

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/marine-life-dashboard&type=Date)](https://star-history.com/#yourusername/marine-life-dashboard&Date)

---

<div align="center">

**Built with ❤️ for Marine Conservation Research**

[⬆ Back to Top](#-advanced-marine-life-research-dashboard)

</div>
