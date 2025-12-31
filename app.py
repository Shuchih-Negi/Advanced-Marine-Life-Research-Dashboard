"""
Marine Life Research Dashboard - Gradio Frontend
Main application entry point with enhanced UI
"""

import gradio as gr
import pandas as pd
from utils import DataLoader
from eda import MarineEDAAnalyzer

# Initialize components
loader = DataLoader()
analyzer = MarineEDAAnalyzer()

# UI Functions
def load_csv_file(file):
    """Load CSV/TSV file"""
    if file is None:
        return "Please upload a CSV/TSV file", None, None, gr.update(choices=[]), gr.update(visible=False)
    
    success, message, df = loader.load_data(file.name)
    
    if success:
        analyzer.set_data(df)
        overview = analyzer.get_overview_stats()
        preview = df.head(20)
        stats = df.describe(include='all')
        
        num_cols = loader.get_numeric_columns()
        
        return (
            overview, 
            preview, 
            stats, 
            gr.update(choices=num_cols, value=num_cols[:5] if len(num_cols) >= 5 else num_cols),
            gr.update(visible=True)
        )
    else:
        return message, None, None, gr.update(choices=[]), gr.update(visible=False)

def generate_population_viz(top_n):
    fig, msg = analyzer.create_population_distribution(top_n=top_n)
    return fig, msg

def generate_geo_map():
    fig, msg = analyzer.create_geographic_heatmap()
    return fig, msg

def generate_indian_heatmap(species_filter):
    fig, msg = analyzer.create_indian_shores_heatmap(species_filter)
    return fig, msg

def generate_indian_species(top_n):
    fig, msg = analyzer.create_indian_species_distribution(top_n)
    return fig, msg

def generate_correlation():
    fig, msg = analyzer.create_correlation_heatmap()
    return fig, msg

def generate_richness_depth(bins):
    fig, msg = analyzer.species_richness_depth(bins=bins)
    return fig, msg

def generate_abundance_depth():
    fig, msg = analyzer.abundance_vs_depth()
    return fig, msg

def generate_temporal():
    fig, msg = analyzer.temporal_trends()
    return fig, msg

def generate_unmixing(feature_cols, n_components):
    fig_ica, fig_fa, msg = analyzer.multivariate_unmixing(feature_cols, n_components)
    return fig_ica, fig_fa, msg

def generate_taxonomic_diversity():
    fig, msg = analyzer.create_taxonomic_diversity()
    return fig, msg

def generate_top_species(top_n):
    fig, msg = analyzer.create_top_species(top_n)
    return fig, msg

def generate_conservation_status():
    fig, msg = analyzer.create_conservation_status()
    return fig, msg

def generate_habitat_distribution():
    fig, msg = analyzer.create_habitat_distribution()
    return fig, msg

def generate_depth_analysis():
    fig, msg = analyzer.create_depth_distribution()
    return fig, msg

def generate_environmental_analysis():
    fig, msg = analyzer.create_environmental_analysis()
    return fig, msg

def generate_seasonal_patterns():
    fig, msg = analyzer.create_seasonal_patterns()
    return fig, msg

def generate_life_stage_distribution():
    fig, msg = analyzer.create_life_stage_distribution()
    return fig, msg

def generate_pca_analysis():
    fig, msg = analyzer.create_pca_analysis()
    return fig, msg

# Create Gradio Dashboard
with gr.Blocks(title="Marine Life Research Dashboard", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🌊 Advanced Marine Life Research Dashboard
    ### For Researchers & Policy Makers
    Upload your marine life dataset (CSV/TSV) for comprehensive exploratory data analysis
    """)
    
    with gr.Row():
        # Enhanced Sidebar
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("## 📁 Data Management")
            file_input = gr.File(label="Upload Marine Dataset", file_types=[".csv", ".tsv", ".txt"])
            load_btn = gr.Button("🔍 Load & Analyze Data", variant="primary", size="lg")
            
            gr.Markdown("---")
            gr.Markdown("## ⚙️ Visualization Controls")
            
            with gr.Accordion("🐟 Population Settings", open=True):
                top_n_slider = gr.Slider(5, 50, value=20, step=5, label="Top N Species")
            
            with gr.Accordion("🇮🇳 Indian Shores Settings", open=True):
                indian_top_n = gr.Slider(5, 30, value=15, step=5, label="Top N Species (India)")
                species_filter_text = gr.Textbox(label="Filter by Species (optional)", placeholder="e.g., Tuna, Sardine")
            
            with gr.Accordion("🌊 Depth Analysis", open=False):
                depth_bins_slider = gr.Slider(5, 20, value=10, step=1, label="Depth Bins")
            
            with gr.Accordion("🔬 Advanced Analysis", open=False):
                n_components_slider = gr.Slider(2, 5, value=3, step=1, label="ICA/FA Components")
                feature_selector = gr.CheckboxGroup(label="Select Features for Unmixing", choices=[])
            
            gr.Markdown("---")
            gr.Markdown("""
            ## 📊 Dashboard Features
            
            ### Global Analysis
            - Population Distribution
            - Geographic Heatmap
            - Correlation Matrix
            - Species Richness
            - Temporal Trends
            
            ### Indian Shores Focus
            - 🇮🇳 Fish Population Heatmap
            - 🇮🇳 Species Distribution
            - Regional Biodiversity
            
            ### Advanced Tools
            - Taxonomic Diversity
            - Conservation Status
            - ICA/Factor Analysis
            - Environmental Analysis
            """)
            
            analysis_status = gr.Markdown("### Status: Ready", visible=True)
        
        # Main Content Area
        with gr.Column(scale=3):
            overview_output = gr.Markdown("### 📤 Upload a dataset to begin analysis")
            
            with gr.Tabs():
                with gr.Tab("📋 Data Overview"):
                    gr.Markdown("### Dataset Preview")
                    preview_output = gr.Dataframe(label="First 20 Rows", interactive=False, wrap=True)
                    gr.Markdown("### Summary Statistics")
                    stats_output = gr.Dataframe(label="Descriptive Statistics", interactive=False)
                
                with gr.Tab("🧬 Taxonomic Analysis"):
                    gr.Markdown("### Taxonomic Diversity & Species Distribution")
                    
                    with gr.Row():
                        refresh_taxonomic_btn = gr.Button("📊 Taxonomic Diversity", variant="primary")
                        refresh_top_species_btn = gr.Button("🏆 Top Species", variant="secondary")
                        refresh_conservation_btn = gr.Button("🛡️ Conservation Status", variant="secondary")
                    
                    taxonomic_status = gr.Markdown()
                    taxonomic_chart = gr.Plot(label="Taxonomic Diversity")
                    
                    top_species_status = gr.Markdown()
                    top_species_chart = gr.Plot(label="Top Species Distribution")
                    
                    conservation_status_msg = gr.Markdown()
                    conservation_chart = gr.Plot(label="Conservation Status")
                
                with gr.Tab("🇮🇳 Indian Shores Analysis"):
                    gr.Markdown("### Fish Population in Indian Waters")
                    gr.Markdown("*Focused analysis for Indian Ocean coastline (Bay of Bengal & Arabian Sea)*")
                    
                    with gr.Row():
                        refresh_indian_map_btn = gr.Button("🗺️ Generate Heatmap", variant="primary")
                        refresh_indian_species_btn = gr.Button("📊 Top Species", variant="secondary")
                    
                    indian_map_status = gr.Markdown()
                    indian_heatmap = gr.Plot(label="Indian Shores Fish Population Heatmap")
                    
                    indian_species_status = gr.Markdown()
                    indian_species_chart = gr.Plot(label="Top Species in Indian Waters")
                
                with gr.Tab("📊 Global Population"):
                    refresh_pop_btn = gr.Button("🔄 Refresh Visualization")
                    pop_status = gr.Markdown()
                    pop_chart = gr.Plot(label="Population Distribution")
                
                with gr.Tab("🗺️ Global Distribution"):
                    refresh_geo_btn = gr.Button("🔄 Refresh Map")
                    geo_status = gr.Markdown()
                    geo_map = gr.Plot(label="Geographic Heatmap")
                
                with gr.Tab("🏞️ Habitat & Environment"):
                    gr.Markdown("### Habitat Distribution & Environmental Conditions")
                    
                    with gr.Row():
                        refresh_habitat_btn = gr.Button("🏝️ Habitat Distribution", variant="primary")
                        refresh_env_btn = gr.Button("🌡️ Environmental Analysis", variant="secondary")
                    
                    habitat_status = gr.Markdown()
                    habitat_chart = gr.Plot(label="Habitat Distribution")
                    
                    env_status = gr.Markdown()
                    env_chart = gr.Plot(label="Environmental Analysis")
                
                with gr.Tab("🔥 Correlation Analysis"):
                    refresh_corr_btn = gr.Button("🔄 Refresh Matrix")
                    corr_status = gr.Markdown()
                    corr_heatmap = gr.Plot(label="Correlation Matrix")
                
                with gr.Tab("🌊 Depth Analysis"):
                    gr.Markdown("### Species Distribution & Abundance by Depth")
                    
                    with gr.Row():
                        refresh_depth_dist_btn = gr.Button("📊 Depth Distribution", variant="primary")
                        refresh_richness_btn = gr.Button("🐠 Species Richness", variant="secondary")
                        refresh_abundance_btn = gr.Button("📈 Abundance vs Depth", variant="secondary")
                    
                    depth_dist_status = gr.Markdown()
                    depth_dist_chart = gr.Plot(label="Depth Distribution")
                    
                    richness_status = gr.Markdown()
                    richness_chart = gr.Plot(label="Species Richness by Depth")
                    
                    abundance_status = gr.Markdown()
                    abundance_chart = gr.Plot(label="Abundance vs Depth")
                
                with gr.Tab("📅 Temporal Trends"):
                    gr.Markdown("### Time Series & Seasonal Analysis")
                    
                    with gr.Row():
                        refresh_temporal_btn = gr.Button("📈 Temporal Trends", variant="primary")
                        refresh_seasonal_btn = gr.Button("🌸 Seasonal Patterns", variant="secondary")
                    
                    temporal_status = gr.Markdown()
                    temporal_chart = gr.Plot(label="Temporal Trends")
                    
                    seasonal_status = gr.Markdown()
                    seasonal_chart = gr.Plot(label="Seasonal Patterns")
                
                with gr.Tab("👶 Demographic Analysis"):
                    gr.Markdown("### Life Stage Distribution")
                    refresh_life_stage_btn = gr.Button("🔄 Refresh Analysis")
                    life_stage_status = gr.Markdown()
                    life_stage_chart = gr.Plot(label="Life Stage Distribution")
                
                with gr.Tab("🔬 Advanced Analytics"):
                    gr.Markdown("### PCA, ICA & Q-mode Factor Analysis")
                    gr.Markdown("*Select numeric features from the sidebar for multivariate analysis*")
                    
                    with gr.Row():
                        pca_btn = gr.Button("📊 PCA Analysis", variant="primary", size="lg")
                        unmix_btn = gr.Button("🧬 ICA/FA Unmixing", variant="secondary", size="lg")
                    
                    pca_status = gr.Markdown()
                    pca_plot = gr.Plot(label="PCA Analysis")
                    
                    unmix_status = gr.Markdown()
                    with gr.Row():
                        ica_plot = gr.Plot(label="ICA Components")
                        fa_plot = gr.Plot(label="Factor Loadings")
    
    # Event Handlers
    load_btn.click(
        fn=load_csv_file,
        inputs=[file_input],
        outputs=[overview_output, preview_output, stats_output, feature_selector, analysis_status]
    )
    
    # Taxonomic Analysis
    refresh_taxonomic_btn.click(
        fn=generate_taxonomic_diversity,
        inputs=[],
        outputs=[taxonomic_chart, taxonomic_status]
    )
    
    refresh_top_species_btn.click(
        fn=generate_top_species,
        inputs=[top_n_slider],
        outputs=[top_species_chart, top_species_status]
    )
    
    refresh_conservation_btn.click(
        fn=generate_conservation_status,
        inputs=[],
        outputs=[conservation_chart, conservation_status_msg]
    )
    
    # Indian Shores
    refresh_indian_map_btn.click(
        fn=generate_indian_heatmap,
        inputs=[species_filter_text],
        outputs=[indian_heatmap, indian_map_status]
    )
    
    refresh_indian_species_btn.click(
        fn=generate_indian_species,
        inputs=[indian_top_n],
        outputs=[indian_species_chart, indian_species_status]
    )
    
    # Global Analysis
    refresh_pop_btn.click(
        fn=generate_population_viz,
        inputs=[top_n_slider],
        outputs=[pop_chart, pop_status]
    )
    
    refresh_geo_btn.click(
        fn=generate_geo_map,
        inputs=[],
        outputs=[geo_map, geo_status]
    )
    
    # Habitat & Environment
    refresh_habitat_btn.click(
        fn=generate_habitat_distribution,
        inputs=[],
        outputs=[habitat_chart, habitat_status]
    )
    
    refresh_env_btn.click(
        fn=generate_environmental_analysis,
        inputs=[],
        outputs=[env_chart, env_status]
    )
    
    # Correlation
    refresh_corr_btn.click(
        fn=generate_correlation,
        inputs=[],
        outputs=[corr_heatmap, corr_status]
    )
    
    # Depth Analysis
    refresh_depth_dist_btn.click(
        fn=generate_depth_analysis,
        inputs=[],
        outputs=[depth_dist_chart, depth_dist_status]
    )
    
    refresh_richness_btn.click(
        fn=generate_richness_depth,
        inputs=[depth_bins_slider],
        outputs=[richness_chart, richness_status]
    )
    
    refresh_abundance_btn.click(
        fn=generate_abundance_depth,
        inputs=[],
        outputs=[abundance_chart, abundance_status]
    )
    
    # Temporal
    refresh_temporal_btn.click(
        fn=generate_temporal,
        inputs=[],
        outputs=[temporal_chart, temporal_status]
    )
    
    refresh_seasonal_btn.click(
        fn=generate_seasonal_patterns,
        inputs=[],
        outputs=[seasonal_chart, seasonal_status]
    )
    
    # Demographics
    refresh_life_stage_btn.click(
        fn=generate_life_stage_distribution,
        inputs=[],
        outputs=[life_stage_chart, life_stage_status]
    )
    
    # Advanced Analytics
    pca_btn.click(
        fn=generate_pca_analysis,
        inputs=[],
        outputs=[pca_plot, pca_status]
    )
    
    unmix_btn.click(
        fn=generate_unmixing,
        inputs=[feature_selector, n_components_slider],
        outputs=[ica_plot, fa_plot, unmix_status]
    )
    
    gr.Markdown("""
    ---
    ### 💡 Dashboard Guide for Researchers & Policy Makers
    
    **Getting Started:**
    1. Upload your marine biodiversity CSV/TSV dataset
    2. Dashboard auto-detects columns (species, coordinates, depth, etc.)
    3. Use sidebar controls to customize visualizations
    
    **Key Features:**
    - **Indian Shores Tab**: Dedicated analysis for Indian Ocean marine life
    - **Auto-detection**: Smart column recognition for various data formats
    - **Interactive Maps**: Density heatmaps with OpenStreetMap
    - **Statistical Tools**: Correlation, PCA, ICA, Factor Analysis
    
    **Data Requirements:**
    - Species names (scientificName, species, etc.)
    - Coordinates (decimalLatitude, decimalLongitude)
    - Optional: depth, abundance, temporal data
    
    🌊 **Developed for Marine Conservation Research**
    """)


if __name__ == "__main__":
    print("🌊 Starting Advanced Marine Life Research Dashboard...")
    print("📊 Designed for Researchers & Policy Makers")
    print("🇮🇳 Includes specialized Indian Shores analysis")
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)
