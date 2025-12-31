"""
Advanced EDA Analysis Module - Plotly Visualizations
All matplotlib/seaborn graphs converted to Plotly for Gradio integration
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from utils import detect_indian_shores, get_season


class MarineEDAAnalyzer:
    """Advanced EDA analyzer with Plotly visualizations"""
    
    def __init__(self):
        self.df = None
        
    def set_data(self, df: pd.DataFrame):
        """Set the dataframe for analysis"""
        self.df = df.copy()
    
    def get_overview_stats(self) -> str:
        """Generate overview statistics"""
        if self.df is None:
            return "No data loaded"
        
        stats = f"""
## 📊 Dataset Overview

- **Total Records**: {len(self.df):,}
- **Total Columns**: {len(self.df.columns)}
- **Memory Usage**: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
"""
        if 'scientificName' in self.df.columns:
            stats += f"- **Unique Species**: {self.df['scientificName'].nunique():,}\n"
        
        if 'eventDate' in self.df.columns:
            stats += f"- **Date Range**: {self.df['eventDate'].min()} to {self.df['eventDate'].max()}\n"
        
        if 'country' in self.df.columns:
            stats += f"- **Countries**: {self.df['country'].nunique()}\n"
            
        return stats
    
    # ==================== TAXONOMIC ANALYSIS ====================
    
    def create_taxonomic_diversity(self) -> Tuple[Optional[go.Figure], str]:
        """Taxonomic diversity across hierarchical levels"""
        if self.df is None:
            return None, "❌ No data loaded"
        
        try:
            taxonomic_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
            available_levels = [level for level in taxonomic_levels if level in self.df.columns]
            
            if not available_levels:
                return None, "⚠️ No taxonomic columns found"
            
            diversity = {level: self.df[level].nunique() for level in available_levels}
            
            fig = go.Figure(data=[
                go.Bar(
                    y=list(diversity.keys()),
                    x=list(diversity.values()),
                    orientation='h',
                    marker=dict(color='steelblue'),
                    text=list(diversity.values()),
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title='Taxonomic Diversity Across Hierarchical Levels',
                xaxis_title='Number of Unique Taxa',
                yaxis_title='Taxonomic Level',
                height=500,
                showlegend=False
            )
            
            return fig, f"✅ Found {len(available_levels)} taxonomic levels"
            
        except Exception as e:
            return None, f"❌ Error: {str(e)}"
    
    def create_top_species(self, top_n: int = 20) -> Tuple[Optional[go.Figure], str]:
        """Top N most observed species"""
        if self.df is None or 'scientificName' not in self.df.columns:
            return None, "❌ scientificName column not found"
        
        try:
            species_counts = self.df['scientificName'].value_counts().head(top_n)
            
            fig = go.Figure(data=[
                go.Bar(
                    y=species_counts.index[::-1],
                    x=species_counts.values[::-1],
                    orientation='h',
                    marker=dict(color='coral'),
                    text=species_counts.values[::-1],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title=f'Top {top_n} Most Observed Species',
                xaxis_title='Number of Observations',
                yaxis_title='Species',
                height=600,
                showlegend=False
            )
            
            return fig, f"✅ Showing top {top_n} species"
            
        except Exception as e:
            return None, f"❌ Error: {str(e)}"
    
    def create_conservation_status(self) -> Tuple[Optional[go.Figure], str]:
        """Conservation status distribution"""
        if self.df is None or 'conservationStatus' not in self.df.columns:
            return None, "❌ conservationStatus column not found"
        
        try:
            status_counts = self.df['conservationStatus'].value_counts()
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=status_counts.index,
                    values=status_counts.values,
                    hole=0.3,
                    marker=dict(colors=px.colors.qualitative.Set3)
                )
            ])
            
            fig.update_layout(
                title='Conservation Status Distribution',
                height=500
            )
            
            return fig, f"✅ {len(status_counts)} conservation categories"
            
        except Exception as e:
            return None, f"❌ Error: {str(e)}"
    
    # ==================== POPULATION & GEOGRAPHIC ====================
    
    def create_population_distribution(self, top_n: int = 20) -> Tuple[Optional[go.Figure], str]:
        """Population distribution visualization"""
        return self.create_top_species(top_n)
    
    def create_geographic_heatmap(self) -> Tuple[Optional[go.Figure], str]:
        """Global geographic distribution heatmap"""
        if self.df is None:
            return None, "❌ No data loaded"
        
        if 'decimalLatitude' not in self.df.columns or 'decimalLongitude' not in self.df.columns:
            return None, "❌ Coordinate columns not found"
        
        try:
            df_geo = self.df[['decimalLatitude', 'decimalLongitude']].dropna()
            
            if len(df_geo) == 0:
                return None, "❌ No valid coordinates found"
            
            fig = go.Figure(go.Densitymapbox(
                lat=df_geo['decimalLatitude'],
                lon=df_geo['decimalLongitude'],
                radius=10,
                colorscale='Viridis',
                showscale=True,
                hoverinfo='skip'
            ))
            
            fig.update_layout(
                mapbox_style="open-street-map",
                mapbox=dict(
                    center=dict(
                        lat=df_geo['decimalLatitude'].mean(),
                        lon=df_geo['decimalLongitude'].mean()
                    ),
                    zoom=2
                ),
                title='Global Marine Species Distribution Heatmap',
                height=600
            )
            
            return fig, f"✅ Plotted {len(df_geo):,} locations"
            
        except Exception as e:
            return None, f"❌ Error: {str(e)}"
    
    # ==================== INDIAN SHORES ANALYSIS ====================
    
    def create_indian_shores_heatmap(self, species_filter: str = "") -> Tuple[Optional[go.Figure], str]:
        """Indian shores (Bay of Bengal & Arabian Sea) heatmap"""
        if self.df is None:
            return None, "❌ No data loaded"
        
        try:
            indian_df = detect_indian_shores(self.df)
            
            if len(indian_df) == 0:
                return None, "❌ No data found for Indian shores"
            
            # Apply species filter if provided
            if species_filter and 'scientificName' in indian_df.columns:
                indian_df = indian_df[
                    indian_df['scientificName'].str.contains(species_filter, case=False, na=False)
                ]
                
                if len(indian_df) == 0:
                    return None, f"❌ No species found matching '{species_filter}'"
            
            fig = go.Figure(go.Densitymapbox(
                lat=indian_df['decimalLatitude'],
                lon=indian_df['decimalLongitude'],
                radius=15,
                colorscale='YlOrRd',
                showscale=True,
                hoverinfo='skip'
            ))
            
            fig.update_layout(
                mapbox_style="open-street-map",
                mapbox=dict(
                    center=dict(lat=15, lon=77),
                    zoom=4
                ),
                title=f'🇮🇳 Indian Shores Fish Population Heatmap',
                height=600
            )
            
            msg = f"✅ Plotted {len(indian_df):,} observations"
            if species_filter:
                msg += f" (filtered: {species_filter})"
                
            return fig, msg
            
        except Exception as e:
            return None, f"❌ Error: {str(e)}"
    
    def create_indian_species_distribution(self, top_n: int = 15) -> Tuple[Optional[go.Figure], str]:
        """Top species in Indian waters"""
        if self.df is None or 'scientificName' not in self.df.columns:
            return None, "❌ No data loaded"
        
        try:
            indian_df = detect_indian_shores(self.df)
            
            if len(indian_df) == 0:
                return None, "❌ No data found for Indian shores"
            
            species_counts = indian_df['scientificName'].value_counts().head(top_n)
            
            fig = go.Figure(data=[
                go.Bar(
                    y=species_counts.index[::-1],
                    x=species_counts.values[::-1],
                    orientation='h',
                    marker=dict(color='#FF6B35'),
                    text=species_counts.values[::-1],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title=f'🇮🇳 Top {top_n} Species in Indian Waters',
                xaxis_title='Number of Observations',
                yaxis_title='Species',
                height=600,
                showlegend=False
            )
            
            return fig, f"✅ {len(indian_df):,} total observations in Indian waters"
            
        except Exception as e:
            return None, f"❌ Error: {str(e)}"
    
    # ==================== HABITAT & ENVIRONMENT ====================
    
    def create_habitat_distribution(self) -> Tuple[Optional[go.Figure], str]:
        """Habitat distribution analysis"""
        if self.df is None or 'habitat' not in self.df.columns:
            return None, "❌ habitat column not found"
        
        try:
            habitat_counts = self.df['habitat'].value_counts().head(15)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=habitat_counts.index,
                    y=habitat_counts.values,
                    marker=dict(color='seagreen'),
                    text=habitat_counts.values,
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title='Habitat Distribution',
                xaxis_title='Habitat Type',
                yaxis_title='Number of Observations',
                xaxis_tickangle=-45,
                height=500,
                showlegend=False
            )
            
            return fig, f"✅ {len(habitat_counts)} habitat types"
            
        except Exception as e:
            return None, f"❌ Error: {str(e)}"
    
    def create_environmental_analysis(self) -> Tuple[Optional[go.Figure], str]:
        """Environmental conditions analysis (Temperature, Salinity, Depth)"""
        if self.df is None:
            return None, "❌ No data loaded"
        
        try:
            env_cols = ['temperature', 'salinity', 'depth']
            available = [col for col in env_cols if col in self.df.columns]
            
            if not available:
                return None, "❌ No environmental columns found"
            
            fig = make_subplots(
                rows=1, cols=len(available),
                subplot_titles=available
            )
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            
            for idx, col in enumerate(available, 1):
                data = self.df[col].dropna()
                
                fig.add_trace(
                    go.Histogram(
                        x=data,
                        name=col,
                        marker_color=colors[idx-1],
                        showlegend=False
                    ),
                    row=1, col=idx
                )
            
            fig.update_layout(
                title='Environmental Conditions Distribution',
                height=400,
                showlegend=False
            )
            
            return fig, f"✅ Analyzed {len(available)} environmental parameters"
            
        except Exception as e:
            return None, f"❌ Error: {str(e)}"
    
    # ==================== DEPTH ANALYSIS ====================
    
    def create_depth_distribution(self) -> Tuple[Optional[go.Figure], str]:
        """Depth distribution analysis"""
        if self.df is None or 'depth' not in self.df.columns:
            return None, "❌ depth column not found"
        
        try:
            depth_data = self.df['depth'].dropna()
            
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=depth_data,
                nbinsx=50,
                marker_color='steelblue',
                name='Depth Distribution'
            ))
            
            fig.update_layout(
                title='Depth Distribution of Marine Observations',
                xaxis_title='Depth (meters)',
                yaxis_title='Frequency',
                height=500,
                showlegend=False
            )
            
            return fig, f"✅ Range: {depth_data.min():.1f}m to {depth_data.max():.1f}m"
            
        except Exception as e:
            return None, f"❌ Error: {str(e)}"
    
    def species_richness_depth(self, bins: int = 10) -> Tuple[Optional[go.Figure], str]:
        """Species richness vs depth"""
        if self.df is None:
            return None, "❌ No data loaded"
        
        if 'depth' not in self.df.columns or 'scientificName' not in self.df.columns:
            return None, "❌ Required columns not found"
        
        try:
            df_clean = self.df[['depth', 'scientificName']].dropna()
            
            if len(df_clean) == 0:
                return None, "❌ No valid data"
            
            df_clean['depth_bin'] = pd.cut(df_clean['depth'], bins=bins)
            richness = df_clean.groupby('depth_bin')['scientificName'].nunique().reset_index()
            richness['depth_midpoint'] = richness['depth_bin'].apply(lambda x: x.mid)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=richness['depth_midpoint'],
                y=richness['scientificName'],
                mode='lines+markers',
                line=dict(color='forestgreen', width=3),
                marker=dict(size=10, color='darkgreen'),
                name='Species Richness'
            ))
            
            fig.update_layout(
                title='Species Richness by Depth',
                xaxis_title='Depth (meters)',
                yaxis_title='Number of Unique Species',
                height=500,
                showlegend=False
            )
            
            return fig, f"✅ Analyzed {bins} depth bins"
            
        except Exception as e:
            return None, f"❌ Error: {str(e)}"
    
    def abundance_vs_depth(self) -> Tuple[Optional[go.Figure], str]:
        """Abundance vs depth scatter plot"""
        if self.df is None:
            return None, "❌ No data loaded"
        
        if 'depth' not in self.df.columns or 'individualCount' not in self.df.columns:
            return None, "❌ Required columns not found"
        
        try:
            df_clean = self.df[['depth', 'individualCount']].dropna()
            
            # Sample if too many points
            if len(df_clean) > 10000:
                df_clean = df_clean.sample(10000, random_state=42)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_clean['depth'],
                y=df_clean['individualCount'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=df_clean['depth'],
                    colorscale='Viridis',
                    showscale=True,
                    opacity=0.6
                ),
                text=df_clean['individualCount'],
                name='Observations'
            ))
            
            fig.update_layout(
                title='Individual Count vs Depth',
                xaxis_title='Depth (meters)',
                yaxis_title='Individual Count',
                height=500,
                showlegend=False
            )
            
            return fig, f"✅ Plotted {len(df_clean):,} observations"
            
        except Exception as e:
            return None, f"❌ Error: {str(e)}"
    
    # ==================== CORRELATION ANALYSIS ====================
    
    def create_correlation_heatmap(self) -> Tuple[Optional[go.Figure], str]:
        """Correlation matrix heatmap"""
        if self.df is None:
            return None, "❌ No data loaded"
        
        try:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                return None, "❌ Insufficient numeric columns"
            
            # Select relevant columns
            relevant_cols = [col for col in ['depth', 'temperature', 'salinity', 
                           'individualCount', 'decimalLatitude', 'decimalLongitude']
                           if col in numeric_cols]
            
            if len(relevant_cols) < 2:
                relevant_cols = numeric_cols[:10]  # Take first 10
            
            corr_matrix = self.df[relevant_cols].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title="Correlation")
            ))
            
            fig.update_layout(
                title='Correlation Matrix',
                height=600,
                xaxis_tickangle=-45
            )
            
            return fig, f"✅ Analyzed {len(relevant_cols)} variables"
            
        except Exception as e:
            return None, f"❌ Error: {str(e)}"
    
    # ==================== TEMPORAL ANALYSIS ====================
    
    def temporal_trends(self) -> Tuple[Optional[go.Figure], str]:
        """Temporal trends analysis"""
        if self.df is None or 'year' not in self.df.columns:
            return None, "❌ year column not found"
        
        try:
            yearly_stats = self.df.groupby('year').agg({
                'occurrenceID': 'count',
                'scientificName': 'nunique'
            }).reset_index()
            yearly_stats.columns = ['year', 'observations', 'species_count']
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Scatter(
                    x=yearly_stats['year'],
                    y=yearly_stats['observations'],
                    name="Observations",
                    line=dict(color='steelblue', width=3),
                    mode='lines+markers'
                ),
                secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(
                    x=yearly_stats['year'],
                    y=yearly_stats['species_count'],
                    name="Species Count",
                    line=dict(color='coral', width=3),
                    mode='lines+markers'
                ),
                secondary_y=True
            )
            
            fig.update_xaxes(title_text="Year")
            fig.update_yaxes(title_text="Number of Observations", secondary_y=False)
            fig.update_yaxes(title_text="Number of Species", secondary_y=True)
            
            fig.update_layout(
                title='Temporal Trends in Marine Observations',
                height=500,
                hovermode='x unified'
            )
            
            return fig, f"✅ Analyzed {len(yearly_stats)} years"
            
        except Exception as e:
            return None, f"❌ Error: {str(e)}"
    
    def create_seasonal_patterns(self) -> Tuple[Optional[go.Figure], str]:
        """Seasonal patterns analysis"""
        if self.df is None or 'month' not in self.df.columns:
            return None, "❌ month column not found"
        
        try:
            df_seasonal = self.df.copy()
            df_seasonal['season'] = df_seasonal['month'].apply(get_season)
            
            seasonal_stats = df_seasonal.groupby('season').agg({
                'occurrenceID': 'count',
                'scientificName': 'nunique'
            }).reset_index()
            seasonal_stats.columns = ['season', 'observations', 'species_count']
            
            # Order seasons
            season_order = ['Winter', 'Spring', 'Summer', 'Fall']
            seasonal_stats['season'] = pd.Categorical(
                seasonal_stats['season'], 
                categories=season_order, 
                ordered=True
            )
            seasonal_stats = seasonal_stats.sort_values('season')
            
            fig = make_subplots(rows=1, cols=2, subplot_titles=['Observations by Season', 'Species Richness by Season'])
            
            fig.add_trace(
                go.Bar(
                    x=seasonal_stats['season'],
                    y=seasonal_stats['observations'],
                    marker_color='skyblue',
                    text=seasonal_stats['observations'],
                    textposition='auto',
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=seasonal_stats['season'],
                    y=seasonal_stats['species_count'],
                    marker_color='seagreen',
                    text=seasonal_stats['species_count'],
                    textposition='auto',
                    showlegend=False
                ),
                row=1, col=2
            )
            
            fig.update_xaxes(title_text="Season", row=1, col=1)
            fig.update_xaxes(title_text="Season", row=1, col=2)
            fig.update_yaxes(title_text="Observations", row=1, col=1)
            fig.update_yaxes(title_text="Species Count", row=1, col=2)
            
            fig.update_layout(
                title='Seasonal Patterns',
                height=500,
                showlegend=False
            )
            
            return fig, "✅ Seasonal analysis complete"
            
        except Exception as e:
            return None, f"❌ Error: {str(e)}"
    
    # ==================== DEMOGRAPHIC ANALYSIS ====================
    
    def create_life_stage_distribution(self) -> Tuple[Optional[go.Figure], str]:
        """Life stage distribution"""
        if self.df is None or 'lifeStage' not in self.df.columns:
            return None, "❌ lifeStage column not found"
        
        try:
            life_stage_counts = self.df['lifeStage'].value_counts()
            
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{'type': 'domain'}, {'type': 'bar'}]],
                subplot_titles=['Distribution', 'Counts']
            )
            
            fig.add_trace(
                go.Pie(
                    labels=life_stage_counts.index,
                    values=life_stage_counts.values,
                    hole=0.3
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=life_stage_counts.index,
                    y=life_stage_counts.values,
                    marker_color='mediumpurple',
                    text=life_stage_counts.values,
                    textposition='auto'
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                title='Life Stage Distribution',
                height=500,
                showlegend=False
            )
            
            return fig, f"✅ {len(life_stage_counts)} life stages"
            
        except Exception as e:
            return None, f"❌ Error: {str(e)}"
    
    # ==================== ADVANCED ANALYTICS ====================
    
    def create_pca_analysis(self) -> Tuple[Optional[go.Figure], str]:
        """PCA analysis"""
        if self.df is None:
            return None, "❌ No data loaded"
        
        try:
            env_cols = ['depth', 'temperature', 'salinity', 'bathymetry',
                       'decimalLatitude', 'decimalLongitude']
            available_cols = [col for col in env_cols if col in self.df.columns]
            
            if len(available_cols) < 3:
                return None, "❌ Insufficient numeric columns for PCA"
            
            env_df = self.df[available_cols].dropna()
            
            if len(env_df) < 10:
                return None, "❌ Insufficient data points"
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(env_df)
            
            pca = PCA()
            pca_components = pca.fit_transform(X_scaled)
            
            explained_var = pca.explained_variance_ratio_
            cumulative_var = np.cumsum(explained_var)
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=['Scree Plot', 'PC1 vs PC2']
            )
            
            # Scree plot
            fig.add_trace(
                go.Bar(
                    x=list(range(1, len(explained_var) + 1)),
                    y=explained_var,
                    name='Individual',
                    marker_color='steelblue'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(cumulative_var) + 1)),
                    y=cumulative_var,
                    name='Cumulative',
                    line=dict(color='red', width=2),
                    mode='lines+markers'
                ),
                row=1, col=1
            )
            
            # Biplot
            sample_size = min(5000, len(pca_components))
            indices = np.random.choice(len(pca_components), sample_size, replace=False)
            
            fig.add_trace(
                go.Scatter(
                    x=pca_components[indices, 0],
                    y=pca_components[indices, 1],
                    mode='markers',
                    marker=dict(size=3, color='steelblue', opacity=0.5),
                    showlegend=False
                ),
                row=1, col=2
            )
            
            fig.update_xaxes(title_text="Principal Component", row=1, col=1)
            fig.update_yaxes(title_text="Variance Explained", row=1, col=1)
            fig.update_xaxes(title_text=f"PC1 ({explained_var[0]*100:.1f}%)", row=1, col=2)
            fig.update_yaxes(title_text=f"PC2 ({explained_var[1]*100:.1f}%)", row=1, col=2)
            
            fig.update_layout(
                title='Principal Component Analysis',
                height=500
            )
            
            msg = f"✅ Variance explained by first 3 PCs: {cumulative_var[2]*100:.2f}%"
            return fig, msg
            
        except Exception as e:
            return None, f"❌ Error: {str(e)}"
    
    def multivariate_unmixing(self, feature_cols: list, n_components: int = 3) -> Tuple[Optional[go.Figure], Optional[go.Figure], str]:
        """ICA and Factor Analysis unmixing"""
        if self.df is None:
            return None, None, "❌ No data loaded"
        
        if not feature_cols or len(feature_cols) < 3:
            return None, None, "❌ Select at least 3 features"
        
        try:
            df_features = self.df[feature_cols].dropna()
            
            if len(df_features) < 10:
                return None, None, "❌ Insufficient data"
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_features)
            
            # ICA
            ica = FastICA(n_components=n_components, random_state=42, max_iter=1000)
            ica_components = ica.fit_transform(X_scaled)
            
            # Factor Analysis
            fa = FactorAnalysis(n_components=n_components, random_state=42)
            fa_components = fa.fit_transform(X_scaled)
            
            # ICA plot
            fig_ica = go.Figure()
            for i in range(n_components):
                fig_ica.add_trace(go.Scatter(
                    y=ica_components[:min(1000, len(ica_components)), i],
                    mode='lines',
                    name=f'IC{i+1}',
                    line=dict(width=1)
                ))
            
            fig_ica.update_layout(
                title='Independent Component Analysis',
                xaxis_title='Sample Index',
                yaxis_title='Component Value',
                height=400
            )
            
            # Factor loadings plot
            loadings = pd.DataFrame(
                fa.components_.T,
                columns=[f'Factor{i+1}' for i in range(n_components)],
                index=feature_cols
            )
            
            fig_fa = go.Figure(data=go.Heatmap(
                z=loadings.values,
                x=loadings.columns,
                y=loadings.index,
                colorscale='RdBu',
                zmid=0,
                text=np.round(loadings.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10}
            ))
            
            fig_fa.update_layout(
                title='Factor Analysis Loadings',
                height=400,
                xaxis_tickangle=-45
            )
            
            return fig_ica, fig_fa, f"✅ Analyzed {len(feature_cols)} features"
            
        except Exception as e:
            return None, None, f"❌ Error: {str(e)}"
