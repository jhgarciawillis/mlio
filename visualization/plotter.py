import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from core import config
from core.exceptions import VisualizationError
from core.state_manager import state_manager
from utils import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution
from visualization.styler import style_manager

class Plotter:
    """Handle all visualization operations."""
    
    def __init__(self):
        self.plot_history: List[Dict[str, Any]] = []
        self.figures: Dict[str, go.Figure] = {}
        
        # Plot type mappings
        self.plot_types = {
            'scatter': self._create_scatter_plot,
            'line': self._create_line_plot,
            'bar': self._create_bar_plot,
            'histogram': self._create_histogram_plot,
            'box': self._create_box_plot,
            'violin': self._create_violin_plot,
            'heatmap': self._create_heatmap_plot,
            'pie': self._create_pie_plot,
            'density': self._create_density_plot,
            'scatter_matrix': self._create_scatter_matrix,
            'parallel_coordinates': self._create_parallel_coordinates,
            
            # Specialized plots
            'silhouette': self._create_silhouette_plot,
            'cluster_map': self._create_cluster_map,
            'feature_importance': self._create_feature_importance_plot,
            'residuals': self._create_residual_plot,
            'qq': self._create_qq_plot,
            'learning_curve': self._create_learning_curve_plot
        }
    
    @monitor_performance
    @handle_exceptions(VisualizationError)
    def create_plot(
        self,
        plot_type: str,
        data: Any,
        **kwargs
    ) -> go.Figure:
        """Create plot based on type."""
        try:
            # Record operation start
            state_monitor.record_operation_start(
                'plot_creation',
                'visualization',
                {'plot_type': plot_type}
            )
            
            # Validate plot type
            if plot_type not in self.plot_types:
                raise VisualizationError(
                    f"Unsupported plot type: {plot_type}",
                    details={'available_types': list(self.plot_types.keys())}
                )
            
            # Create plot
            fig = self.plot_types[plot_type](data, **kwargs)
            
            # Apply theme
            fig = style_manager.apply_theme_to_figure(fig)
            
            # Store figure
            plot_id = f"plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.figures[plot_id] = fig
            
            # Record operation completion
            state_monitor.record_operation_end(
                'plot_creation',
                'completed',
                {'plot_id': plot_id}
            )
            
            # Record plot creation
            self._record_plot(plot_type, plot_id, kwargs)
            
            return fig
            
        except Exception as e:
            state_monitor.record_operation_end(
                'plot_creation',
                'failed',
                {'error': str(e)}
            )
            raise VisualizationError(f"Error creating plot: {str(e)}") from e
    
    @monitor_performance
    def _create_scatter_plot(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        color: Optional[str] = None,
        size: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create scatter plot."""
        fig = px.scatter(
            data,
            x=x,
            y=y,
            color=color,
            size=size,
            title=title,
            **kwargs
        )
        
        fig.update_layout(
            title_x=0.5,
            legend_title_text=color if color else None
        )
        
        return fig

@monitor_performance
    def _create_line_plot(
        self,
        data: pd.DataFrame,
        x: str,
        y: Union[str, List[str]],
        color: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create line plot."""
        fig = px.line(
            data,
            x=x,
            y=y,
            color=color,
            title=title,
            **kwargs
        )
        
        fig.update_layout(
            title_x=0.5,
            showlegend=True if color else False
        )
        
        return fig
    
    @monitor_performance
    def _create_bar_plot(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        color: Optional[str] = None,
        title: Optional[str] = None,
        orientation: str = 'v',
        **kwargs
    ) -> go.Figure:
        """Create bar plot."""
        fig = px.bar(
            data,
            x=x,
            y=y,
            color=color,
            title=title,
            orientation=orientation,
            **kwargs
        )
        
        fig.update_layout(
            title_x=0.5,
            bargap=0.2
        )
        
        return fig
    
    @monitor_performance
    def _create_histogram_plot(
        self,
        data: pd.DataFrame,
        x: str,
        nbins: Optional[int] = None,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create histogram plot."""
        fig = px.histogram(
            data,
            x=x,
            nbins=nbins,
            title=title,
            **kwargs
        )
        
        fig.update_layout(
            title_x=0.5,
            bargap=0.1
        )
        
        return fig
    
    @monitor_performance
    def _create_box_plot(
        self,
        data: pd.DataFrame,
        x: Optional[str] = None,
        y: str = None,
        color: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create box plot."""
        fig = px.box(
            data,
            x=x,
            y=y,
            color=color,
            title=title,
            **kwargs
        )
        
        fig.update_layout(
            title_x=0.5
        )
        
        return fig
    
    @monitor_performance
    def _create_violin_plot(
        self,
        data: pd.DataFrame,
        x: Optional[str] = None,
        y: str = None,
        color: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create violin plot."""
        fig = px.violin(
            data,
            x=x,
            y=y,
            color=color,
            title=title,
            **kwargs
        )
        
        fig.update_layout(
            title_x=0.5
        )
        
        return fig
    
    @monitor_performance
    def _create_heatmap_plot(
        self,
        data: pd.DataFrame,
        title: Optional[str] = None,
        colorscale: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create heatmap plot."""
        fig = px.imshow(
            data,
            title=title,
            color_continuous_scale=colorscale or 'RdBu_r',
            **kwargs
        )
        
        fig.update_layout(
            title_x=0.5
        )
        
        return fig
    
    @monitor_performance
    def _create_silhouette_plot(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create silhouette plot."""
        from sklearn.metrics import silhouette_samples
        
        # Calculate silhouette scores
        silhouette_vals = silhouette_samples(data, labels)
        
        # Create figure
        fig = go.Figure()
        
        y_lower = 10
        for i in range(len(np.unique(labels))):
            cluster_silhouette_vals = silhouette_vals[labels == i]
            cluster_silhouette_vals.sort()
            
            size_cluster = len(cluster_silhouette_vals)
            y_upper = y_lower + size_cluster
            
            fig.add_trace(go.Scatter(
                x=cluster_silhouette_vals,
                y=np.arange(y_lower, y_upper),
                name=f'Cluster {i}',
                mode='lines',
                showlegend=True
            ))
            
            y_lower = y_upper + 10
        
        fig.update_layout(
            title=title or 'Silhouette Plot',
            title_x=0.5,
            xaxis_title='Silhouette Coefficient',
            yaxis_title='Cluster'
        )
        
        return fig
    
    @monitor_performance
    def _create_cluster_map(
        self,
        data: pd.DataFrame,
        labels: np.ndarray,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create cluster map visualization."""
        # Perform dimensionality reduction if needed
        if data.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            coords = pca.fit_transform(data)
        else:
            coords = data.values
        
        # Create scatter plot
        fig = go.Figure()
        
        for label in np.unique(labels):
            mask = labels == label
            fig.add_trace(go.Scatter(
                x=coords[mask, 0],
                y=coords[mask, 1],
                mode='markers',
                name=f'Cluster {label}',
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title=title or 'Cluster Map',
            title_x=0.5,
            xaxis_title='Component 1',
            yaxis_title='Component 2'
        )
        
        return fig
    
@monitor_performance
    def _create_feature_importance_plot(
        self,
        feature_names: List[str],
        importance_values: np.ndarray,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create feature importance visualization."""
        # Sort features by importance
        sorted_idx = np.argsort(importance_values)
        pos = np.arange(sorted_idx.shape[0]) + .5
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=importance_values[sorted_idx],
            y=[feature_names[i] for i in sorted_idx],
            orientation='h'
        ))
        
        fig.update_layout(
            title=title or 'Feature Importance',
            title_x=0.5,
            xaxis_title='Importance Score',
            yaxis_title='Feature'
        )
        
        return fig
    
    @monitor_performance
    def _create_residual_plot(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create residual plot."""
        residuals = y_true - y_pred
        
        fig = make_subplots(rows=2, cols=1)
        
        # Scatter plot of residuals vs predicted
        fig.add_trace(
            go.Scatter(
                x=y_pred,
                y=residuals,
                mode='markers',
                name='Residuals'
            ),
            row=1, col=1
        )
        
        # Histogram of residuals
        fig.add_trace(
            go.Histogram(
                x=residuals,
                name='Residual Distribution'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=title or 'Residual Analysis',
            title_x=0.5,
            showlegend=True
        )
        
        fig.update_xaxes(title_text='Predicted Values', row=1, col=1)
        fig.update_xaxes(title_text='Residuals', row=2, col=1)
        fig.update_yaxes(title_text='Residuals', row=1, col=1)
        fig.update_yaxes(title_text='Frequency', row=2, col=1)
        
        return fig
    
    @monitor_performance
    def _create_qq_plot(
        self,
        data: np.ndarray,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create Q-Q plot."""
        sorted_data = np.sort(data)
        theoretical_quantiles = stats.norm.ppf(
            np.linspace(0.01, 0.99, len(sorted_data))
        )
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=theoretical_quantiles,
            y=sorted_data,
            mode='markers',
            name='Q-Q Plot'
        ))
        
        # Add diagonal line
        min_val = min(theoretical_quantiles.min(), sorted_data.min())
        max_val = max(theoretical_quantiles.max(), sorted_data.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Reference Line',
            line=dict(dash='dash')
        ))
        
        fig.update_layout(
            title=title or 'Q-Q Plot',
            title_x=0.5,
            xaxis_title='Theoretical Quantiles',
            yaxis_title='Sample Quantiles'
        )
        
        return fig
    
    @monitor_performance
    def _create_learning_curve_plot(
        self,
        train_sizes: np.ndarray,
        train_scores: np.ndarray,
        val_scores: np.ndarray,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create learning curve visualization."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=np.mean(train_scores, axis=1),
            mode='lines+markers',
            name='Training Score',
            error_y=dict(
                type='data',
                array=np.std(train_scores, axis=1),
                visible=True
            )
        ))
        
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=np.mean(val_scores, axis=1),
            mode='lines+markers',
            name='Validation Score',
            error_y=dict(
                type='data',
                array=np.std(val_scores, axis=1),
                visible=True
            )
        ))
        
        fig.update_layout(
            title=title or 'Learning Curve',
            title_x=0.5,
            xaxis_title='Training Examples',
            yaxis_title='Score'
        )
        
        return fig
    
    def _record_plot(
        self,
        plot_type: str,
        plot_id: str,
        plot_args: Dict[str, Any]
    ) -> None:
        """Record plot creation in history."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'plot_type': plot_type,
            'plot_id': plot_id,
            'arguments': plot_args
        }
        
        self.plot_history.append(record)
        state_manager.set_state(
            f'visualization.history.{len(self.plot_history)}',
            record
        )
    
    @monitor_performance
    def save_plot(
        self,
        fig: go.Figure,
        path: Path,
        format: str = 'html',
        **kwargs
    ) -> None:
        """Save plot to file."""
        try:
            if format == 'html':
                fig.write_html(str(path), **kwargs)
            elif format == 'png':
                fig.write_image(str(path), **kwargs)
            elif format == 'json':
                fig.write_json(str(path), **kwargs)
            else:
                raise VisualizationError(f"Unsupported format: {format}")
                
            logger.info(f"Plot saved to {path}")
            
        except Exception as e:
            raise VisualizationError(
                f"Error saving plot: {str(e)}",
                details={'path': str(path), 'format': format}
            ) from e
    
    @monitor_performance
    def get_plot_history(
        self,
        plot_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get plot creation history."""
        if plot_type:
            return [
                record for record in self.plot_history
                if record['plot_type'] == plot_type
            ]
        return self.plot_history

# Create global plotter instance
plotter = Plotter()