import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

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
        self.default_height = config.ui.chart_height
        self.default_width = config.ui.chart_width
        
        # Mapping of plot types to creation methods
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
            # Metrics visualization types
            'metric_comparison': self._create_metric_comparison_plot,
            'residual_plot': self._create_residual_plot,
            'learning_curve': self._create_learning_curve_plot,
            'validation_curve': self._create_validation_curve_plot,
            'feature_importance': self._create_feature_importance_plot,
            'confusion_matrix': self._create_confusion_matrix_plot,
            'roc_curve': self._create_roc_curve_plot,
            'pr_curve': self._create_pr_curve_plot,
            'calibration_curve': self._create_calibration_curve_plot
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
        if plot_type not in self.plot_types:
            raise VisualizationError(f"Unsupported plot type: {plot_type}")
        
        # Add default styling
        kwargs.setdefault('height', self.default_height)
        kwargs.setdefault('width', self.default_width)
        kwargs.setdefault('template', 'plotly_white')
        
        # Create plot
        fig = self.plot_types[plot_type](data, **kwargs)
        
        # Apply theme
        fig = style_manager.apply_theme_to_figure(fig)
        
        # Store plot
        plot_id = f"plot_{len(self.plot_history)}"
        self.figures[plot_id] = fig
        
        # Record plot creation
        self._record_plot(plot_type, kwargs)
        
        return fig
    
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
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create line plot."""
        fig = px.line(
            data,
            x=x,
            y=y,
            title=title,
            **kwargs
        )
        
        fig.update_layout(
            title_x=0.5,
            showlegend=True
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
        y: str,
        x: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create box plot."""
        fig = px.box(
            data,
            x=x,
            y=y,
            title=title,
            **kwargs
        )
        
        fig.update_layout(
            title_x=0.5,
            showlegend=True if x else False
        )
        
        return fig
    
    @monitor_performance
    def _create_violin_plot(
        self,
        data: pd.DataFrame,
        y: str,
        x: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create violin plot."""
        fig = px.violin(
            data,
            x=x,
            y=y,
            title=title,
            **kwargs
        )
        
        fig.update_layout(
            title_x=0.5,
            showlegend=True if x else False
        )
        
        return fig
    
    @monitor_performance
    def _create_heatmap_plot(
        self,
        data: pd.DataFrame,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create heatmap plot."""
        fig = px.imshow(
            data,
            title=title,
            **kwargs
        )
        
        fig.update_layout(
            title_x=0.5
        )
        
        return fig
    
    @monitor_performance
    def _create_pie_plot(
        self,
        data: pd.DataFrame,
        values: str,
        names: str,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create pie plot."""
        fig = px.pie(
            data,
            values=values,
            names=names,
            title=title,
            **kwargs
        )
        
        fig.update_layout(
            title_x=0.5,
            showlegend=True
        )
        
        return fig
    
    @monitor_performance
    def _create_density_plot(
        self,
        data: pd.DataFrame,
        x: str,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create density plot."""
        fig = ff.create_distplot(
            [data[x].dropna()],
            [x],
            show_hist=False,
            show_rug=False
        )
        
        fig.update_layout(
            title=title,
            title_x=0.5,
            showlegend=False,
            **kwargs
        )
        
        return fig
    
    # Metric Visualization Methods (from metrics/visualizer.py)
    @monitor_performance
    def _create_metric_comparison_plot(
        self,
        metrics: Dict[str, float],
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create metric comparison plot."""
        fig = px.bar(
            pd.DataFrame({
                'Metric': list(metrics.keys()),
                'Value': list(metrics.values())
            }),
            x='Metric',
            y='Value',
            title=title or 'Metric Comparison'
        )
        
        fig.update_layout(
            title_x=0.5,
            bargap=0.2
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
        
        fig = px.scatter(
            pd.DataFrame({
                'Predicted': y_pred,
                'Residuals': residuals
            }),
            x='Predicted',
            y='Residuals',
            title=title or 'Residual Plot'
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        
        fig.update_layout(
            title_x=0.5
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
        """Create learning curve plot."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=train_scores.mean(axis=1),
            mode='lines+markers',
            name='Training Score',
            error_y=dict(
                type='data',
                array=train_scores.std(axis=1),
                visible=True
            )
        ))
        
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=val_scores.mean(axis=1),
            mode='lines+markers',
            name='Validation Score',
            error_y=dict(
                type='data',
                array=val_scores.std(axis=1),
                visible=True
            )
        ))
        
        fig.update_layout(
            title=title or 'Learning Curves',
            title_x=0.5,
            xaxis_title="Training Examples",
            yaxis_title="Score"
        )
        
        return fig
    
    @monitor_performance
    def _create_validation_curve_plot(
        self,
        param_range: np.ndarray,
        train_scores: np.ndarray,
        val_scores: np.ndarray,
        param_name: str,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create validation curve plot."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=param_range,
            y=train_scores.mean(axis=1),
            mode='lines+markers',
            name='Training Score',
            error_y=dict(
                type='data',
                array=train_scores.std(axis=1),
                visible=True
            )
        ))
        
        fig.add_trace(go.Scatter(
            x=param_range,
            y=val_scores.mean(axis=1),
            mode='lines+markers',
            name='Validation Score',
            error_y=dict(
                type='data',
                array=val_scores.std(axis=1),
                visible=True
            )
        ))
        
        fig.update_layout(
            title=title or f'Validation Curve ({param_name})',
            title_x=0.5,
            xaxis_title=param_name,
            yaxis_title="Score"
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
        """Create feature importance plot."""
        fig = px.bar(
            pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance_values
            }).sort_values('Importance', ascending=True),
            x='Importance',
            y='Feature',
            orientation='h',
            title=title or 'Feature Importance'
        )
        
        fig.update_layout(
            title_x=0.5
        )
        
        return fig
    
    @monitor_performance
    def _create_confusion_matrix_plot(
        self,
        confusion_matrix: np.ndarray,
        labels: Optional[List[str]] = None,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create confusion matrix plot."""
        fig = px.imshow(
            confusion_matrix,
            labels=dict(
                x="Predicted",
                y="True",
                color="Count"
            ),
            x=labels,
            y=labels,
            title=title or 'Confusion Matrix'
        )
        
        fig.update_layout(
            title_x=0.5
        )
        
        return fig
    
    @monitor_performance
    def _create_roc_curve_plot(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc_score: float,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create ROC curve plot."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'ROC curve (AUC = {auc_score:.3f})'
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            line=dict(dash='dash'),
            name='Random'
        ))
        
        fig.update_layout(
            title=title or 'ROC Curve',
            title_x=0.5,
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate'
        )
        
        return fig

    @monitor_performance
    def _create_pr_curve_plot(
        self,
        precision: np.ndarray,
        recall: np.ndarray,
        average_precision: float,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create precision-recall curve plot."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recall,
            y=precision,
            mode='lines',
            name=f'PR curve (AP = {average_precision:.3f})'
        ))
        
        fig.update_layout(
            title=title or 'Precision-Recall Curve',
            title_x=0.5,
            xaxis_title='Recall',
            yaxis_title='Precision',
            yaxis=dict(range=[0, 1.05]),
            xaxis=dict(range=[0, 1.05])
        )
        
        return fig

    @monitor_performance
    def _create_calibration_curve_plot(
        self,
        prob_true: np.ndarray,
        prob_pred: np.ndarray,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """Create calibration curve plot."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=prob_pred,
            y=prob_true,
            mode='lines',
            name='Calibration curve'
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            line=dict(dash='dash'),
            name='Perfect calibration'
        ))
        
        fig.update_layout(
            title=title or 'Calibration Curve',
            title_x=0.5,
            xaxis_title='Mean predicted probability',
            yaxis_title='Fraction of positives',
            yaxis=dict(range=[0, 1.05]),
            xaxis=dict(range=[0, 1.05])
        )
        
        return fig

    def _record_plot(
        self,
        plot_type: str,
        plot_args: Dict[str, Any]
    ) -> None:
        """Record plot creation in history."""
        record = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'plot_type': plot_type,
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
    def load_plot(
        self,
        path: Path
    ) -> go.Figure:
        """Load plot from file."""
        try:
            if not path.exists():
                raise VisualizationError(f"Plot file not found: {path}")
            
            if path.suffix == '.html':
                fig = go.Figure(px.load_figure(path))
            elif path.suffix == '.json':
                with open(path, 'r') as f:
                    fig = go.Figure(json.load(f))
            else:
                raise VisualizationError(f"Unsupported file format: {path.suffix}")
            
            return fig
            
        except Exception as e:
            raise VisualizationError(
                f"Error loading plot: {str(e)}",
                details={'path': str(path)}
            ) from e

    def get_plot_summary(self) -> Dict[str, Any]:
        """Get summary of plot creation history."""
        return {
            'total_plots': len(self.plot_history),
            'plot_types': list(set(
                record['plot_type'] for record in self.plot_history
            )),
            'recent_plots': self.plot_history[-5:]
        }

# Create global plotter instance
plotter = Plotter()