import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
from datetime import datetime
from docx import Document
from docx.shared import Inches

from core import config
from core.exceptions import PredictionError
from core.state_manager import state_manager
from utils import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution
from visualization import plotter
from prediction import evaluator, explainer

class PredictionReporter:
    """Handle prediction reporting operations."""
    
    def __init__(self):
        self.report_history: List[Dict[str, Any]] = []
        self.reports: Dict[str, Dict[str, Any]] = {}
        self.report_templates: Dict[str, Dict[str, Any]] = {}
        
    @monitor_performance
    @handle_exceptions(PredictionError)
    def generate_report(
        self,
        predictions: Union[pd.Series, np.ndarray],
        true_values: Optional[Union[pd.Series, np.ndarray]] = None,
        report_config: Optional[Dict[str, Any]] = None,
        template_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate prediction report."""
        try:
            report_id = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if report_config is None:
                report_config = self._get_default_report_config()
            
            # Get template if specified
            template = self.report_templates.get(template_name, {}) if template_name else {}
            
            report = {
                'metadata': {
                    'report_id': report_id,
                    'timestamp': datetime.now().isoformat(),
                    'config': report_config
                },
                'predictions': {
                    'values': predictions.tolist() if isinstance(predictions, (pd.Series, np.ndarray)) else predictions,
                    'statistics': self._calculate_prediction_statistics(predictions)
                }
            }

            # Add actual values comparison if provided
            if true_values is not None:
                report['evaluation'] = self._evaluate_predictions(predictions, true_values)
            
            # Generate visualizations
            report['visualizations'] = self._create_visualizations(
                predictions, true_values
            )
            
            # Store report
            self.reports[report_id] = report
            
            # Record report generation
            self._record_report(report_id, report_config)
            
            return report
            
        except Exception as e:
            raise PredictionError(
                f"Error generating prediction report: {str(e)}"
            ) from e
    
    @monitor_performance
    def _calculate_prediction_statistics(
        self,
        predictions: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """Calculate basic statistics for predictions."""
        return {
            'mean': float(np.mean(predictions)),
            'std': float(np.std(predictions)),
            'min': float(np.min(predictions)),
            'max': float(np.max(predictions)),
            'median': float(np.median(predictions)),
            'q25': float(np.percentile(predictions, 25)),
            'q75': float(np.percentile(predictions, 75))
        }
    
    @monitor_performance
    def _evaluate_predictions(
        self,
        predictions: Union[pd.Series, np.ndarray],
        true_values: Union[pd.Series, np.ndarray]
    ) -> Dict[str, Any]:
        """Evaluate predictions against true values."""
        # Calculate metrics
        metrics = metrics_calculator.calculate_metrics(true_values, predictions)
        
        # Calculate error statistics
        errors = true_values - predictions
        error_stats = {
            'mean_error': float(np.mean(errors)),
            'std_error': float(np.std(errors)),
            'mean_absolute_error': float(np.mean(np.abs(errors))),
            'median_absolute_error': float(np.median(np.abs(errors)))
        }
        
        return {
            'metrics': metrics,
            'error_statistics': error_stats
        }
    
    @monitor_performance
    def _create_visualizations(
        self,
        predictions: Union[pd.Series, np.ndarray],
        true_values: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """Create prediction visualizations."""
        visualizations = {}
        
        # Prediction distribution
        visualizations['prediction_distribution'] = plotter.create_plot(
            'histogram',
            data=pd.DataFrame({'predictions': predictions}),
            x='predictions',
            title='Prediction Distribution'
        )
        
        if true_values is not None:
            # Actual vs Predicted
            visualizations['actual_vs_predicted'] = plotter.create_plot(
                'scatter',
                data=pd.DataFrame({
                    'Actual': true_values,
                    'Predicted': predictions
                }),
                x='Actual',
                y='Predicted',
                title='Actual vs Predicted Values'
            )
            
            # Error distribution
            errors = true_values - predictions
            visualizations['error_distribution'] = plotter.create_plot(
                'histogram',
                data=pd.DataFrame({'errors': errors}),
                x='errors',
                title='Prediction Error Distribution'
            )
            
            # Error vs Predicted
            visualizations['error_vs_predicted'] = plotter.create_plot(
                'scatter',
                data=pd.DataFrame({
                    'Predicted': predictions,
                    'Error': errors
                }),
                x='Predicted',
                y='Error',
                title='Error vs Predicted Values'
            )
        
        return visualizations
    
    @monitor_performance
    def export_report(
        self,
        report_id: str,
        format: str = 'json',
        path: Optional[Path] = None
    ) -> Path:
        """Export report in specified format."""
        if report_id not in self.reports:
            raise PredictionError(f"Report not found: {report_id}")
        
        if path is None:
            path = config.directories.reports / 'predictions'
            path.mkdir(parents=True, exist_ok=True)
        
        report = self.reports[report_id]
        export_path = path / f"prediction_report_{report_id}.{format}"
        
        if format == 'json':
            with open(export_path, 'w') as f:
                json.dump(report, f, indent=4)
        elif format == 'html':
            self._export_html_report(report, export_path)
        elif format == 'pdf':
            self._export_pdf_report(report, export_path)
        else:
            raise PredictionError(f"Unsupported export format: {format}")
        
        return export_path
    
    def _export_html_report(
        self,
        report: Dict[str, Any],
        path: Path
    ) -> None:
        """Export report as HTML."""
        # This is a placeholder for HTML report generation
        raise NotImplementedError("HTML export not yet implemented")
    
    def _export_pdf_report(
        self,
        report: Dict[str, Any],
        path: Path
    ) -> None:
        """Export report as PDF."""
        # This is a placeholder for PDF report generation
        raise NotImplementedError("PDF export not yet implemented")
    
    def _get_default_report_config(self) -> Dict[str, Any]:
        """Get default report configuration."""
        return {
            'include_statistics': True,
            'include_evaluation': True,
            'include_visualizations': True,
            'visualization_config': {
                'width': config.ui.chart_width,
                'height': config.ui.chart_height
            }
        }
    
    def _record_report(
        self,
        report_id: str,
        config: Dict[str, Any]
    ) -> None:
        """Record report generation in history."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'report_id': report_id,
            'configuration': config
        }
        
        self.report_history.append(record)
        state_manager.set_state(
            f'prediction.reports.history.{len(self.report_history)}',
            record
        )

# Create global prediction reporter instance
prediction_reporter = PredictionReporter()