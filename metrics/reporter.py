import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
from datetime import datetime
import plotly.io as pio
from docx import Document
from docx.shared import Inches

from core import config
from core.exceptions import MetricError
from core.state_manager import state_manager
from utils import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution
from metrics import calculator, evaluator
from visualization import plotter

class PerformanceReporter:
    """Handle performance reporting and result presentation."""
    
    def __init__(self):
        self.report_history: List[Dict[str, Any]] = []
        self.report_templates: Dict[str, Dict[str, Any]] = {}
        
    @monitor_performance
    @handle_exceptions(ReportingError)
    def generate_report(
        self,
        results: Dict[str, Any],
        report_config: Optional[Dict[str, Any]] = None,
        template_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate performance report."""
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
                'sections': {}
            }
            
            # Generate each section based on config
            if report_config.get('summary', True):
                report['sections']['summary'] = self._generate_summary_section(results)
            
            if report_config.get('detailed_metrics', True):
                report['sections']['metrics'] = self._generate_metrics_section(results)
            
            if report_config.get('analysis', True):
                report['sections']['analysis'] = self._generate_analysis_section(results)
            
            if report_config.get('visualizations', True):
                report['sections']['visualizations'] = self._generate_visualization_section(results)
            
            if report_config.get('recommendations', True):
                report['sections']['recommendations'] = self._generate_recommendations(results)
            
            # Record report generation
            self._record_report(report_id, report_config)
            
            return report
            
        except Exception as e:
            raise ReportingError(
                f"Error generating report: {str(e)}"
            ) from e
    
    @monitor_performance
    def export_report(
        self,
        report: Dict[str, Any],
        format: str = 'docx',
        output_path: Optional[Path] = None
    ) -> Path:
        """Export report in specified format."""
        if output_path is None:
            output_path = config.directories.reports
            output_path.mkdir(parents=True, exist_ok=True)
        
        export_funcs = {
            'docx': self._export_to_docx,
            'html': self._export_to_html,
            'pdf': self._export_to_pdf,
            'json': self._export_to_json
        }
        
        if format not in export_funcs:
            raise ReportingError(f"Unsupported export format: {format}")
        
        return export_funcs[format](report, output_path)
    
    def _generate_summary_section(
        self,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate summary section of the report."""
        metrics = results.get('metrics', {})
        analysis = results.get('analysis', {})
        
        summary = {
            'overall_performance': {
                'r2_score': metrics.get('r2', 'N/A'),
                'rmse': metrics.get('rmse', 'N/A'),
                'mae': metrics.get('mae', 'N/A')
            },
            'key_findings': self._extract_key_findings(analysis),
            'highlights': self._identify_highlights(results)
        }
        
        return summary
    
    def _generate_metrics_section(
        self,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate detailed metrics section."""
        metrics = results.get('metrics', {})
        
        return {
            'performance_metrics': metrics,
            'metric_descriptions': self._get_metric_descriptions(),
            'comparative_analysis': self._analyze_metric_relationships(metrics)
        }
    
    def _generate_analysis_section(
        self,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate analysis section."""
        analysis = results.get('analysis', {})
        
        return {
            'residual_analysis': analysis.get('residuals', {}),
            'error_distribution': analysis.get('error_distribution', {}),
            'performance_breakdown': analysis.get('performance_breakdown', {}),
            'interpretation': self._interpret_analysis_results(analysis)
        }
    
    def _generate_visualization_section(
        self,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate visualization section."""
        visualizations = results.get('visualizations', {})
        
        return {
            'plots': visualizations,
            'descriptions': self._generate_plot_descriptions(visualizations)
        }
    
    def _generate_recommendations(
        self,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate recommendations based on results."""
        recommendations = {
            'model_improvements': [],
            'data_quality': [],
            'monitoring': []
        }
        
        # Analyze metrics for model improvement recommendations
        metrics = results.get('metrics', {})
        if metrics.get('r2', 0) < 0.7:
            recommendations['model_improvements'].append(
                "Consider using more complex model architectures"
            )
        
        # Analyze residuals for data quality recommendations
        analysis = results.get('analysis', {})
        residuals = analysis.get('residuals', {})
        if residuals.get('normality', {}).get('is_normal', True) is False:
            recommendations['data_quality'].append(
                "Consider feature transformations to improve error distribution"
            )
        
        # Add monitoring recommendations
        recommendations['monitoring'].extend([
            "Implement regular model performance monitoring",
            "Set up alerts for significant performance degradation",
            "Track feature drift over time"
        ])
        
        return recommendations
    
    def _export_to_docx(
        self,
        report: Dict[str, Any],
        output_path: Path
    ) -> Path:
        """Export report to Word document."""
        doc = Document()
        
        # Title
        doc.add_heading('Performance Analysis Report', 0)
        
        # Summary Section
        if 'summary' in report['sections']:
            doc.add_heading('Summary', level=1)
            summary = report['sections']['summary']
            
            doc.add_paragraph('Overall Performance:')
            for metric, value in summary['overall_performance'].items():
                doc.add_paragraph(f"{metric}: {value}", style='List Bullet')
        
        # Metrics Section
        if 'metrics' in report['sections']:
            doc.add_heading('Detailed Metrics', level=1)
            metrics = report['sections']['metrics']
            
            for metric, value in metrics['performance_metrics'].items():
                doc.add_paragraph(f"{metric}: {value}")
        
        # Analysis Section
        if 'analysis' in report['sections']:
            doc.add_heading('Analysis', level=1)
            analysis = report['sections']['analysis']
            
            for section, content in analysis.items():
                doc.add_heading(section.replace('_', ' ').title(), level=2)
                doc.add_paragraph(str(content))
        
        # Save visualizations
        if 'visualizations' in report['sections']:
            doc.add_heading('Visualizations', level=1)
            for name, fig in report['sections']['visualizations']['plots'].items():
                img_path = output_path / f"{name}.png"
                fig.write_image(str(img_path))
                doc.add_picture(str(img_path), width=Inches(6))
                doc.add_paragraph(f"Figure: {name}")
        
        # Recommendations
        if 'recommendations' in report['sections']:
            doc.add_heading('Recommendations', level=1)
            recommendations = report['sections']['recommendations']
            
            for category, items in recommendations.items():
                doc.add_heading(category.replace('_', ' ').title(), level=2)
                for item in items:
                    doc.add_paragraph(item, style='List Bullet')
        
        # Save document
        file_path = output_path / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
        doc.save(str(file_path))
        
        return file_path
    
    def _export_to_html(
        self,
        report: Dict[str, Any],
        output_path: Path
    ) -> Path:
        """Export report to HTML."""
        # Implementation for HTML export
        raise NotImplementedError("HTML export not yet implemented")
    
    def _export_to_pdf(
        self,
        report: Dict[str, Any],
        output_path: Path
    ) -> Path:
        """Export report to PDF."""
        # Implementation for PDF export
        raise NotImplementedError("PDF export not yet implemented")
    
    def _export_to_json(
        self,
        report: Dict[str, Any],
        output_path: Path
    ) -> Path:
        """Export report to JSON."""
        file_path = output_path / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        return file_path
    
    def _record_report(
        self,
        report_id: str,
        report_config: Dict[str, Any]
    ) -> None:
        """Record report generation in history."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'report_id': report_id,
            'configuration': report_config
        }
        
        self.report_history.append(record)
        state_manager.set_state(
            f'metrics.reports.history.{len(self.report_history)}',
            record
        )
    
    def _get_default_report_config(self) -> Dict[str, Any]:
        """Get default report configuration."""
        return {
            'summary': True,
            'detailed_metrics': True,
            'analysis': True,
            'visualizations': True,
            'recommendations': True
        }

# Create global performance reporter instance
performance_reporter = PerformanceReporter()