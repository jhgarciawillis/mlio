import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import re

from core import config
from core.exceptions import QualityError
from core.state_manager import state_manager
from utils import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution
from visualization import plotter

class QualityAnalyzer:
    """Handle data quality analysis operations."""
    
    def __init__(self):
        self.quality_scores: Dict[str, float] = {}
        self.quality_issues: Dict[str, List[Dict[str, Any]]] = {}
        self.quality_history: List[Dict[str, Any]] = []
        self.validation_rules: Dict[str, Dict[str, Any]] = {}
        
    @monitor_performance
    @handle_exceptions(QualityError)
    def analyze_data_quality(
        self,
        data: pd.DataFrame,
        rules: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive data quality analysis."""
        try:
            # Get default rules if none provided
            if rules is None:
                rules = self._get_default_rules()
            
            analysis_id = f"quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            results = {
                'completeness': self.analyze_completeness(data),
                'accuracy': self.analyze_accuracy(data, rules.get('accuracy', {})),
                'consistency': self.analyze_consistency(data, rules.get('consistency', {})),
                'validity': self.analyze_validity(data, rules.get('validity', {})),
                'timeliness': self.analyze_timeliness(data, rules.get('timeliness', {})),
                'uniqueness': self.analyze_uniqueness(data),
                'integrity': self.analyze_integrity(data, rules.get('integrity', {}))
            }
            
            # Calculate overall quality score
            quality_score = self._calculate_quality_score(results)
            results['overall_score'] = quality_score
            
            # Store results
            self.quality_scores[analysis_id] = quality_score
            self.quality_issues[analysis_id] = self._extract_quality_issues(results)
            
            # Record analysis
            self._record_quality_analysis(analysis_id, results)
            
            return results
            
        except Exception as e:
            raise QualityError(
                f"Error performing quality analysis: {str(e)}"
            ) from e
    
    @monitor_performance
    def analyze_completeness(
        self,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze data completeness."""
        completeness = {
            'overall': float((1 - data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100),
            'columns': {}
        }
        
        for col in data.columns:
            non_null_count = data[col].count()
            total_count = len(data)
            completeness_score = (non_null_count / total_count) * 100
            
            completeness['columns'][col] = {
                'score': float(completeness_score),
                'missing_count': int(total_count - non_null_count),
                'missing_percentage': float((total_count - non_null_count) / total_count * 100)
            }
        
        # Create completeness visualization
        completeness_data = pd.DataFrame({
            'Column': list(completeness['columns'].keys()),
            'Completeness': [info['score'] for info in completeness['columns'].values()]
        })
        completeness['visualization'] = plotter.create_plot(
            'bar',
            data=completeness_data,
            x='Column',
            y='Completeness',
            title='Data Completeness by Column'
        )
        
        return completeness
    
    @monitor_performance
    def analyze_accuracy(
        self,
        data: pd.DataFrame,
        rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze data accuracy."""
        accuracy = {
            'columns': {},
            'issues': []
        }
        
        for col, rule in rules.items():
            if col in data.columns:
                valid_values = rule.get('valid_values', None)
                value_range = rule.get('value_range', None)
                pattern = rule.get('pattern', None)
                
                accuracy['columns'][col] = self._check_column_accuracy(
                    data[col], valid_values, value_range, pattern
                )
        
        # Calculate overall accuracy
        column_scores = [
            info['score'] 
            for info in accuracy['columns'].values()
        ]
        accuracy['overall'] = float(np.mean(column_scores)) if column_scores else 0.0
        
        # Create accuracy visualization
        accuracy_data = pd.DataFrame({
            'Column': list(accuracy['columns'].keys()),
            'Accuracy': [info['score'] for info in accuracy['columns'].values()]
        })
        accuracy['visualization'] = plotter.create_plot(
            'bar',
            data=accuracy_data,
            x='Column',
            y='Accuracy',
            title='Data Accuracy by Column'
        )
        
        return accuracy
    
    @monitor_performance
    def analyze_consistency(
        self,
        data: pd.DataFrame,
        rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze data consistency."""
        consistency = {
            'columns': {},
            'relationships': {},
            'formats': {},
            'issues': []
        }
        
        # Check column value consistency
        for col in data.columns:
            consistency['columns'][col] = self._check_column_consistency(data[col])
        
        # Check relationship consistency
        for relation_name, relation_rule in rules.get('relationships', {}).items():
            consistency['relationships'][relation_name] = self._check_relationship_consistency(
                data, relation_rule
            )
        
        # Check format consistency
        for col, format_rule in rules.get('formats', {}).items():
            if col in data.columns:
                consistency['formats'][col] = self._check_format_consistency(
                    data[col], format_rule
                )
        
        # Calculate overall consistency
        scores = (
            [info['score'] for info in consistency['columns'].values()] +
            [info['score'] for info in consistency['relationships'].values()] +
            [info['score'] for info in consistency['formats'].values()]
        )
        consistency['overall'] = float(np.mean(scores)) if scores else 0.0
        
        return consistency
    
    @monitor_performance
    def analyze_validity(
        self,
        data: pd.DataFrame,
        rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze data validity."""
        validity = {
            'columns': {},
            'issues': []
        }
        
        for col, rule in rules.items():
            if col in data.columns:
                validity['columns'][col] = self._check_column_validity(
                    data[col], rule
                )
        
        # Calculate overall validity
        column_scores = [
            info['score'] 
            for info in validity['columns'].values()
        ]
        validity['overall'] = float(np.mean(column_scores)) if column_scores else 0.0
        
        return validity
    
    @monitor_performance
    def analyze_timeliness(
        self,
        data: pd.DataFrame,
        rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze data timeliness."""
        timeliness = {
            'columns': {},
            'issues': []
        }
        
        for col, rule in rules.items():
            if col in data.columns and pd.api.types.is_datetime64_any_dtype(data[col]):
                timeliness['columns'][col] = self._check_column_timeliness(
                    data[col], rule
                )
        
        # Calculate overall timeliness
        column_scores = [
            info['score'] 
            for info in timeliness['columns'].values()
        ]
        timeliness['overall'] = float(np.mean(column_scores)) if column_scores else 0.0
        
        return timeliness
    
    @monitor_performance
    def analyze_uniqueness(
        self,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze data uniqueness."""
        uniqueness = {
            'columns': {},
            'duplicate_rows': {
                'count': int(data.duplicated().sum()),
                'percentage': float(data.duplicated().sum() / len(data) * 100)
            }
        }
        
        for col in data.columns:
            unique_count = data[col].nunique()
            total_count = len(data)
            uniqueness_score = (unique_count / total_count) * 100
            
            uniqueness['columns'][col] = {
                'score': float(uniqueness_score),
                'unique_count': int(unique_count),
                'duplicate_count': int(total_count - unique_count),
                'duplicate_percentage': float((total_count - unique_count) / total_count * 100)
            }
        
        # Calculate overall uniqueness
        column_scores = [
            info['score'] 
            for info in uniqueness['columns'].values()
        ]
        uniqueness['overall'] = float(np.mean(column_scores)) if column_scores else 0.0
        
        return uniqueness
    
    @monitor_performance
    def analyze_integrity(
        self,
        data: pd.DataFrame,
        rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze data integrity."""
        integrity = {
            'referential': {},
            'domain': {},
            'entity': {},
            'issues': []
        }
        
        # Check referential integrity
        for ref_name, ref_rule in rules.get('referential', {}).items():
            integrity['referential'][ref_name] = self._check_referential_integrity(
                data, ref_rule
            )
        
        # Check domain integrity
        for col, domain_rule in rules.get('domain', {}).items():
            if col in data.columns:
                integrity['domain'][col] = self._check_domain_integrity(
                    data[col], domain_rule
                )
        
        # Check entity integrity
        for entity_name, entity_rule in rules.get('entity', {}).items():
            integrity['entity'][entity_name] = self._check_entity_integrity(
                data, entity_rule
            )
        
        # Calculate overall integrity
        scores = (
            [info['score'] for info in integrity['referential'].values()] +
            [info['score'] for info in integrity['domain'].values()] +
            [info['score'] for info in integrity['entity'].values()]
        )
        integrity['overall'] = float(np.mean(scores)) if scores else 0.0
        
        return integrity
    
    def _calculate_quality_score(
        self,
        results: Dict[str, Any]
    ) -> float:
        """Calculate overall quality score."""
        weights = {
            'completeness': 0.2,
            'accuracy': 0.2,
            'consistency': 0.15,
            'validity': 0.15,
            'timeliness': 0.1,
            'uniqueness': 0.1,
            'integrity': 0.1
        }
        
        weighted_scores = [
            results[metric]['overall'] * weights[metric]
            for metric in weights.keys()
            if metric in results and 'overall' in results[metric]
        ]
        
        return float(sum(weighted_scores))
    
    def _record_quality_analysis(
        self,
        analysis_id: str,
        results: Dict[str, Any]
    ) -> None:
        """Record quality analysis in history."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'analysis_id': analysis_id,
            'results': results
        }
        
        self.quality_history.append(record)
        state_manager.set_state(
            f'analysis.quality.history.{len(self.quality_history)}',
            record
        )
    
    def _get_default_rules(self) -> Dict[str, Any]:
        """Get default quality rules."""
        return {
            'accuracy': {},
            'consistency': {
                'relationships': {},
                'formats': {}
            },
            'validity': {},
            'timeliness': {},
            'integrity': {
                'referential': {},
                'domain': {},
                'entity': {}
            }
        }

# Create global quality analyzer instance
quality_analyzer = QualityAnalyzer()