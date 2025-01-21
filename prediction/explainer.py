import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import shap
from lime import lime_tabular
from pdpbox import pdp
from sklearn.inspection import partial_dependence, permutation_importance

from core import config
from core.exceptions import ExplanationError
from core.state_manager import state_manager
from utils import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution
from visualization import plotter

class PredictionExplainer:
    """Handle prediction explanation and interpretability."""
    
    def __init__(self):
        self.explanation_history: List[Dict[str, Any]] = []
        self.explanation_results: Dict[str, Dict[str, Any]] = {}
        self.shap_explainer: Optional[Any] = None
        self.lime_explainer: Optional[Any] = None
        
    @monitor_performance
    @handle_exceptions(ExplanationError)
    def explain_predictions(
        self,
        model: Any,
        X: pd.DataFrame,
        explanation_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive prediction explanations."""
        try:
            explanation_id = f"explanation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if explanation_config is None:
                explanation_config = self._get_default_explanation_config()
            
            results = {}
            
            # SHAP explanations
            if explanation_config.get('shap', {}).get('enabled', True):
                results['shap'] = self._generate_shap_explanations(
                    model,
                    X,
                    explanation_config['shap']
                )
            
            # LIME explanations
            if explanation_config.get('lime', {}).get('enabled', True):
                results['lime'] = self._generate_lime_explanations(
                    model,
                    X,
                    explanation_config['lime']
                )
            
            # Partial Dependence Plots
            if explanation_config.get('pdp', {}).get('enabled', True):
                results['pdp'] = self._generate_pdp_explanations(
                    model,
                    X,
                    explanation_config['pdp']
                )
            
            # Feature Importance
            if explanation_config.get('feature_importance', {}).get('enabled', True):
                results['feature_importance'] = self._analyze_feature_importance(
                    model,
                    X,
                    explanation_config['feature_importance']
                )
            
            # Store results
            self.explanation_results[explanation_id] = results
            
            # Record explanation
            self._record_explanation(explanation_id, explanation_config)
            
            return results
            
        except Exception as e:
            raise ExplanationError(
                f"Error generating explanations: {str(e)}"
            ) from e
    
    @monitor_performance
    def _generate_shap_explanations(
        self,
        model: Any,
        X: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate SHAP-based explanations."""
        # Initialize SHAP explainer if not already done
        if self.shap_explainer is None:
            self.shap_explainer = shap.Explainer(model, X)
        
        # Calculate SHAP values
        shap_values = self.shap_explainer(X)
        
        results = {
            'shap_values': shap_values,
            'summary': {
                'mean_abs_shap': np.mean(np.abs(shap_values.values), axis=0).tolist(),
                'feature_importance_ranking': pd.DataFrame({
                    'feature': X.columns,
                    'importance': np.mean(np.abs(shap_values.values), axis=0)
                }).sort_values('importance', ascending=False).to_dict('records')
            }
        }
        
        # Generate visualizations
        if config.get('generate_plots', True):
            results['visualizations'] = self._create_shap_plots(shap_values, X)
        
        return results
    
    @monitor_performance
    def _generate_lime_explanations(
        self,
        model: Any,
        X: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate LIME-based explanations."""
        if self.lime_explainer is None:
            self.lime_explainer = lime_tabular.LimeTabularExplainer(
                X.values,
                feature_names=X.columns,
                mode='regression'
            )
        
        results = {
            'explanations': [],
            'summary': {}
        }
        
        # Generate explanations for samples
        n_samples = config.get('n_samples', 5)
        for idx in range(min(n_samples, len(X))):
            exp = self.lime_explainer.explain_instance(
                X.iloc[idx].values,
                model.predict
            )
            results['explanations'].append({
                'sample_idx': idx,
                'features': exp.as_list(),
                'local_prediction': float(exp.local_pred[0]),
                'score': float(exp.score)
            })
        
        # Generate summary
        results['summary'] = self._summarize_lime_explanations(
            results['explanations']
        )
        
        return results
    
    @monitor_performance
    def _generate_pdp_explanations(
        self,
        model: Any,
        X: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate Partial Dependence Plot explanations."""
        results = {
            'single_feature_pdp': {},
            'interaction_pdp': {}
        }
        
        # Generate single feature PDPs
        features = config.get('features', X.columns.tolist())
        for feature in features:
            pdp_result = pdp.pdp_isolate(
                model=model,
                dataset=X,
                model_features=X.columns,
                feature=feature
            )
            results['single_feature_pdp'][feature] = {
                'values': pdp_result.pdp.tolist(),
                'feature_values': pdp_result.feature_values.tolist()
            }
        
        # Generate interaction PDPs if requested
        if config.get('interactions', True):
            feature_pairs = config.get('feature_pairs', [])
            for feat1, feat2 in feature_pairs:
                pdp_interact = pdp.pdp_interact(
                    model=model,
                    dataset=X,
                    model_features=X.columns,
                    features=[feat1, feat2]
                )
                results['interaction_pdp'][f'{feat1}_{feat2}'] = {
                    'pdp_values': pdp_interact.pdp.tolist(),
                    'feature1_values': pdp_interact.feature_values[0].tolist(),
                    'feature2_values': pdp_interact.feature_values[1].tolist()
                }
        
        return results
    
    @monitor_performance
    def _analyze_feature_importance(
        self,
        model: Any,
        X: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze feature importance."""
        results = {}
        
        # Permutation importance
        if config.get('permutation', True):
            perm_importance = permutation_importance(
                model,
                X,
                random_state=config.get('random_state', 42)
            )
            results['permutation_importance'] = {
                'importances_mean': perm_importance.importances_mean.tolist(),
                'importances_std': perm_importance.importances_std.tolist(),
                'feature_ranking': pd.DataFrame({
                    'feature': X.columns,
                    'importance': perm_importance.importances_mean
                }).sort_values('importance', ascending=False).to_dict('records')
            }
        
        # Model-specific feature importance if available
        if hasattr(model, 'feature_importances_'):
            results['model_importance'] = {
                'importances': model.feature_importances_.tolist(),
                'feature_ranking': pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False).to_dict('records')
            }
        
        return results
    
    def _create_shap_plots(
        self,
        shap_values: Any,
        X: pd.DataFrame
    ) -> Dict[str, Any]:
        """Create SHAP-based visualization plots."""
        plots = {}
        
        # Summary plot
        plots['summary_plot'] = plotter.create_plot(
            'shap_summary',
            shap_values=shap_values,
            features=X,
            title='SHAP Summary Plot'
        )
        
        # Dependence plots for top features
        top_features = np.argsort(np.mean(np.abs(shap_values.values), axis=0))[-5:]
        for feature_idx in top_features:
            feature_name = X.columns[feature_idx]
            plots[f'dependence_{feature_name}'] = plotter.create_plot(
                'shap_dependence',
                shap_values=shap_values,
                features=X,
                feature_idx=feature_idx,
                title=f'SHAP Dependence Plot: {feature_name}'
            )
        
        return plots
    
    def _summarize_lime_explanations(
        self,
        explanations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Summarize LIME explanations."""
        feature_importance = {}
        for exp in explanations:
            for feature, importance in exp['features']:
                if feature not in feature_importance:
                    feature_importance[feature] = []
                feature_importance[feature].append(abs(importance))
        
        return {
            'average_feature_importance': {
                feature: float(np.mean(importances))
                for feature, importances in feature_importance.items()
            },
            'feature_stability': {
                feature: float(np.std(importances))
                for feature, importances in feature_importance.items()
            }
        }
    
    def _get_default_explanation_config(self) -> Dict[str, Any]:
        """Get default explanation configuration."""
        return {
            'shap': {
                'enabled': True,
                'generate_plots': True
            },
            'lime': {
                'enabled': True,
                'n_samples': 5
            },
            'pdp': {
                'enabled': True,
                'interactions': True,
                'feature_pairs': []
            },
            'feature_importance': {
                'enabled': True,
                'permutation': True,
                'random_state': 42
            }
        }
    
    def _record_explanation(
        self,
        explanation_id: str,
        explanation_config: Dict[str, Any]
    ) -> None:
        """Record explanation in history."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'explanation_id': explanation_id,
            'configuration': explanation_config
        }
        
        self.explanation_history.append(record)
        state_manager.set_state(
            f'prediction.explanation.history.{len(self.explanation_history)}',
            record
        )

# Create global prediction explainer instance
prediction_explainer = PredictionExplainer()