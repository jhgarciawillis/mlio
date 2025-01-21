import streamlit as st
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime
import json

from core import config
from core.exceptions import FormError
from core.state_manager import state_manager
from utils import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution
from utils.validators import validate_input

class FormManager:
    """Handle form creation and management."""
    
    def __init__(self):
        self.form_history: List[Dict[str, Any]] = []
        self.active_forms: Dict[str, Dict[str, Any]] = {}
        self.form_validators: Dict[str, Callable] = {}
        self.form_callbacks: Dict[str, Callable] = {}
        
    @monitor_performance
    @handle_exceptions(FormError)
    def create_form(
        self,
        title: str,
        fields: List[Dict[str, Any]],
        key: str,
        validator: Optional[Callable] = None,
        on_submit: Optional[Callable] = None
    ) -> Optional[Dict[str, Any]]:
        """Create form with specified fields."""
        with st.form(key):
            st.subheader(title)
            
            field_values = {}
            
            # Create form fields
            for field in fields:
                value = self._create_form_field(field)
                field_values[field['key']] = value
            
            # Submit button
            submitted = st.form_submit_button("Submit")
            
            if submitted:
                # Validate form if validator provided
                if validator:
                    is_valid, message = validator(field_values)
                    if not is_valid:
                        st.error(message)
                        return None
                
                # Execute callback if provided
                if on_submit:
                    try:
                        on_submit(field_values)
                    except Exception as e:
                        st.error(f"Error processing form: {str(e)}")
                        return None
                
                # Track form submission
                self._track_form_submission(key, field_values)
                
                return field_values
            
            return None
    
    @monitor_performance
    def create_model_config_form(
        self,
        model_params: Dict[str, Dict[str, Any]],
        key: str = "model_config"
    ) -> Optional[Dict[str, Any]]:
        """Create form for model configuration."""
        with st.form(key):
            st.subheader("Model Configuration")
            
            param_values = {}
            
            # Create parameter fields
            for param_name, param_config in model_params.items():
                value = self._create_parameter_field(
                    param_name,
                    param_config
                )
                param_values[param_name] = value
            
            # Submit button
            submitted = st.form_submit_button("Apply Configuration")
            
            if submitted:
                # Validate parameters
                is_valid = self._validate_model_params(param_values)
                if not is_valid:
                    return None
                
                # Track form submission
                self._track_form_submission(key, param_values)
                
                return param_values
            
            return None
    
    @monitor_performance
    def create_feature_selection_form(
        self,
        features: List[str],
        key: str = "feature_selection"
    ) -> Optional[Dict[str, List[str]]]:
        """Create form for feature selection."""
        with st.form(key):
            st.subheader("Feature Selection")
            
            # Feature type selection
            numeric_features = st.multiselect(
                "Select Numeric Features",
                options=features,
                key=f"{key}_numeric"
            )
            
            categorical_features = st.multiselect(
                "Select Categorical Features",
                options=[f for f in features if f not in numeric_features],
                key=f"{key}_categorical"
            )
            
            # Target feature selection
            remaining_features = [
                f for f in features 
                if f not in numeric_features and f not in categorical_features
            ]
            target_feature = st.selectbox(
                "Select Target Feature",
                options=remaining_features,
                key=f"{key}_target"
            )
            
            # Submit button
            submitted = st.form_submit_button("Confirm Selection")
            
            if submitted:
                selected_features = {
                    'numeric': numeric_features,
                    'categorical': categorical_features,
                    'target': target_feature
                }
                
                # Validate selection
                is_valid = self._validate_feature_selection(selected_features)
                if not is_valid:
                    return None
                
                # Track form submission
                self._track_form_submission(key, selected_features)
                
                return selected_features
            
            return None
    
    def _create_form_field(
        self,
        field_config: Dict[str, Any]
    ) -> Any:
        """Create individual form field."""
        field_type = field_config['type']
        
        if field_type == 'text':
            return st.text_input(
                field_config['label'],
                key=field_config['key']
            )
        elif field_type == 'number':
            return st.number_input(
                field_config['label'],
                min_value=field_config.get('min'),
                max_value=field_config.get('max'),
                value=field_config.get('default', 0),
                step=field_config.get('step', 1),
                key=field_config['key']
            )
        elif field_type == 'select':
            return st.selectbox(
                field_config['label'],
                options=field_config['options'],
                index=field_config.get('default_index', 0),
                key=field_config['key']
            )
        elif field_type == 'multiselect':
            return st.multiselect(
                field_config['label'],
                options=field_config['options'],
                default=field_config.get('default', []),
                key=field_config['key']
            )
        elif field_type == 'slider':
            return st.slider(
                field_config['label'],
                min_value=field_config['min'],
                max_value=field_config['max'],
                value=field_config.get('default'),
                step=field_config.get('step', 1),
                key=field_config['key']
            )
        elif field_type == 'checkbox':
            return st.checkbox(
                field_config['label'],
                value=field_config.get('default', False),
                key=field_config['key']
            )
        elif field_type == 'radio':
            return st.radio(
                field_config['label'],
                options=field_config['options'],
                index=field_config.get('default_index', 0),
                key=field_config['key']
            )
        elif field_type == 'text_area':
            return st.text_area(
                field_config['label'],
                value=field_config.get('default', ''),
                height=field_config.get('height', 100),
                key=field_config['key']
            )
    
    def _create_parameter_field(
        self,
        param_name: str,
        param_config: Dict[str, Any]
    ) -> Any:
        """Create parameter input field."""
        param_type = param_config.get('type', 'number')
        
        if param_type == 'number':
            return st.number_input(
                param_name,
                min_value=param_config.get('min'),
                max_value=param_config.get('max'),
                value=param_config.get('default', 0),
                step=param_config.get('step', 1)
            )
        elif param_type == 'select':
            return st.selectbox(
                param_name,
                options=param_config['options'],
                index=param_config.get('default_index', 0)
            )
        elif param_type == 'checkbox':
            return st.checkbox(
                param_name,
                value=param_config.get('default', False)
            )
    
    def _validate_model_params(
        self,
        params: Dict[str, Any]
    ) -> bool:
        """Validate model parameters."""
        try:
            # Validate numeric parameters
            for param_name, value in params.items():
                if isinstance(value, (int, float)):
                    if not validate_input(value, param_name):
                        st.error(f"Invalid value for {param_name}")
                        return False
            return True
        except Exception as e:
            st.error(f"Error validating parameters: {str(e)}")
            return False
    
    def _validate_feature_selection(
        self,
        selection: Dict[str, List[str]]
    ) -> bool:
        """Validate feature selection."""
        try:
            # Check for overlap between feature types
            numeric = set(selection['numeric'])
            categorical = set(selection['categorical'])
            target = {selection['target']}
            
            if len(numeric & categorical) > 0:
                st.error("Features cannot be both numeric and categorical")
                return False
            
            if len((numeric | categorical) & target) > 0:
                st.error("Target feature cannot be in numeric or categorical features")
                return False
            
            return True
        except Exception as e:
            st.error(f"Error validating feature selection: {str(e)}")
            return False
    
    def _track_form_submission(
        self,
        form_key: str,
        values: Dict[str, Any]
    ) -> None:
        """Track form submission."""
        submission = {
            'form_key': form_key,
            'values': values,
            'timestamp': datetime.now().isoformat()
        }
        
        self.form_history.append(submission)
        
        # Update state
        state_manager.set_state(
            f'ui.forms.{form_key}',
            {
                'last_submission': submission,
                'submission_count': len([
                    s for s in self.form_history 
                    if s['form_key'] == form_key
                ])
            }
        )

# Create global form manager instance
form_manager = FormManager()