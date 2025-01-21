import streamlit as st
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime
import re

from core import config
from core.exceptions import InputError
from core.state_manager import state_manager
from utils import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution
from utils.validators import validate_input

class InputManager:
    """Handle input component creation and validation."""
    
    def __init__(self):
        self.input_history: List[Dict[str, Any]] = []
        self.active_inputs: Dict[str, Dict[str, Any]] = {}
        self.input_validators: Dict[str, Callable] = {}
        self.default_values: Dict[str, Any] = {}
        
    @monitor_performance
    @handle_exceptions(InputError)
    def create_text_input(
        self,
        label: str,
        key: str,
        validators: Optional[List[Callable]] = None,
        **kwargs
    ) -> Optional[str]:
        """Create text input with validation."""
        value = st.text_input(
            label,
            value=self.default_values.get(key, ""),
            key=key,
            **kwargs
        )
        
        if value:
            # Validate input if validators provided
            if validators:
                for validator in validators:
                    if not validator(value):
                        st.error(f"Invalid input for {label}")
                        return None
                        
            # Track input
            self._track_input(key, "text", value)
            
            return value
        return None
    
    @monitor_performance
    @handle_exceptions(InputError)
    def create_numeric_input(
        self,
        label: str,
        key: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        step: Optional[float] = None,
        validators: Optional[List[Callable]] = None,
        **kwargs
    ) -> Optional[float]:
        """Create numeric input with validation."""
        value = st.number_input(
            label,
            min_value=min_value,
            max_value=max_value,
            value=self.default_values.get(key, 0.0),
            step=step,
            key=key,
            **kwargs
        )
        
        # Validate input if validators provided
        if validators:
            for validator in validators:
                if not validator(value):
                    st.error(f"Invalid input for {label}")
                    return None
        
        # Track input
        self._track_input(key, "numeric", value)
        
        return value
    
    @monitor_performance
    def create_date_input(
        self,
        label: str,
        key: str,
        min_date: Optional[datetime] = None,
        max_date: Optional[datetime] = None,
        validators: Optional[List[Callable]] = None,
        **kwargs
    ) -> Optional[datetime]:
        """Create date input with validation."""
        value = st.date_input(
            label,
            value=self.default_values.get(key, datetime.now()),
            min_value=min_date,
            max_value=max_date,
            key=key,
            **kwargs
        )
        
        # Validate input if validators provided
        if validators:
            for validator in validators:
                if not validator(value):
                    st.error(f"Invalid input for {label}")
                    return None
        
        # Track input
        self._track_input(key, "date", value)
        
        return value
    
    @monitor_performance
    def create_file_upload(
        self,
        label: str,
        key: str,
        allowed_types: Optional[List[str]] = None,
        max_size: Optional[int] = None,
        **kwargs
    ) -> Optional[Any]:
        """Create file upload input with validation."""
        file = st.file_uploader(
            label,
            type=allowed_types,
            key=key,
            **kwargs
        )
        
        if file:
            # Validate file size if max_size provided
            if max_size and file.size > max_size:
                st.error(f"File size exceeds maximum limit of {max_size} bytes")
                return None
            
            # Track input
            self._track_input(key, "file", file.name)
            
            return file
        return None
    
    @monitor_performance
    def create_selection_input(
        self,
        label: str,
        options: List[Any],
        key: str,
        multiselect: bool = False,
        default: Optional[Any] = None,
        validators: Optional[List[Callable]] = None,
        **kwargs
    ) -> Optional[Union[Any, List[Any]]]:
        """Create selection input."""
        if multiselect:
            value = st.multiselect(
                label,
                options=options,
                default=self.default_values.get(key, default or []),
                key=key,
                **kwargs
            )
        else:
            value = st.selectbox(
                label,
                options=options,
                index=options.index(self.default_values.get(key, default))
                if default in options else 0,
                key=key,
                **kwargs
            )
        
        # Validate input if validators provided
        if validators:
            for validator in validators:
                if not validator(value):
                    st.error(f"Invalid selection for {label}")
                    return None
        
        # Track input
        self._track_input(
            key,
            "multiselect" if multiselect else "select",
            value
        )
        
        return value
    
    @monitor_performance
    def create_slider_input(
        self,
        label: str,
        min_value: float,
        max_value: float,
        key: str,
        step: Optional[float] = None,
        validators: Optional[List[Callable]] = None,
        **kwargs
    ) -> Optional[float]:
        """Create slider input."""
        value = st.slider(
            label,
            min_value=min_value,
            max_value=max_value,
            value=self.default_values.get(key, min_value),
            step=step,
            key=key,
            **kwargs
        )
        
        # Validate input if validators provided
        if validators:
            for validator in validators:
                if not validator(value):
                    st.error(f"Invalid value for {label}")
                    return None
        
        # Track input
        self._track_input(key, "slider", value)
        
        return value
    
    @monitor_performance
    def register_validator(
        self,
        input_type: str,
        validator: Callable
    ) -> None:
        """Register validator for input type."""
        self.input_validators[input_type] = validator
    
    @monitor_performance
    def set_default_value(
        self,
        key: str,
        value: Any
    ) -> None:
        """Set default value for input."""
        self.default_values[key] = value
    
    @monitor_performance
    def clear_input(
        self,
        key: str
    ) -> None:
        """Clear input value."""
        if key in st.session_state:
            del st.session_state[key]
        if key in self.default_values:
            del self.default_values[key]
    
    @monitor_performance
    def get_input_value(
        self,
        key: str
    ) -> Optional[Any]:
        """Get current input value."""
        return st.session_state.get(key)
    
    def _track_input(
        self,
        key: str,
        input_type: str,
        value: Any
    ) -> None:
        """Track input interaction."""
        record = {
            'key': key,
            'type': input_type,
            'value': value,
            'timestamp': datetime.now().isoformat()
        }
        
        self.input_history.append(record)
        self.active_inputs[key] = record
        
        # Update state
        state_manager.set_state(
            f'ui.inputs.{key}',
            record
        )
    
    @monitor_performance
    def get_input_history(
        self,
        key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get input history."""
        if key:
            return [
                record for record in self.input_history
                if record['key'] == key
            ]
        return self.input_history
    
    @monitor_performance
    def validate_all_inputs(self) -> bool:
        """Validate all active inputs."""
        for key, input_info in self.active_inputs.items():
            input_type = input_info['type']
            value = input_info['value']
            
            # Check type-specific validator
            if input_type in self.input_validators:
                if not self.input_validators[input_type](value):
                    return False
        
        return True

# Create global input manager instance
input_manager = InputManager()

# Register default validators
input_manager.register_validator(
    'text',
    lambda x: bool(x and not x.isspace())
)
input_manager.register_validator(
    'numeric',
    lambda x: isinstance(x, (int, float))
)
input_manager.register_validator(
    'date',
    lambda x: isinstance(x, datetime)
)