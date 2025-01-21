import streamlit as st
from typing import Dict, List, Optional, Any, Union, Set, Tuple, Callable
from datetime import datetime
import re
import numpy as np

from core import config
from core.exceptions import ValidationError
from core.state_manager import state_manager
from utils import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution

class UIValidator:
    """Handle UI input validation."""
    
    def __init__(self):
        self.validators: Dict[str, Dict[str, Callable]] = {
            'text': {},
            'numeric': {},
            'date': {},
            'file': {},
            'selection': {}
        }
        self.validation_history: List[Dict[str, Any]] = []
        
    @monitor_performance
    @handle_exceptions(ValidationError)
    def register_validator(
        self,
        validator_type: str,
        validator_name: str,
        validator_func: Callable,
        description: Optional[str] = None
    ) -> None:
        """Register a validator function."""
        if validator_type not in self.validators:
            raise ValidationError(f"Invalid validator type: {validator_type}")
            
        self.validators[validator_type][validator_name] = {
            'function': validator_func,
            'description': description or validator_func.__doc__,
            'registered_at': datetime.now().isoformat()
        }
    
    @monitor_performance
    def validate_text(
        self,
        text: str,
        validators: List[str] = None,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        """Validate text input."""
        if validators is None:
            validators = ['required']  # Default validator
            
        for validator_name in validators:
            if validator_name in self.validators['text']:
                validator = self.validators['text'][validator_name]['function']
                try:
                    if not validator(text, **kwargs):
                        return False, f"Validation failed: {validator_name}"
                except Exception as e:
                    return False, f"Validation error: {str(e)}"
        
        return True, None
    
    @monitor_performance
    def validate_numeric(
        self,
        value: Union[int, float],
        validators: List[str] = None,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        """Validate numeric input."""
        if validators is None:
            validators = ['range']  # Default validator
            
        for validator_name in validators:
            if validator_name in self.validators['numeric']:
                validator = self.validators['numeric'][validator_name]['function']
                try:
                    if not validator(value, **kwargs):
                        return False, f"Validation failed: {validator_name}"
                except Exception as e:
                    return False, f"Validation error: {str(e)}"
        
        return True, None
    
    @monitor_performance
    def validate_date(
        self,
        date: datetime,
        validators: List[str] = None,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        """Validate date input."""
        if validators is None:
            validators = ['range']  # Default validator
            
        for validator_name in validators:
            if validator_name in self.validators['date']:
                validator = self.validators['date'][validator_name]['function']
                try:
                    if not validator(date, **kwargs):
                        return False, f"Validation failed: {validator_name}"
                except Exception as e:
                    return False, f"Validation error: {str(e)}"
        
        return True, None
    
    @monitor_performance
    def validate_file(
        self,
        file: Any,
        validators: List[str] = None,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        """Validate file input."""
        if validators is None:
            validators = ['type', 'size']  # Default validators
            
        for validator_name in validators:
            if validator_name in self.validators['file']:
                validator = self.validators['file'][validator_name]['function']
                try:
                    if not validator(file, **kwargs):
                        return False, f"Validation failed: {validator_name}"
                except Exception as e:
                    return False, f"Validation error: {str(e)}"
        
        return True, None
    
    @monitor_performance
    def validate_selection(
        self,
        selection: Union[Any, List[Any]],
        validators: List[str] = None,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        """Validate selection input."""
        if validators is None:
            validators = ['required']  # Default validator
            
        for validator_name in validators:
            if validator_name in self.validators['selection']:
                validator = self.validators['selection'][validator_name]['function']
                try:
                    if not validator(selection, **kwargs):
                        return False, f"Validation failed: {validator_name}"
                except Exception as e:
                    return False, f"Validation error: {str(e)}"
        
        return True, None
    
    # Default validators
    @staticmethod
    def _validate_required(value: Any) -> bool:
        """Validate required field."""
        if isinstance(value, str):
            return bool(value and not value.isspace())
        return value is not None
    
    @staticmethod
    def _validate_length(value: str, min_length: int = 0, max_length: Optional[int] = None) -> bool:
        """Validate text length."""
        length = len(value)
        if length < min_length:
            return False
        if max_length is not None and length > max_length:
            return False
        return True
    
    @staticmethod
    def _validate_regex(value: str, pattern: str) -> bool:
        """Validate text against regex pattern."""
        return bool(re.match(pattern, value))
    
    @staticmethod
    def _validate_numeric_range(
        value: Union[int, float],
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> bool:
        """Validate numeric range."""
        if min_value is not None and value < min_value:
            return False
        if max_value is not None and value > max_value:
            return False
        return True
    
    @staticmethod
    def _validate_date_range(
        date: datetime,
        min_date: Optional[datetime] = None,
        max_date: Optional[datetime] = None
    ) -> bool:
        """Validate date range."""
        if min_date is not None and date < min_date:
            return False
        if max_date is not None and date > max_date:
            return False
        return True
    
    @staticmethod
    def _validate_file_type(file: Any, allowed_types: List[str]) -> bool:
        """Validate file type."""
        return any(file.name.lower().endswith(t.lower()) for t in allowed_types)
    
    @staticmethod
    def _validate_file_size(file: Any, max_size: int) -> bool:
        """Validate file size."""
        return file.size <= max_size
    
    @staticmethod
    def _validate_selection_required(selection: Union[Any, List[Any]]) -> bool:
        """Validate selection is made."""
        if isinstance(selection, list):
            return len(selection) > 0
        return selection is not None
    
    def _register_default_validators(self) -> None:
        """Register default validators."""
        # Text validators
        self.register_validator('text', 'required', self._validate_required)
        self.register_validator('text', 'length', self._validate_length)
        self.register_validator('text', 'regex', self._validate_regex)
        
        # Numeric validators
        self.register_validator('numeric', 'range', self._validate_numeric_range)
        
        # Date validators
        self.register_validator('date', 'range', self._validate_date_range)
        
        # File validators
        self.register_validator('file', 'type', self._validate_file_type)
        self.register_validator('file', 'size', self._validate_file_size)
        
        # Selection validators
        self.register_validator('selection', 'required', self._validate_selection_required)
    
    def _track_validation(
        self,
        input_type: str,
        validation_name: str,
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """Track validation execution."""
        record = {
            'input_type': input_type,
            'validation': validation_name,
            'success': success,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
        
        self.validation_history.append(record)
        
        # Update state
        state_manager.set_state(
            f'ui.validation.history.{len(self.validation_history)}',
            record
        )

# Create global UI validator instance
ui_validator = UIValidator()

# Register default validators
ui_validator._register_default_validators()