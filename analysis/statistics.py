import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.tsa.seasonal import seasonal_decompose

from core import config
from core.exceptions import StatisticalError
from core.state_manager import state_manager
from utils import logger
from utils.decorators import monitor_performance, handle_exceptions, log_execution

class StatisticalAnalyzer:
    """Handle statistical analysis operations."""
    
    def __init__(self):
        self.test_results: Dict[str, Dict[str, Any]] = {}
        self.statistics_history: List[Dict[str, Any]] = []
        
    @monitor_performance
    @handle_exceptions(StatisticalError)
    def compare_groups(
        self,
        data: pd.DataFrame,
        group_column: str,
        value_column: str,
        test_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compare groups using appropriate statistical tests."""
        groups = data[group_column].unique()
        num_groups = len(groups)
        
        if test_type is None:
            test_type = self._determine_test_type(data, group_column, value_column)
        
        results = {
            'test_type': test_type,
            'num_groups': num_groups,
            'groups': groups.tolist()
        }
        
        try:
            if test_type == 'anova':
                results.update(
                    self._perform_anova(data, group_column, value_column)
                )
            elif test_type == 'kruskal':
                results.update(
                    self._perform_kruskal(data, group_column, value_column)
                )
            elif test_type == 'ttest':
                results.update(
                    self._perform_ttest(data, group_column, value_column)
                )
            elif test_type == 'chi_square':
                results.update(
                    self._perform_chi_square(data, group_column, value_column)
                )
                
            # Add post-hoc analysis if applicable
            if num_groups > 2 and test_type in ['anova', 'kruskal']:
                results['post_hoc'] = self._perform_post_hoc(
                    data, group_column, value_column
                )
            
            # Store results
            test_id = f"{test_type}_{group_column}_{value_column}"
            self.test_results[test_id] = results
            
            # Record test
            self._record_test(test_id, results)
            
            return results
            
        except Exception as e:
            raise StatisticalError(
                f"Error performing group comparison: {str(e)}"
            ) from e
    
    def _determine_test_type(
        self,
        data: pd.DataFrame,
        group_column: str,
        value_column: str
    ) -> str:
        """Determine appropriate statistical test."""
        # Check if value column is numeric
        is_numeric = pd.api.types.is_numeric_dtype(data[value_column])
        
        if not is_numeric:
            return 'chi_square'
            
        # Check normality for each group
        groups = []
        for group in data[group_column].unique():
            group_data = data[data[group_column] == group][value_column]
            groups.append(group_data)
            
            # Shapiro-Wilk test for normality
            if len(group_data) >= 3:  # Minimum required for Shapiro-Wilk
                _, p_value = stats.shapiro(group_data)
                if p_value < 0.05:  # Not normal
                    return 'kruskal'
        
        # If we get here, data is normal - use ANOVA or t-test
        return 'anova' if len(groups) > 2 else 'ttest'
    
    @monitor_performance
    def _perform_anova(
        self,
        data: pd.DataFrame,
        group_column: str,
        value_column: str
    ) -> Dict[str, Any]:
        """Perform one-way ANOVA test."""
        groups = [
            data[data[group_column] == group][value_column]
            for group in data[group_column].unique()
        ]
        
        f_statistic, p_value = stats.f_oneway(*groups)
        
        return {
            'f_statistic': float(f_statistic),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        }
    
    @monitor_performance
    def _perform_kruskal(
        self,
        data: pd.DataFrame,
        group_column: str,
        value_column: str
    ) -> Dict[str, Any]:
        """Perform Kruskal-Wallis H-test."""
        groups = [
            data[data[group_column] == group][value_column]
            for group in data[group_column].unique()
        ]
        
        h_statistic, p_value = stats.kruskal(*groups)
        
        return {
            'h_statistic': float(h_statistic),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        }
    
    @monitor_performance
    def _perform_ttest(
        self,
        data: pd.DataFrame,
        group_column: str,
        value_column: str
    ) -> Dict[str, Any]:
        """Perform independent t-test."""
        groups = data[group_column].unique()
        if len(groups) != 2:
            raise StatisticalError("T-test requires exactly two groups")
        
        group1 = data[data[group_column] == groups[0]][value_column]
        group2 = data[data[group_column] == groups[1]][value_column]
        
        t_statistic, p_value = stats.ttest_ind(group1, group2)
        
        return {
            't_statistic': float(t_statistic),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'group1_mean': float(group1.mean()),
            'group2_mean': float(group2.mean())
        }
    
    @monitor_performance
    def _perform_chi_square(
        self,
        data: pd.DataFrame,
        group_column: str,
        value_column: str
    ) -> Dict[str, Any]:
        """Perform chi-square test of independence."""
        contingency_table = pd.crosstab(
            data[group_column],
            data[value_column]
        )
        
        chi2_statistic, p_value, dof, expected = stats.chi2_contingency(
            contingency_table
        )
        
        return {
            'chi2_statistic': float(chi2_statistic),
            'p_value': float(p_value),
            'degrees_of_freedom': int(dof),
            'contingency_table': contingency_table.to_dict(),
            'expected_frequencies': expected.tolist(),
            'significant': p_value < 0.05
        }
    
    @monitor_performance
    def _perform_post_hoc(
        self,
        data: pd.DataFrame,
        group_column: str,
        value_column: str
    ) -> Dict[str, Any]:
        """Perform post-hoc analysis."""
        # Tukey's HSD test
        tukey = pairwise_tukeyhsd(
            data[value_column],
            data[group_column]
        )
        
        return {
            'tukey_results': {
                'group1': tukey.groupsunique[0],
                'group2': tukey.groupsunique[1],
                'meandiff': float(tukey.meandiffs[0]),
                'p_value': float(tukey.pvalues[0]),
                'reject': bool(tukey.reject[0])
            }
        }
    
    def _record_test(
        self,
        test_id: str,
        results: Dict[str, Any]
    ) -> None:
        """Record statistical test in history."""
        record = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'test_id': test_id,
            'results': results
        }
        
        self.statistics_history.append(record)
        state_manager.set_state(
            f'analysis.statistics.history.{len(self.statistics_history)}',
            record
        )

# Create global statistical analyzer instance
statistical_analyzer = StatisticalAnalyzer()