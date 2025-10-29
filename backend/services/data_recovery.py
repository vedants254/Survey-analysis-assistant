"""
Intelligent Data Error Recovery System
Automatically detects and fixes common data issues to enable analysis continuation
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class DataRecoveryError(Exception):
    """Raised when data recovery is not possible"""
    pass


class ColumnMapper:
    """Intelligent column name mapping and recovery"""
    
    # Common column name variations for business data
    COLUMN_MAPPINGS = {
        'revenue': ['revenue', 'sales', 'total_sales', 'gross_sales', 'income', 'turnover'],
        'units': ['units', 'quantity', 'qty', 'count', 'items_sold', 'volume'],
        'discount': ['discount', 'discount_rate', 'discount_percent', 'discount_amount'],
        'product': ['product', 'product_name', 'item', 'sku', 'product_id'],
        'region': ['region', 'area', 'territory', 'location', 'country', 'state'],
        'date': ['date', 'order_date', 'sale_date', 'transaction_date', 'timestamp'],
        'customer': ['customer', 'client', 'customer_name', 'customer_id'],
        'category': ['category', 'product_category', 'type', 'class'],
        'price': ['price', 'unit_price', 'cost', 'rate'],
        'margin': ['margin', 'profit_margin', 'gross_margin', 'profit'],
    }
    
    # Common date formats to try
    DATE_FORMATS = [
        '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S',
        '%m/%d/%Y %H:%M:%S', '%Y-%m', '%m/%Y', '%B %Y', '%b %Y'
    ]
    
    def find_column_matches(self, available_columns: List[str], required_column: str) -> List[str]:
        """Find potential matches for a required column"""
        required_lower = required_column.lower()
        available_lower = [col.lower() for col in available_columns]
        
        matches = []
        
        # Exact match
        if required_lower in available_lower:
            exact_match = available_columns[available_lower.index(required_lower)]
            matches.append(exact_match)
        
        # Check common mappings
        if required_lower in self.COLUMN_MAPPINGS:
            for synonym in self.COLUMN_MAPPINGS[required_lower]:
                if synonym in available_lower:
                    match = available_columns[available_lower.index(synonym)]
                    if match not in matches:
                        matches.append(match)
        
        # Fuzzy matching - contains substring
        for col in available_columns:
            col_lower = col.lower()
            if (required_lower in col_lower or col_lower in required_lower) and col not in matches:
                matches.append(col)
        
        # Pattern matching for common business terms
        patterns = {
            'revenue': [r'.*revenue.*', r'.*sales.*', r'.*income.*'],
            'units': [r'.*unit.*', r'.*qty.*', r'.*quantity.*'],
            'discount': [r'.*discount.*', r'.*rebate.*'],
            'date': [r'.*date.*', r'.*time.*'],
            'product': [r'.*product.*', r'.*item.*', r'.*sku.*'],
        }
        
        if required_lower in patterns:
            for pattern in patterns[required_lower]:
                for col in available_columns:
                    if re.match(pattern, col.lower()) and col not in matches:
                        matches.append(col)
        
        return matches
    
    def create_column_mapping(self, dataframes: Dict[str, pd.DataFrame], 
                             required_columns: List[str]) -> Dict[str, Dict[str, str]]:
        """Create column mapping for all dataframes"""
        mappings = {}
        
        for df_name, df in dataframes.items():
            available_columns = df.columns.tolist()
            df_mapping = {}
            
            for required_col in required_columns:
                matches = self.find_column_matches(available_columns, required_col)
                if matches:
                    # Use the first (best) match
                    df_mapping[required_col] = matches[0]
                    logger.info(f"📊 {df_name}: Mapped '{required_col}' → '{matches[0]}'")
            
            mappings[df_name] = df_mapping
        
        return mappings


class DataCleaner:
    """Cleans and standardizes data formats"""
    
    def clean_dataframe(self, df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
        """Clean and standardize a dataframe"""
        cleaned_df = df.copy()
        
        # Apply column mapping
        rename_dict = {v: k for k, v in column_mapping.items() if v in cleaned_df.columns}
        if rename_dict:
            cleaned_df = cleaned_df.rename(columns=rename_dict)
            logger.info(f"🔄 Renamed columns: {rename_dict}")
        
        # Clean numeric columns
        cleaned_df = self._clean_numeric_columns(cleaned_df)
        
        # Clean date columns
        cleaned_df = self._clean_date_columns(cleaned_df)
        
        # Clean text columns
        cleaned_df = self._clean_text_columns(cleaned_df)
        
        # Handle missing values
        cleaned_df = self._handle_missing_values(cleaned_df)
        
        return cleaned_df
    
    def _clean_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean numeric columns - remove currency symbols, convert strings, etc."""
        numeric_candidates = ['revenue', 'sales', 'units', 'quantity', 'price', 'discount', 'margin']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(candidate in col_lower for candidate in numeric_candidates):
                try:
                    # Remove currency symbols and commas
                    if df[col].dtype == 'object':
                        cleaned_series = df[col].astype(str)
                        cleaned_series = cleaned_series.str.replace(r'[$€£¥,]', '', regex=True)
                        cleaned_series = cleaned_series.str.replace(r'[^\d.-]', '', regex=True)
                        
                        # Convert to numeric
                        df[col] = pd.to_numeric(cleaned_series, errors='coerce')
                        logger.info(f"🔢 Cleaned numeric column: {col}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Could not clean numeric column {col}: {e}")
        
        return df
    
    def _clean_date_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize date columns"""
        date_candidates = ['date', 'time', 'order_date', 'sale_date', 'transaction_date']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(candidate in col_lower for candidate in date_candidates):
                try:
                    # Try different date formats
                    for date_format in ColumnMapper.DATE_FORMATS:
                        try:
                            df[col] = pd.to_datetime(df[col], format=date_format)
                            logger.info(f"📅 Parsed date column {col} with format {date_format}")
                            break
                        except:
                            continue
                    else:
                        # If no format worked, try pandas auto-detection
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        logger.info(f"📅 Auto-parsed date column: {col}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Could not parse date column {col}: {e}")
        
        return df
    
    def _clean_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean text columns - standardize case, remove extra spaces"""
        text_candidates = ['product', 'region', 'category', 'customer']
        
        for col in df.columns:
            if df[col].dtype == 'object':
                col_lower = col.lower()
                if any(candidate in col_lower for candidate in text_candidates):
                    try:
                        # Remove extra spaces and standardize
                        df[col] = df[col].astype(str).str.strip()
                        df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
                        
                        # Standardize case for certain columns
                        if any(x in col_lower for x in ['region', 'country', 'category']):
                            df[col] = df[col].str.title()
                            
                        logger.info(f"📝 Cleaned text column: {col}")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not clean text column {col}: {e}")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values intelligently"""
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                try:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        # For numeric columns, use median for small gaps, 0 for large gaps
                        null_percentage = df[col].isnull().sum() / len(df)
                        if null_percentage < 0.1:  # Less than 10% missing
                            df[col] = df[col].fillna(df[col].median())
                        else:
                            df[col] = df[col].fillna(0)
                        logger.info(f"🔧 Filled missing numeric values in {col}")
                    
                    elif pd.api.types.is_datetime64_any_dtype(df[col]):
                        # For date columns, forward fill or use a default
                        df[col] = df[col].fillna(method='ffill')
                        logger.info(f"🔧 Forward-filled missing dates in {col}")
                    
                    else:
                        # For text columns, use 'Unknown' or most frequent value
                        null_percentage = df[col].isnull().sum() / len(df)
                        if null_percentage < 0.3:  # Less than 30% missing
                            most_frequent = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                            df[col] = df[col].fillna(most_frequent)
                        else:
                            df[col] = df[col].fillna('Unknown')
                        logger.info(f"🔧 Filled missing text values in {col}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Could not handle missing values in {col}: {e}")
        
        return df


class DataValidator:
    """Validates data quality and completeness"""
    
    def validate_dataframes(self, dataframes: Dict[str, pd.DataFrame], 
                           required_columns: List[str]) -> Dict[str, Any]:
        """Validate data quality across all dataframes"""
        validation_report = {
            'overall_status': 'valid',
            'issues': [],
            'warnings': [],
            'dataframe_reports': {}
        }
        
        for df_name, df in dataframes.items():
            df_report = self._validate_single_dataframe(df, df_name, required_columns)
            validation_report['dataframe_reports'][df_name] = df_report
            
            # Aggregate issues
            validation_report['issues'].extend([f"{df_name}: {issue}" for issue in df_report['issues']])
            validation_report['warnings'].extend([f"{df_name}: {warning}" for warning in df_report['warnings']])
        
        # Set overall status
        if validation_report['issues']:
            validation_report['overall_status'] = 'has_issues'
        elif validation_report['warnings']:
            validation_report['overall_status'] = 'has_warnings'
        
        return validation_report
    
    def _validate_single_dataframe(self, df: pd.DataFrame, df_name: str, 
                                  required_columns: List[str]) -> Dict[str, Any]:
        """Validate a single dataframe"""
        report = {
            'status': 'valid',
            'shape': df.shape,
            'issues': [],
            'warnings': [],
            'column_coverage': {},
            'data_quality': {}
        }
        
        # Check if dataframe is empty
        if df.empty:
            report['issues'].append('Dataframe is empty')
            report['status'] = 'invalid'
            return report
        
        # Check column coverage
        available_columns = set(df.columns)
        required_set = set(required_columns)
        missing_columns = required_set - available_columns
        
        if missing_columns:
            report['warnings'].append(f'Missing columns: {list(missing_columns)}')
        
        report['column_coverage'] = {
            'required': len(required_columns),
            'available': len(available_columns),
            'missing': len(missing_columns),
            'coverage_percentage': ((len(required_set & available_columns)) / len(required_set)) * 100
        }
        
        # Check data quality
        for col in df.columns:
            col_quality = self._assess_column_quality(df[col])
            report['data_quality'][col] = col_quality
            
            if col_quality['null_percentage'] > 50:
                report['warnings'].append(f'Column {col} has high null percentage: {col_quality["null_percentage"]:.1f}%')
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            report['warnings'].append(f'Found {duplicate_count} duplicate rows')
        
        return report
    
    def _assess_column_quality(self, series: pd.Series) -> Dict[str, Any]:
        """Assess quality of a single column"""
        total_count = len(series)
        null_count = series.isnull().sum()
        
        quality = {
            'total_count': total_count,
            'null_count': null_count,
            'null_percentage': (null_count / total_count) * 100 if total_count > 0 else 0,
            'data_type': str(series.dtype),
            'unique_values': series.nunique(),
            'uniqueness_percentage': (series.nunique() / total_count) * 100 if total_count > 0 else 0
        }
        
        # Type-specific quality metrics
        if pd.api.types.is_numeric_dtype(series):
            quality.update({
                'min_value': series.min(),
                'max_value': series.max(),
                'mean_value': series.mean(),
                'has_negative': (series < 0).any(),
                'has_zero': (series == 0).any()
            })
        elif pd.api.types.is_datetime64_any_dtype(series):
            quality.update({
                'date_range_start': series.min(),
                'date_range_end': series.max(),
                'date_range_days': (series.max() - series.min()).days if pd.notna(series.max()) else 0
            })
        else:
            # Text columns
            quality.update({
                'most_frequent_value': series.mode()[0] if not series.mode().empty else None,
                'avg_length': series.astype(str).str.len().mean()
            })
        
        return quality


class DataRecoveryEngine:
    """Main data recovery orchestrator"""
    
    def __init__(self):
        self.column_mapper = ColumnMapper()
        self.data_cleaner = DataCleaner()
        self.data_validator = DataValidator()
    
    async def recover_data_for_analysis(
        self, 
        dataframes: Dict[str, pd.DataFrame],
        analysis_requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Main recovery function that attempts to fix data issues to enable analysis
        """
        logger.info("🚀 Starting intelligent data recovery process")
        
        recovery_result = {
            'status': 'success',
            'original_dataframes_count': len(dataframes),
            'recovered_dataframes': {},
            'recovery_log': [],
            'validation_report': {},
            'column_mappings': {},
            'issues_fixed': [],
            'remaining_issues': [],
            'recommendations': []
        }
        
        try:
            # Extract required columns from analysis requirements
            required_columns = self._extract_required_columns(analysis_requirements)
            recovery_result['recovery_log'].append(f"Identified required columns: {required_columns}")
            
            # Step 1: Initial validation
            initial_validation = self.data_validator.validate_dataframes(dataframes, required_columns)
            recovery_result['recovery_log'].append(f"Initial validation: {initial_validation['overall_status']}")
            
            # Step 2: Create column mappings
            column_mappings = self.column_mapper.create_column_mapping(dataframes, required_columns)
            recovery_result['column_mappings'] = column_mappings
            recovery_result['recovery_log'].append("Created column mappings for data alignment")
            
            # Step 3: Clean and recover each dataframe
            recovered_dataframes = {}
            for df_name, df in dataframes.items():
                try:
                    logger.info(f"🔧 Recovering dataframe: {df_name}")
                    
                    # Apply column mapping and cleaning
                    mapping = column_mappings.get(df_name, {})
                    cleaned_df = self.data_cleaner.clean_dataframe(df, mapping)
                    
                    # Verify the cleaned dataframe has minimum required data
                    if not cleaned_df.empty and len(cleaned_df.columns) > 0:
                        recovered_dataframes[df_name] = cleaned_df
                        recovery_result['issues_fixed'].append(f"Successfully recovered {df_name}")
                        logger.info(f"✅ Successfully recovered {df_name}: {cleaned_df.shape}")
                    else:
                        recovery_result['remaining_issues'].append(f"Could not recover {df_name}: insufficient data")
                        logger.warning(f"⚠️ Could not recover {df_name}: insufficient data")
                        
                except Exception as e:
                    error_msg = f"Failed to recover {df_name}: {str(e)}"
                    recovery_result['remaining_issues'].append(error_msg)
                    logger.error(f"❌ {error_msg}")
            
            recovery_result['recovered_dataframes'] = recovered_dataframes
            recovery_result['recovery_log'].append(f"Recovered {len(recovered_dataframes)}/{len(dataframes)} dataframes")
            
            # Step 4: Final validation
            if recovered_dataframes:
                final_validation = self.data_validator.validate_dataframes(recovered_dataframes, required_columns)
                recovery_result['validation_report'] = final_validation
                recovery_result['recovery_log'].append(f"Final validation: {final_validation['overall_status']}")
                
                # Generate recommendations
                recovery_result['recommendations'] = self._generate_recommendations(
                    final_validation, recovery_result['remaining_issues']
                )
                
            else:
                recovery_result['status'] = 'failed'
                recovery_result['recovery_log'].append("No dataframes could be recovered")
                recovery_result['recommendations'] = [
                    "Check data file formats and structure",
                    "Verify column names match expected business terms",
                    "Ensure files contain sufficient data",
                    "Review data quality and completeness"
                ]
            
            # Log summary
            logger.info(f"🎯 Data recovery completed: {recovery_result['status']}")
            logger.info(f"📊 Recovered {len(recovered_dataframes)} dataframes with {len(recovery_result['issues_fixed'])} issues fixed")
            
            return recovery_result
            
        except Exception as e:
            logger.error(f"❌ Data recovery process failed: {e}")
            recovery_result['status'] = 'error'
            recovery_result['error'] = str(e)
            recovery_result['recommendations'] = [
                "Review data files for basic format correctness",
                "Check file permissions and accessibility", 
                "Verify data files are not corrupted",
                "Try with a smaller subset of data"
            ]
            return recovery_result
    
    def _extract_required_columns(self, analysis_requirements: Dict[str, Any]) -> List[str]:
        """Extract required columns from analysis requirements"""
        required = []
        
        # Check for explicitly mentioned columns
        if 'target_metrics' in analysis_requirements:
            required.extend(analysis_requirements['target_metrics'])
        
        if 'grouping_columns' in analysis_requirements:
            required.extend(analysis_requirements['grouping_columns'])
        
        # Add common business analysis columns
        common_columns = ['revenue', 'sales', 'units', 'product', 'region', 'date']
        for col in common_columns:
            if col not in required:
                required.append(col)
        
        # Extract from query text if available
        query = analysis_requirements.get('query', '').lower()
        query_terms = {
            'revenue': ['revenue', 'sales', 'income'],
            'units': ['units', 'quantity', 'volume'],
            'discount': ['discount', 'rebate'],
            'product': ['product', 'item'],
            'region': ['region', 'area', 'country'],
            'date': ['date', 'time', 'month', 'quarter']
        }
        
        for standard_col, terms in query_terms.items():
            if any(term in query for term in terms) and standard_col not in required:
                required.append(standard_col)
        
        return required
    
    def _generate_recommendations(self, validation_report: Dict[str, Any], 
                                remaining_issues: List[str]) -> List[str]:
        """Generate actionable recommendations based on validation results"""
        recommendations = []
        
        if validation_report.get('overall_status') == 'valid':
            recommendations.append("✅ Data is ready for analysis")
        
        # Coverage recommendations
        for df_name, df_report in validation_report.get('dataframe_reports', {}).items():
            coverage = df_report.get('column_coverage', {})
            if coverage.get('coverage_percentage', 0) < 70:
                recommendations.append(f"📊 {df_name}: Consider adding missing columns for better analysis")
        
        # Data quality recommendations
        for df_name, df_report in validation_report.get('dataframe_reports', {}).items():
            for col, quality in df_report.get('data_quality', {}).items():
                if quality.get('null_percentage', 0) > 30:
                    recommendations.append(f"🔧 {df_name}.{col}: High missing data rate, consider data imputation")
        
        # General recommendations based on remaining issues
        if remaining_issues:
            recommendations.append("⚠️ Some data recovery issues remain - analysis may be limited")
            if any('column' in issue.lower() for issue in remaining_issues):
                recommendations.append("📋 Review column names and ensure they match business terminology")
        
        if not recommendations:
            recommendations.append("🎯 Data recovery completed successfully - ready for analysis")
        
        return recommendations


# Factory function for convenience
def create_data_recovery_engine() -> DataRecoveryEngine:
    """Create a configured data recovery engine"""
    return DataRecoveryEngine()


# Example usage
async def example_data_recovery():
    """Example of data recovery usage"""
    engine = create_data_recovery_engine()
    
    # Example problematic dataframes
    dataframes = {
        'sales_data': pd.DataFrame({
            'Product_nov': ['A', 'B', 'C'],  # Problematic column name
            'Sales_Amount': ['$100', '$200', '$300'],  # String currency
            'Qty': [10, None, 15],  # Missing values
            'Date_sold': ['2023-01-01', '2023/01/02', 'Jan 3 2023']  # Inconsistent dates
        })
    }
    
    analysis_requirements = {
        'query': 'Analyze revenue trends by product',
        'target_metrics': ['revenue', 'units'],
        'operation_type': 'trend_analysis'
    }
    
    result = await engine.recover_data_for_analysis(dataframes, analysis_requirements)
    logger.info(f"Recovery result: {result}")
    
    return result