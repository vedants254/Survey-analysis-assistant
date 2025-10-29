"""
Graceful Degradation System
Ensures users always receive valuable insights even when full analysis fails
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class FallbackAnalyzer:
    """Provides basic analysis when advanced methods fail"""
    
    def __init__(self):
        self.analysis_templates = {
            "revenue_analysis": self._revenue_fallback,
            "trend_analysis": self._trend_fallback,
            "comparison_analysis": self._comparison_fallback,
            "correlation_analysis": self._correlation_fallback,
            "generic": self._generic_fallback
        }
    
    async def generate_fallback_analysis(
        self, 
        state: Dict[str, Any], 
        error_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate fallback analysis based on available data"""
        error_context = error_context or {}
        
        logger.info("🚨 Generating graceful degradation analysis")
        
        fallback_result = {
            "status": "fallback_analysis_completed",
            "analysis_type": "graceful_degradation",
            "message": "Analysis completed with simplified methods due to technical limitations",
            "timestamp": datetime.now().isoformat(),
            "data_summary": {},
            "basic_insights": [],
            "recommendations": [],
            "limitations": [],
            "error_context": error_context.get("error_summary", "Technical issues encountered")
        }
        
        try:
            # Determine analysis type from state
            operation_type = state.get("operation_type", "generic")
            query = state.get("query", "").lower()
            
            # Map query terms to analysis types
            if any(term in query for term in ["revenue", "sales", "income"]):
                analysis_type = "revenue_analysis"
            elif any(term in query for term in ["trend", "time", "growth", "change"]):
                analysis_type = "trend_analysis"
            elif any(term in query for term in ["compare", "vs", "difference", "between"]):
                analysis_type = "comparison_analysis"
            elif any(term in query for term in ["correlation", "relationship", "impact"]):
                analysis_type = "correlation_analysis"
            else:
                analysis_type = "generic"
            
            # Load available data
            dataframes = await self._load_available_data(state)
            
            if dataframes:
                # Generate data summary
                fallback_result["data_summary"] = self._generate_data_summary(dataframes)
                
                # Apply appropriate analysis method
                if analysis_type in self.analysis_templates:
                    analysis_results = await self.analysis_templates[analysis_type](dataframes, state)
                    fallback_result.update(analysis_results)
                else:
                    analysis_results = await self._generic_fallback(dataframes, state)
                    fallback_result.update(analysis_results)
                
                # Add general insights
                fallback_result["basic_insights"].extend(
                    self._generate_general_insights(dataframes, state)
                )
                
            else:
                # No data available - provide query-based insights
                fallback_result.update(await self._query_based_insights(state))
            
            # Add limitations and recommendations
            fallback_result["limitations"] = self._generate_limitations(error_context)
            fallback_result["recommendations"].extend(self._generate_recommendations(state, error_context))
            
            logger.info(f"✅ Fallback analysis generated with {len(fallback_result['basic_insights'])} insights")
            
        except Exception as e:
            logger.error(f"❌ Fallback analysis also failed: {e}")
            fallback_result.update({
                "status": "minimal_fallback",
                "basic_insights": [
                    "Data analysis encountered technical difficulties",
                    "Unable to process data with current methods",
                    "Please review data format and try again"
                ],
                "recommendations": [
                    "Check data file format and structure",
                    "Ensure data contains expected columns",
                    "Try with a simpler analysis question",
                    "Contact support if issues persist"
                ],
                "limitations": [
                    "No automated analysis could be performed",
                    "Results are based on general business knowledge only"
                ]
            })
        
        return fallback_result
    
    async def _load_available_data(self, state: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Load whatever data is available from the state"""
        dataframes = {}
        
        try:
            # Try to load from aligned_data first
            if state.get("aligned_data"):
                for time_key, file_data in state["aligned_data"].items():
                    try:
                        file_path = Path(file_data['file_path'])
                        if file_path.exists():
                            if file_path.suffix.lower() == '.csv':
                                df = pd.read_csv(file_path)
                            else:
                                df = pd.read_excel(file_path)
                            
                            if not df.empty:
                                dataframes[time_key] = df
                    except Exception as e:
                        logger.warning(f"Could not load {time_key}: {e}")
            
            # Fallback to parsed_files
            elif state.get("parsed_files"):
                for file_info in state["parsed_files"]:
                    try:
                        file_path = Path(file_info["file_path"])
                        if file_path.exists():
                            if file_path.suffix.lower() == '.csv':
                                df = pd.read_csv(file_path)
                            else:
                                df = pd.read_excel(file_path)
                            
                            if not df.empty:
                                file_name = file_info.get("original_filename", file_path.name)
                                dataframes[file_name] = df
                    except Exception as e:
                        logger.warning(f"Could not load {file_info.get('original_filename', 'unknown')}: {e}")
        
        except Exception as e:
            logger.error(f"Error loading data for fallback analysis: {e}")
        
        return dataframes
    
    def _generate_data_summary(self, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Generate basic data summary statistics"""
        summary = {
            "total_files": len(dataframes),
            "total_rows": 0,
            "total_columns": 0,
            "numeric_columns": [],
            "text_columns": [],
            "date_columns": [],
            "file_details": {}
        }
        
        for name, df in dataframes.items():
            try:
                file_summary = {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": df.columns.tolist(),
                    "numeric_cols": df.select_dtypes(include=['number']).columns.tolist(),
                    "text_cols": df.select_dtypes(include=['object']).columns.tolist(),
                    "missing_data": df.isnull().sum().sum()
                }
                
                summary["file_details"][name] = file_summary
                summary["total_rows"] += len(df)
                summary["total_columns"] += len(df.columns)
                
                # Aggregate column types
                summary["numeric_columns"].extend(file_summary["numeric_cols"])
                summary["text_columns"].extend(file_summary["text_cols"])
                
                # Try to detect date columns
                for col in df.columns:
                    if any(date_term in col.lower() for date_term in ["date", "time", "created", "modified"]):
                        summary["date_columns"].append(f"{name}.{col}")
                
            except Exception as e:
                logger.warning(f"Error summarizing {name}: {e}")
        
        # Remove duplicates
        summary["numeric_columns"] = list(set(summary["numeric_columns"]))
        summary["text_columns"] = list(set(summary["text_columns"]))
        
        return summary
    
    async def _revenue_fallback(self, dataframes: Dict[str, pd.DataFrame], state: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback analysis for revenue-related queries"""
        insights = []
        revenue_data = {}
        
        for name, df in dataframes.items():
            # Look for revenue-like columns
            revenue_cols = []
            for col in df.columns:
                if any(term in col.lower() for term in ["revenue", "sales", "income", "amount", "price", "cost"]):
                    if pd.api.types.is_numeric_dtype(df[col]):
                        revenue_cols.append(col)
            
            if revenue_cols:
                for col in revenue_cols:
                    try:
                        total = df[col].sum()
                        average = df[col].mean()
                        count = df[col].count()
                        
                        revenue_data[f"{name}_{col}"] = {
                            "total": float(total) if pd.notna(total) else 0,
                            "average": float(average) if pd.notna(average) else 0,
                            "count": int(count)
                        }
                        
                        insights.append(f"📊 {name}: Total {col} is {total:,.2f} across {count} records")
                        insights.append(f"💰 {name}: Average {col} per record is {average:,.2f}")
                        
                    except Exception as e:
                        logger.warning(f"Error calculating revenue stats for {col}: {e}")
        
        return {
            "analysis_focus": "revenue_analysis",
            "revenue_data": revenue_data,
            "basic_insights": insights,
            "executive_summary": f"Revenue analysis identified {len(revenue_data)} revenue streams across {len(dataframes)} data sources."
        }
    
    async def _trend_fallback(self, dataframes: Dict[str, pd.DataFrame], state: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback analysis for trend-related queries"""
        insights = []
        trend_data = {}
        
        for name, df in dataframes.items():
            # Look for time-based data
            date_cols = []
            numeric_cols = []
            
            for col in df.columns:
                if any(term in col.lower() for term in ["date", "time", "month", "year"]):
                    date_cols.append(col)
                elif pd.api.types.is_numeric_dtype(df[col]):
                    numeric_cols.append(col)
            
            if date_cols and numeric_cols:
                try:
                    # Basic trend analysis
                    date_col = date_cols[0]
                    for metric_col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                        # Simple growth calculation
                        first_value = df[metric_col].iloc[0] if len(df) > 0 else 0
                        last_value = df[metric_col].iloc[-1] if len(df) > 1 else first_value
                        
                        if first_value != 0:
                            growth_rate = ((last_value - first_value) / first_value) * 100
                            trend_direction = "increasing" if growth_rate > 5 else "decreasing" if growth_rate < -5 else "stable"
                            
                            trend_data[f"{name}_{metric_col}"] = {
                                "growth_rate": float(growth_rate),
                                "direction": trend_direction,
                                "start_value": float(first_value),
                                "end_value": float(last_value)
                            }
                            
                            insights.append(f"📈 {name}: {metric_col} shows {trend_direction} trend ({growth_rate:+.1f}%)")
                        
                except Exception as e:
                    logger.warning(f"Error calculating trends for {name}: {e}")
        
        return {
            "analysis_focus": "trend_analysis",
            "trend_data": trend_data,
            "basic_insights": insights,
            "executive_summary": f"Trend analysis examined {len(trend_data)} metrics across time periods."
        }
    
    async def _comparison_fallback(self, dataframes: Dict[str, pd.DataFrame], state: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback analysis for comparison queries"""
        insights = []
        comparison_data = {}
        
        if len(dataframes) >= 2:
            # Compare between files
            file_names = list(dataframes.keys())
            file1, file2 = file_names[0], file_names[1]
            df1, df2 = dataframes[file1], dataframes[file2]
            
            # Find common numeric columns
            common_numeric = []
            for col in df1.columns:
                if col in df2.columns and pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col]):
                    common_numeric.append(col)
            
            for col in common_numeric[:3]:  # Limit to first 3 columns
                try:
                    sum1 = df1[col].sum()
                    sum2 = df2[col].sum()
                    
                    if sum1 != 0:
                        difference = ((sum2 - sum1) / sum1) * 100
                        comparison_data[col] = {
                            "file1_total": float(sum1),
                            "file2_total": float(sum2),
                            "difference_percent": float(difference)
                        }
                        
                        insights.append(f"🔄 {col}: {file2} vs {file1} shows {difference:+.1f}% difference")
                    
                except Exception as e:
                    logger.warning(f"Error comparing {col}: {e}")
        
        else:
            # Compare within single file (if categorical columns exist)
            for name, df in dataframes.items():
                categorical_cols = df.select_dtypes(include=['object']).columns
                numeric_cols = df.select_dtypes(include=['number']).columns
                
                if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                    cat_col = categorical_cols[0]
                    num_col = numeric_cols[0]
                    
                    try:
                        grouped = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False)
                        top_categories = grouped.head(3)
                        
                        comparison_data[f"{name}_{cat_col}_breakdown"] = top_categories.to_dict()
                        
                        for category, value in top_categories.items():
                            insights.append(f"🏆 {name}: {category} leads in {num_col} with {value:,.2f}")
                        
                    except Exception as e:
                        logger.warning(f"Error analyzing categories in {name}: {e}")
        
        return {
            "analysis_focus": "comparison_analysis",
            "comparison_data": comparison_data,
            "basic_insights": insights,
            "executive_summary": f"Comparison analysis found {len(comparison_data)} key differences."
        }
    
    async def _correlation_fallback(self, dataframes: Dict[str, pd.DataFrame], state: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback analysis for correlation queries"""
        insights = []
        correlation_data = {}
        
        for name, df in dataframes.items():
            numeric_cols = df.select_dtypes(include=['number']).columns
            
            if len(numeric_cols) >= 2:
                try:
                    # Calculate correlation matrix for first few numeric columns
                    corr_matrix = df[numeric_cols[:4]].corr()  # Limit to 4 columns
                    
                    # Find strong correlations
                    for i, col1 in enumerate(corr_matrix.columns):
                        for j, col2 in enumerate(corr_matrix.columns):
                            if i < j:  # Avoid duplicates
                                corr_value = corr_matrix.loc[col1, col2]
                                if abs(corr_value) > 0.5 and pd.notna(corr_value):
                                    correlation_data[f"{col1}_vs_{col2}"] = float(corr_value)
                                    
                                    strength = "strong" if abs(corr_value) > 0.7 else "moderate"
                                    direction = "positive" if corr_value > 0 else "negative"
                                    
                                    insights.append(f"🔗 {name}: {strength} {direction} correlation between {col1} and {col2} ({corr_value:.2f})")
                
                except Exception as e:
                    logger.warning(f"Error calculating correlations for {name}: {e}")
        
        return {
            "analysis_focus": "correlation_analysis", 
            "correlation_data": correlation_data,
            "basic_insights": insights,
            "executive_summary": f"Correlation analysis found {len(correlation_data)} significant relationships."
        }
    
    async def _generic_fallback(self, dataframes: Dict[str, pd.DataFrame], state: Dict[str, Any]) -> Dict[str, Any]:
        """Generic fallback analysis when specific type cannot be determined"""
        insights = []
        generic_data = {}
        
        for name, df in dataframes.items():
            try:
                # Basic statistics
                numeric_cols = df.select_dtypes(include=['number']).columns
                text_cols = df.select_dtypes(include=['object']).columns
                
                file_insights = []
                
                # Numeric column insights
                for col in numeric_cols[:3]:  # Limit to 3 columns
                    total = df[col].sum()
                    avg = df[col].mean()
                    file_insights.append(f"📊 {col}: Total {total:,.0f}, Average {avg:,.2f}")
                
                # Categorical column insights  
                for col in text_cols[:2]:  # Limit to 2 columns
                    unique_count = df[col].nunique()
                    most_common = df[col].mode()[0] if not df[col].mode().empty else "N/A"
                    file_insights.append(f"📝 {col}: {unique_count} unique values, most common: {most_common}")
                
                generic_data[name] = {
                    "row_count": len(df),
                    "column_count": len(df.columns),
                    "numeric_columns": len(numeric_cols),
                    "text_columns": len(text_cols),
                    "missing_values": df.isnull().sum().sum()
                }
                
                insights.append(f"📋 {name}: {len(df)} rows, {len(df.columns)} columns")
                insights.extend(file_insights)
                
            except Exception as e:
                logger.warning(f"Error in generic analysis for {name}: {e}")
        
        return {
            "analysis_focus": "generic_analysis",
            "generic_data": generic_data,
            "basic_insights": insights,
            "executive_summary": f"Data overview of {len(dataframes)} files with basic statistics provided."
        }
    
    async def _query_based_insights(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights based on the query when no data is available"""
        query = state.get("query", "")
        
        insights = [
            "❌ Unable to access data files for analysis",
            "🤔 Analysis question understood but data processing failed",
            f"📝 Original query: '{query}'"
        ]
        
        # Add query-specific guidance
        query_lower = query.lower()
        if any(term in query_lower for term in ["revenue", "sales"]):
            insights.extend([
                "💡 For revenue analysis, ensure data includes revenue/sales columns",
                "📊 Typical revenue metrics include total sales, average order value, growth rates"
            ])
        elif any(term in query_lower for term in ["trend", "growth"]):
            insights.extend([
                "📈 For trend analysis, ensure data includes time/date columns",
                "⏰ Typical trend metrics include period-over-period growth, seasonal patterns"
            ])
        elif any(term in query_lower for term in ["compare", "comparison"]):
            insights.extend([
                "⚖️ For comparisons, ensure data can be grouped by categories or time periods",
                "🔍 Consider comparing totals, averages, or growth rates between groups"
            ])
        
        return {
            "analysis_focus": "query_based_guidance",
            "basic_insights": insights,
            "executive_summary": "Unable to process data but provided guidance based on analysis intent."
        }
    
    def _generate_general_insights(self, dataframes: Dict[str, pd.DataFrame], state: Dict[str, Any]) -> List[str]:
        """Generate general insights that apply to most analyses"""
        insights = []
        
        total_rows = sum(len(df) for df in dataframes.values())
        total_files = len(dataframes)
        
        insights.extend([
            f"📂 Analysis processed {total_files} data file(s) containing {total_rows:,} total records",
            f"🕒 Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ])
        
        # Data quality insights
        missing_data_files = []
        for name, df in dataframes.items():
            missing_count = df.isnull().sum().sum()
            if missing_count > 0:
                missing_data_files.append(f"{name} ({missing_count} missing values)")
        
        if missing_data_files:
            insights.append(f"⚠️ Missing data detected in: {', '.join(missing_data_files)}")
        else:
            insights.append("✅ No missing data detected in processed files")
        
        return insights
    
    def _generate_limitations(self, error_context: Dict[str, Any]) -> List[str]:
        """Generate limitations based on error context"""
        limitations = [
            "Analysis completed using simplified methods due to technical constraints",
            "Results may be less detailed than full analysis would provide",
            "Statistical significance testing was not performed"
        ]
        
        # Add specific limitations based on errors
        if error_context:
            if "timeout" in str(error_context).lower():
                limitations.append("Analysis time was limited to prevent system overload")
            if "column" in str(error_context).lower() or "data" in str(error_context).lower():
                limitations.append("Some expected data columns may not have been processed")
            if "llm" in str(error_context).lower() or "api" in str(error_context).lower():
                limitations.append("Advanced AI-powered insights are not available")
        
        return limitations
    
    def _generate_recommendations(self, state: Dict[str, Any], error_context: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving future analysis"""
        recommendations = [
            "📋 Review data structure to ensure all expected columns are present",
            "🧹 Clean data by removing empty rows and ensuring consistent formatting",
            "📏 Consider analyzing smaller data subsets for better performance"
        ]
        
        # Add specific recommendations based on query
        query = state.get("query", "").lower()
        if "revenue" in query or "sales" in query:
            recommendations.append("💰 Ensure revenue/sales data is in numeric format without currency symbols")
        if "trend" in query or "time" in query:
            recommendations.append("📅 Ensure date columns are properly formatted and consistent")
        if "compare" in query:
            recommendations.append("🏷️ Ensure categorical columns use consistent naming conventions")
        
        return recommendations


# Factory function
def create_fallback_analyzer() -> FallbackAnalyzer:
    """Create a fallback analyzer instance"""
    return FallbackAnalyzer()


# Integration function for workflow
async def apply_graceful_degradation(
    state: Dict[str, Any],
    errors: List[str],
    error_context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Apply graceful degradation to provide value despite errors"""
    logger.info("🛡️ Applying graceful degradation analysis")
    
    analyzer = create_fallback_analyzer()
    
    # Generate fallback analysis
    fallback_result = await analyzer.generate_fallback_analysis(state, error_context)
    
    # Format as final result
    final_result = {
        "operation_type": state.get("operation_type", "graceful_degradation"),
        "analysis_status": "completed_with_graceful_degradation",
        "executive_summary": fallback_result.get("executive_summary", "Analysis completed with simplified methods"),
        "key_findings": fallback_result.get("basic_insights", []),
        "recommended_actions": fallback_result.get("recommendations", []),
        "data_summary": fallback_result.get("data_summary", {}),
        "analysis_details": fallback_result,
        "limitations": fallback_result.get("limitations", []),
        "confidence_score": 0.6,  # Moderate confidence for fallback analysis
        "methodology": "graceful_degradation_analysis",
        "errors_encountered": errors,
        "timestamp": datetime.now().isoformat()
    }
    
    return final_result