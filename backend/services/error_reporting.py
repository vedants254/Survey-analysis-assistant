"""
Comprehensive Error Reporting System
Provides detailed, user-friendly error reports with actionable recommendations
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for prioritization"""
    LOW = "low"
    MEDIUM = "medium"  
    HIGH = "high"
    CRITICAL = "critical"


class ImpactLevel(Enum):
    """Impact levels on analysis outcome"""
    MINIMAL = "minimal"        # Analysis can proceed normally
    MODERATE = "moderate"      # Analysis proceeds with limitations
    SEVERE = "severe"          # Analysis significantly affected
    BLOCKING = "blocking"      # Analysis cannot proceed


@dataclass
class ErrorDetail:
    """Detailed error information"""
    error_id: str
    category: str
    severity: ErrorSeverity
    impact: ImpactLevel
    title: str
    description: str
    technical_details: str
    user_message: str
    suggested_actions: List[str] = field(default_factory=list)
    auto_fix_available: bool = False
    auto_fix_applied: bool = False
    node_context: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


class ErrorReportGenerator:
    """Generates comprehensive error reports for users and developers"""
    
    def __init__(self):
        self.error_templates = self._load_error_templates()
    
    def _load_error_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load error message templates for consistent reporting"""
        return {
            "data_column_missing": {
                "title": "Required Data Column Not Found",
                "severity": ErrorSeverity.HIGH,
                "impact": ImpactLevel.SEVERE,
                "user_template": "The analysis expected to find a column called '{missing_column}' but it wasn't found in your data. This is commonly needed for {analysis_type} analysis.",
                "technical_template": "Column '{missing_column}' not found in DataFrame. Available columns: {available_columns}",
                "suggestions": [
                    "Check if the column has a different name (e.g., 'Sales' instead of 'Revenue')",
                    "Verify the data file contains all expected columns",
                    "Consider renaming the column in your data to match the expected name",
                    "Upload a file that contains the required column"
                ],
                "auto_fix_available": True
            },
            "data_type_mismatch": {
                "title": "Data Type Issue Detected",
                "severity": ErrorSeverity.MEDIUM,
                "impact": ImpactLevel.MODERATE,
                "user_template": "The column '{column_name}' contains {actual_type} data but {expected_type} was expected for this analysis.",
                "technical_template": "Data type mismatch: column '{column_name}' has type {actual_type}, expected {expected_type}",
                "suggestions": [
                    "Check if the column contains the correct type of data",
                    "Remove any text or special characters from numeric columns",
                    "Verify date formats are consistent",
                    "Clean the data before uploading"
                ],
                "auto_fix_available": True
            },
            "code_execution_timeout": {
                "title": "Analysis Taking Too Long",
                "severity": ErrorSeverity.HIGH,
                "impact": ImpactLevel.SEVERE,
                "user_template": "The analysis is taking longer than expected to complete, possibly due to large data size or complex calculations.",
                "technical_template": "Code execution timed out after {timeout_seconds} seconds",
                "suggestions": [
                    "Try analyzing a smaller subset of your data first",
                    "Simplify your analysis question",
                    "Remove unnecessary columns from your data",
                    "Break down complex analysis into smaller parts"
                ],
                "auto_fix_available": False
            },
            "llm_api_error": {
                "title": "AI Service Temporarily Unavailable",
                "severity": ErrorSeverity.HIGH,
                "impact": ImpactLevel.SEVERE,
                "user_template": "The AI service is currently experiencing issues: {error_details}. This may be temporary.",
                "technical_template": "LLM API error: {error_details}",
                "suggestions": [
                    "Wait a few minutes and try again",
                    "Try simplifying your analysis question",
                    "Check if there are any known service outages",
                    "Contact support if the issue persists"
                ],
                "auto_fix_available": False
            },
            "pandas_scalar_error": {
                "title": "Data Calculation Error",
                "severity": ErrorSeverity.MEDIUM,
                "impact": ImpactLevel.MODERATE,
                "user_template": "There was an issue calculating results from your data. This often happens when data formats are inconsistent.",
                "technical_template": "Pandas scalar/iloc error: {error_details}",
                "suggestions": [
                    "Check for consistent data formats in all columns",
                    "Ensure numeric columns contain only numbers",
                    "Remove any summary rows or headers within the data",
                    "Verify there are no empty rows in your data"
                ],
                "auto_fix_available": True
            },
            "file_not_found": {
                "title": "Data File Not Found",
                "severity": ErrorSeverity.CRITICAL,
                "impact": ImpactLevel.BLOCKING,
                "user_template": "The data file '{filename}' could not be found or accessed.",
                "technical_template": "File not found: {filepath}",
                "suggestions": [
                    "Verify the file was uploaded successfully",
                    "Check if the file still exists",
                    "Try re-uploading the file",
                    "Ensure the file isn't corrupted"
                ],
                "auto_fix_available": False
            },
            "empty_dataset": {
                "title": "Empty Dataset Detected", 
                "severity": ErrorSeverity.HIGH,
                "impact": ImpactLevel.BLOCKING,
                "user_template": "The data file '{filename}' appears to be empty or contains no valid data rows.",
                "technical_template": "Empty DataFrame detected for file: {filename}",
                "suggestions": [
                    "Verify the data file contains actual data rows",
                    "Check if the file format is correct (CSV, Excel)", 
                    "Ensure the file isn't just headers without data",
                    "Try opening the file in Excel to verify its contents"
                ],
                "auto_fix_available": False
            }
        }
    
    def create_error_report(
        self, 
        errors: List[str],
        context: Dict[str, Any] = None,
        state: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate a comprehensive error report"""
        context = context or {}
        state = state or {}
        
        error_details = []
        summary_stats = {
            "total_errors": len(errors),
            "critical_errors": 0,
            "high_severity_errors": 0,
            "auto_fixable_errors": 0,
            "blocking_errors": 0
        }
        
        for i, error_msg in enumerate(errors):
            error_detail = self._analyze_error(error_msg, context, state)
            error_detail.error_id = f"error_{i+1:03d}"
            error_details.append(error_detail)
            
            # Update summary stats
            if error_detail.severity == ErrorSeverity.CRITICAL:
                summary_stats["critical_errors"] += 1
            elif error_detail.severity == ErrorSeverity.HIGH:
                summary_stats["high_severity_errors"] += 1
            
            if error_detail.auto_fix_available:
                summary_stats["auto_fixable_errors"] += 1
            
            if error_detail.impact == ImpactLevel.BLOCKING:
                summary_stats["blocking_errors"] += 1
        
        # Generate overall assessment
        overall_status = self._assess_overall_status(error_details)
        recommendations = self._generate_overall_recommendations(error_details, context, state)
        
        report = {
            "report_id": f"error_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status,
            "summary": summary_stats,
            "can_continue_analysis": self._can_continue_analysis(error_details),
            "user_summary": self._generate_user_summary(error_details, overall_status),
            "errors": [self._format_error_for_user(detail) for detail in error_details],
            "technical_details": [self._format_error_for_developers(detail) for detail in error_details],
            "recommendations": recommendations,
            "next_steps": self._generate_next_steps(error_details, overall_status),
            "context": {
                "analysis_type": context.get("operation_type", "unknown"),
                "files_count": len(state.get("parsed_files", [])),
                "node_context": context.get("current_node", "unknown"),
                "enhanced_error_handling": context.get("error_handling_available", False)
            }
        }
        
        return report
    
    def _analyze_error(self, error_msg: str, context: Dict[str, Any], state: Dict[str, Any]) -> ErrorDetail:
        """Analyze a single error and classify it"""
        error_lower = error_msg.lower()
        
        # Detect error patterns and map to templates
        if any(term in error_lower for term in ["column", "not found", "keyerror"]):
            if any(term in error_lower for term in ["product_nov", "missing"]):
                return self._create_error_from_template(
                    "data_column_missing", error_msg, context, state,
                    missing_column="Product_nov",
                    analysis_type=context.get("operation_type", "business"),
                    available_columns="[columns not available in context]"
                )
        
        elif any(term in error_lower for term in ["float64", "iloc", "scalar", "numpy"]):
            return self._create_error_from_template(
                "pandas_scalar_error", error_msg, context, state,
                error_details=error_msg[:100]
            )
        
        elif any(term in error_lower for term in ["timeout", "timed out"]):
            return self._create_error_from_template(
                "code_execution_timeout", error_msg, context, state,
                timeout_seconds=context.get("timeout_seconds", "unknown")
            )
        
        elif any(term in error_lower for term in ["rate limit", "quota", "api"]):
            return self._create_error_from_template(
                "llm_api_error", error_msg, context, state,
                error_details=error_msg
            )
        
        elif any(term in error_lower for term in ["file not found", "no such file"]):
            return self._create_error_from_template(
                "file_not_found", error_msg, context, state,
                filename=context.get("filename", "unknown"),
                filepath=error_msg
            )
        
        elif any(term in error_lower for term in ["empty", "no data"]):
            return self._create_error_from_template(
                "empty_dataset", error_msg, context, state,
                filename=context.get("filename", "unknown")
            )
        
        # Generic error if no pattern matches
        return ErrorDetail(
            error_id="",
            category="unknown",
            severity=ErrorSeverity.MEDIUM,
            impact=ImpactLevel.MODERATE,
            title="Unexpected Error",
            description=error_msg,
            technical_details=error_msg,
            user_message=f"An unexpected error occurred: {error_msg[:100]}...",
            suggested_actions=[
                "Try refreshing and running the analysis again",
                "Check your data for any formatting issues",
                "Contact support if the issue persists"
            ],
            node_context=context.get("current_node", "unknown")
        )
    
    def _create_error_from_template(
        self, 
        template_key: str, 
        error_msg: str, 
        context: Dict[str, Any], 
        state: Dict[str, Any],
        **kwargs
    ) -> ErrorDetail:
        """Create error detail from template"""
        template = self.error_templates[template_key]
        
        # Safely format templates with provided kwargs
        try:
            description = template["user_template"].format(**kwargs) if kwargs else template["user_template"]
        except KeyError:
            description = template["user_template"]
        
        try:
            technical_details = template["technical_template"].format(**kwargs) if kwargs else error_msg
        except KeyError:
            technical_details = error_msg
            
        try:
            user_message = template["user_template"].format(**kwargs) if kwargs else template["user_template"]
        except KeyError:
            user_message = template["user_template"]
        
        return ErrorDetail(
            error_id="",
            category=template_key,
            severity=template["severity"],
            impact=template["impact"],
            title=template["title"],
            description=description,
            technical_details=technical_details,
            user_message=user_message,
            suggested_actions=template["suggestions"],
            auto_fix_available=template["auto_fix_available"],
            node_context=context.get("current_node", "unknown")
        )
    
    def _assess_overall_status(self, error_details: List[ErrorDetail]) -> str:
        """Assess overall status based on error details"""
        if not error_details:
            return "success"
        
        if any(e.impact == ImpactLevel.BLOCKING for e in error_details):
            return "blocked"
        elif any(e.severity == ErrorSeverity.CRITICAL for e in error_details):
            return "critical_issues"
        elif any(e.severity == ErrorSeverity.HIGH for e in error_details):
            return "has_issues"
        else:
            return "completed_with_warnings"
    
    def _can_continue_analysis(self, error_details: List[ErrorDetail]) -> bool:
        """Determine if analysis can continue despite errors"""
        return not any(e.impact == ImpactLevel.BLOCKING for e in error_details)
    
    def _generate_user_summary(self, error_details: List[ErrorDetail], overall_status: str) -> str:
        """Generate user-friendly summary"""
        if not error_details:
            return "Analysis completed successfully without any issues."
        
        summary_map = {
            "blocked": "Analysis cannot proceed due to critical issues that need to be resolved.",
            "critical_issues": "Analysis encountered critical issues that significantly impact results.",
            "has_issues": "Analysis completed but encountered some issues that may affect results.",
            "completed_with_warnings": "Analysis completed successfully with minor warnings."
        }
        
        base_summary = summary_map.get(overall_status, "Analysis encountered some issues.")
        
        # Add specific counts
        critical_count = sum(1 for e in error_details if e.severity == ErrorSeverity.CRITICAL)
        high_count = sum(1 for e in error_details if e.severity == ErrorSeverity.HIGH)
        fixable_count = sum(1 for e in error_details if e.auto_fix_available)
        
        details = []
        if critical_count > 0:
            details.append(f"{critical_count} critical error(s)")
        if high_count > 0:
            details.append(f"{high_count} high-priority error(s)")
        if fixable_count > 0:
            details.append(f"{fixable_count} error(s) can be automatically fixed")
        
        if details:
            base_summary += f" Found: {', '.join(details)}."
        
        return base_summary
    
    def _generate_overall_recommendations(
        self, 
        error_details: List[ErrorDetail], 
        context: Dict[str, Any], 
        state: Dict[str, Any]
    ) -> List[str]:
        """Generate overall recommendations"""
        recommendations = []
        
        # Priority-based recommendations
        blocking_errors = [e for e in error_details if e.impact == ImpactLevel.BLOCKING]
        if blocking_errors:
            recommendations.append("🚨 Address blocking issues first before retrying analysis")
            recommendations.extend([f"• {action}" for action in blocking_errors[0].suggested_actions[:2]])
        
        # Auto-fix recommendations
        auto_fixable = [e for e in error_details if e.auto_fix_available and not e.auto_fix_applied]
        if auto_fixable:
            recommendations.append(f"🔧 {len(auto_fixable)} error(s) can be automatically corrected")
        
        # Data quality recommendations
        data_errors = [e for e in error_details if "data" in e.category]
        if data_errors:
            recommendations.append("📊 Review data quality and formatting")
            recommendations.append("• Ensure column names match expected business terms")
            recommendations.append("• Check for consistent data types across all files")
        
        # General recommendations
        if len(error_details) > 1:
            recommendations.append("🔍 Review the detailed error list below for specific actions")
        
        return recommendations
    
    def _generate_next_steps(self, error_details: List[ErrorDetail], overall_status: str) -> List[str]:
        """Generate actionable next steps"""
        if not error_details:
            return ["✅ Continue with analysis - no issues detected"]
        
        next_steps = []
        
        if overall_status == "blocked":
            next_steps.extend([
                "1. Fix the blocking issues identified above",
                "2. Re-upload any corrected data files",
                "3. Retry the analysis"
            ])
        elif overall_status in ["critical_issues", "has_issues"]:
            next_steps.extend([
                "1. Review the errors and suggested actions",
                "2. Apply available automatic fixes",
                "3. Update data files if needed",
                "4. Continue analysis with current results or retry"
            ])
        else:
            next_steps.extend([
                "1. Review warnings to ensure they don't affect your analysis goals",
                "2. Proceed with analysis results",
                "3. Consider addressing warnings for improved accuracy"
            ])
        
        return next_steps
    
    def _format_error_for_user(self, detail: ErrorDetail) -> Dict[str, Any]:
        """Format error for user-friendly display"""
        return {
            "id": detail.error_id,
            "title": detail.title,
            "severity": detail.severity.value,
            "impact": detail.impact.value,
            "message": detail.user_message,
            "suggested_actions": detail.suggested_actions,
            "auto_fix_available": detail.auto_fix_available,
            "auto_fix_applied": detail.auto_fix_applied,
            "can_continue": detail.impact != ImpactLevel.BLOCKING
        }
    
    def _format_error_for_developers(self, detail: ErrorDetail) -> Dict[str, Any]:
        """Format error for technical debugging"""
        return {
            "id": detail.error_id,
            "category": detail.category,
            "severity": detail.severity.value,
            "impact": detail.impact.value,
            "node_context": detail.node_context,
            "technical_details": detail.technical_details,
            "timestamp": detail.timestamp.isoformat()
        }


def create_error_report(
    errors: List[str],
    context: Dict[str, Any] = None,
    state: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Convenience function to create error report"""
    generator = ErrorReportGenerator()
    return generator.create_error_report(errors, context, state)


# Example usage
def example_error_reporting():
    """Example of comprehensive error reporting"""
    errors = [
        "KeyError: 'Product_nov'",
        "Code execution timed out after 30 seconds",
        "'numpy.float64' object has no attribute 'iloc'"
    ]
    
    context = {
        "operation_type": "revenue_analysis",
        "current_node": "execute_code",
        "timeout_seconds": 30,
        "error_handling_available": True
    }
    
    state = {
        "parsed_files": [{"filename": "sales_data.csv"}],
        "query": "Analyze revenue trends"
    }
    
    report = create_error_report(errors, context, state)
    
    logger.info("Error Report Generated:")
    logger.info(f"Overall Status: {report['overall_status']}")
    logger.info(f"User Summary: {report['user_summary']}")
    logger.info(f"Can Continue: {report['can_continue_analysis']}")
    
    return report