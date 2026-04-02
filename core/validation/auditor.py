#!/usr/bin/env python3
"""
MARK5 Comprehensive System Auditor

Analyzes entire codebase for gaps, issues, and optimization opportunities.
Generates detailed audit reports with prioritized action plans.

Author: MARK5 Elite Development Team
Date: 2025-10-22
"""

import os
import sys
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import importlib.util

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


class SystemAuditor:
    """Comprehensive system audit and gap analysis."""
    
    def __init__(self, project_root: str = None):
        self.logger = logging.getLogger(__name__)
        self.project_root = project_root or os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        
        self.issues = []
        self.missing_features = []
        self.optimization_opportunities = []
        self.strengths = []
        
        # Critical component requirements
        self.required_components = {
            'data_pipeline': [
                'data_collector',
                'feature_engineering',
                'data_quality_validator',
                'alternative_data_integration'
            ],
            'models': [
                'ensemble_system',
                'model_trainer',
                'prediction_engine',
                'uncertainty_quantification'
            ],
            'trading_logic': [
                'signal_generator',
                'risk_manager',
                'position_manager',
                'portfolio_optimizer'
            ],
            'validation': [
                'backtesting_engine',
                'paper_trading_system',
                'walk_forward_validator',
                'regime_validator'
            ],
            'production': [
                'real_time_pipeline',
                'monitoring_system',
                'alert_system',
                'failsafe_mechanisms'
            ]
        }
    
    def audit_full_system(self) -> Dict:
        """Execute comprehensive system audit."""
        
        self.logger.info("="*80)
        self.logger.info("MARK5 COMPREHENSIVE SYSTEM AUDIT - STARTING")
        self.logger.info("="*80)
        
        audit_results = {
            'timestamp': datetime.now().isoformat(),
            'project_root': self.project_root,
            'checks_performed': {},
            'overall_status': 'UNKNOWN',
            'critical_issues': 0,
            'warnings': 0,
            'passed_checks': 0
        }
        
        # Execute all audits
        audit_results['checks_performed']['data_pipeline'] = self.audit_data_pipeline()
        audit_results['checks_performed']['model_architecture'] = self.audit_model_architecture()
        audit_results['checks_performed']['trading_logic'] = self.audit_trading_logic()
        audit_results['checks_performed']['testing_validation'] = self.audit_testing_validation()
        audit_results['checks_performed']['production_infra'] = self.audit_production_infrastructure()
        audit_results['checks_performed']['class_balance'] = self.audit_class_balance()
        audit_results['checks_performed']['code_quality'] = self.audit_code_quality()
        
        # Calculate overall status
        audit_results['critical_issues'] = len([i for i in self.issues if i['severity'] == 'CRITICAL'])
        audit_results['warnings'] = len([i for i in self.issues if i['severity'] == 'WARNING'])
        audit_results['passed_checks'] = len(self.strengths)
        
        if audit_results['critical_issues'] > 0:
            audit_results['overall_status'] = 'CRITICAL'
        elif audit_results['warnings'] > 5:
            audit_results['overall_status'] = 'WARNING'
        else:
            audit_results['overall_status'] = 'HEALTHY'
        
        audit_results['issues'] = self.issues
        audit_results['missing_features'] = self.missing_features
        audit_results['optimizations'] = self.optimization_opportunities
        audit_results['strengths'] = self.strengths
        
        return audit_results
    
    def audit_data_pipeline(self) -> Dict:
        """Audit data collection and processing pipeline."""
        
        self.logger.info("\n[1/7] Auditing Data Pipeline...")
        
        results = {
            'status': 'OK',
            'checks': []
        }
        
        # Check 1: Data Collector exists
        data_collector_path = os.path.join(self.project_root, 'core/data/data_collector.py')
        if os.path.exists(data_collector_path):
            results['checks'].append({'name': 'data_collector', 'status': 'PASS'})
            self.strengths.append('Data collector module exists')
        else:
            results['checks'].append({'name': 'data_collector', 'status': 'FAIL'})
            self.issues.append({
                'severity': 'CRITICAL',
                'component': 'data_pipeline',
                'issue': 'Missing data_collector.py',
                'fix': 'Create data collection module'
            })
        
        # Check 2: Feature Engineering
        feature_eng_path = os.path.join(self.project_root, 'core/features/feature_engineering.py')
        if os.path.exists(feature_eng_path):
            results['checks'].append({'name': 'feature_engineering', 'status': 'PASS'})
            self.strengths.append('Feature engineering module exists')
        else:
            results['checks'].append({'name': 'feature_engineering', 'status': 'FAIL'})
            self.issues.append({
                'severity': 'CRITICAL',
                'component': 'data_pipeline',
                'issue': 'Missing feature_engineering.py',
                'fix': 'Create feature engineering module'
            })
        
        # Check 3: Data Quality Validation
        if not self._component_exists('data_quality_validator'):
            self.missing_features.append({
                'feature': 'Data Quality Validator',
                'priority': 'HIGH',
                'description': 'Automated data quality checks for missing values, outliers, staleness'
            })
        
        # Check 4: Alternative Data Integration
        if not self._component_exists('alternative_data'):
            self.missing_features.append({
                'feature': 'Alternative Data Integration',
                'priority': 'MEDIUM',
                'description': 'FII/DII flows, Options PCR, News sentiment, Global indices'
            })
            self.optimization_opportunities.append({
                'area': 'Alpha Generation',
                'opportunity': 'Integrate FII/DII flows and options data',
                'impact': 'HIGH'
            })
        
        return results
    
    def audit_model_architecture(self) -> Dict:
        """Audit ML model implementation."""
        
        self.logger.info("\n[2/7] Auditing Model Architecture...")
        
        results = {
            'status': 'OK',
            'checks': []
        }
        
        # Check 1: Model Trainer
        trainer_path = os.path.join(self.project_root, 'core/models/trainer.py')
        if os.path.exists(trainer_path):
            results['checks'].append({'name': 'model_trainer', 'status': 'PASS'})
            self.strengths.append('Model trainer exists with ensemble support')
            
            # Check ensemble diversity
            with open(trainer_path, 'r') as f:
                content = f.read()
                models_found = []
                if 'xgboost' in content.lower():
                    models_found.append('XGBoost')
                if 'lightgbm' in content.lower():
                    models_found.append('LightGBM')
                if 'randomforest' in content.lower():
                    models_found.append('RandomForest')
                if 'catboost' in content.lower():
                    models_found.append('CatBoost')
                if 'lstm' in content.lower():
                    models_found.append('LSTM')
                
                if len(models_found) >= 4:
                    self.strengths.append(f'Ensemble includes {len(models_found)} models: {", ".join(models_found)}')
                else:
                    self.optimization_opportunities.append({
                        'area': 'Model Diversity',
                        'opportunity': 'Add more model types (N-BEATS, TabNet, Temporal Fusion Transformer)',
                        'impact': 'MEDIUM'
                    })
        
        # Check 2: Uncertainty Quantification
        if not self._code_contains('prediction_engine', 'uncertainty') and \
           not self._code_contains('trainer', 'uncertainty'):
            self.missing_features.append({
                'feature': 'Prediction Uncertainty Quantification',
                'priority': 'HIGH',
                'description': 'Confidence intervals and prediction variance tracking'
            })
        
        # Check 3: Adaptive Weighting
        if not self._code_contains('trainer', 'adaptive_weight'):
            self.missing_features.append({
                'feature': 'Adaptive Model Weighting',
                'priority': 'MEDIUM',
                'description': 'Dynamic ensemble weights based on recent performance'
            })
        
        return results
    
    def audit_trading_logic(self) -> Dict:
        """Audit trading signal generation and risk management."""
        
        self.logger.info("\n[3/7] Auditing Trading Logic...")
        
        results = {
            'status': 'OK',
            'checks': []
        }
        
        # Check 1: Signal Generator
        if self._component_exists('prediction_engine'):
            results['checks'].append({'name': 'signal_generator', 'status': 'PASS'})
            self.strengths.append('Prediction engine exists for signal generation')
        else:
            results['checks'].append({'name': 'signal_generator', 'status': 'FAIL'})
            self.issues.append({
                'severity': 'CRITICAL',
                'component': 'trading_logic',
                'issue': 'Missing prediction/signal generation engine',
                'fix': 'Implement signal generation module'
            })
        
        # Check 2: Risk Management
        if not self._component_exists('risk_manager'):
            self.missing_features.append({
                'feature': 'Risk Management System',
                'priority': 'CRITICAL',
                'description': 'Position sizing, stop losses, portfolio-level risk limits'
            })
        
        # Check 3: Position Management
        if not self._component_exists('position_manager'):
            self.missing_features.append({
                'feature': 'Position Management',
                'priority': 'HIGH',
                'description': 'Track open positions, calculate P&L, manage exits'
            })
        
        # Check 4: Portfolio Optimization
        if not self._component_exists('portfolio'):
            self.optimization_opportunities.append({
                'area': 'Portfolio Management',
                'opportunity': 'Implement portfolio-level optimization and correlation analysis',
                'impact': 'HIGH'
            })
        
        return results
    
    def audit_testing_validation(self) -> Dict:
        """Audit backtesting and validation frameworks."""
        
        self.logger.info("\n[4/7] Auditing Testing & Validation...")
        
        results = {
            'status': 'OK',
            'checks': []
        }
        
        # Check 1: Backtesting Engine
        if not self._component_exists('backtest'):
            self.missing_features.append({
                'feature': 'Production-Grade Backtesting Engine',
                'priority': 'CRITICAL',
                'description': 'Realistic backtesting with transaction costs, slippage, market impact'
            })
            self.issues.append({
                'severity': 'CRITICAL',
                'component': 'validation',
                'issue': 'No backtesting engine found',
                'fix': 'Implement comprehensive backtesting system'
            })
        
        # Check 2: Paper Trading System
        if not self._component_exists('paper_trading'):
            self.missing_features.append({
                'feature': 'Live Paper Trading System',
                'priority': 'CRITICAL',
                'description': 'Real-time paper trading with live market data'
            })
        
        # Check 3: Walk-Forward Validation
        if not self._code_contains('trainer', 'walk_forward') and \
           not self._code_contains('trainer', 'walk-forward'):
            self.logger.warning("Walk-forward validation not found in trainer")
        else:
            self.strengths.append('Walk-forward validation implemented')
        
        # Check 4: Multi-Regime Validation
        if not self._component_exists('regime_validator'):
            self.optimization_opportunities.append({
                'area': 'Validation Robustness',
                'opportunity': 'Test performance across different market regimes (bull/bear/sideways)',
                'impact': 'HIGH'
            })
        
        return results
    
    def audit_production_infrastructure(self) -> Dict:
        """Audit production readiness."""
        
        self.logger.info("\n[5/7] Auditing Production Infrastructure...")
        
        results = {
            'status': 'OK',
            'checks': []
        }
        
        # Check 1: Real-time Pipeline
        if not self._component_exists('real_time'):
            self.missing_features.append({
                'feature': 'Real-Time Data Pipeline',
                'priority': 'HIGH',
                'description': 'Live market data fetching during trading hours'
            })
        
        # Check 2: Monitoring System
        if not self._component_exists('monitor'):
            self.missing_features.append({
                'feature': 'System Monitoring & Alerts',
                'priority': 'HIGH',
                'description': 'Track system health, model performance, data quality'
            })
        
        # Check 3: Logging
        log_dir = os.path.join(self.project_root, 'core/logs')
        if os.path.exists(log_dir):
            results['checks'].append({'name': 'logging_system', 'status': 'PASS'})
            self.strengths.append('Logging infrastructure exists')
        else:
            self.issues.append({
                'severity': 'WARNING',
                'component': 'production',
                'issue': 'No logs directory found',
                'fix': 'Create logging infrastructure'
            })
        
        # Check 4: Failsafe Mechanisms
        if not self._code_contains('*', 'circuit_breaker') and \
           not self._code_contains('*', 'failsafe'):
            self.missing_features.append({
                'feature': 'Failsafe & Circuit Breakers',
                'priority': 'CRITICAL',
                'description': 'Automatic system shutdown on critical errors or excessive losses'
            })
        
        return results
    
    def audit_class_balance(self) -> Dict:
        """Audit for HOLD bias in training data."""
        
        self.logger.info("\n[6/7] Auditing Class Balance...")
        
        results = {
            'status': 'OK',
            'checks': []
        }
        
        # Check threshold configuration
        threshold_path = os.path.join(self.project_root, 'core/utils/threshold_manager.py')
        if os.path.exists(threshold_path):
            with open(threshold_path, 'r') as f:
                content = f.read()
                
                # Check for fixed thresholds
                if 'buy_threshold = 1.5' in content or 'buy_threshold = 1.7' in content:
                    results['checks'].append({'name': 'threshold_fix', 'status': 'PASS'})
                    self.strengths.append('Threshold fix applied (±1.5% to ±1.7%)')
                elif 'buy_threshold = 2.0' in content:
                    self.issues.append({
                        'severity': 'CRITICAL',
                        'component': 'class_balance',
                        'issue': 'Thresholds still at ±2.0% - will cause HOLD bias',
                        'fix': 'Reduce to ±1.5% for normal volatility stocks'
                    })
                
                # Check for class balance validator
                validator_path = os.path.join(self.project_root, 'core/utils/class_balance_validator.py')
                if os.path.exists(validator_path):
                    results['checks'].append({'name': 'balance_validator', 'status': 'PASS'})
                    self.strengths.append('Class balance validator implemented')
                else:
                    self.optimization_opportunities.append({
                        'area': 'Data Quality',
                        'opportunity': 'Implement automated class balance monitoring',
                        'impact': 'MEDIUM'
                    })
        
        return results
    
    def audit_code_quality(self) -> Dict:
        """Audit code quality and best practices."""
        
        self.logger.info("\n[7/7] Auditing Code Quality...")
        
        results = {
            'status': 'OK',
            'checks': []
        }
        
        # Check for tests
        tests_dir = os.path.join(self.project_root, 'tests')
        if os.path.exists(tests_dir):
            test_files = list(Path(tests_dir).rglob('test_*.py'))
            if len(test_files) > 0:
                results['checks'].append({'name': 'unit_tests', 'status': 'PASS'})
                self.strengths.append(f'Unit tests exist ({len(test_files)} test files)')
            else:
                self.optimization_opportunities.append({
                    'area': 'Code Quality',
                    'opportunity': 'Add comprehensive unit tests',
                    'impact': 'MEDIUM'
                })
        else:
            self.optimization_opportunities.append({
                'area': 'Code Quality',
                'opportunity': 'Create test suite for critical components',
                'impact': 'MEDIUM'
            })
        
        # Check for documentation
        docs_dir = os.path.join(self.project_root, 'docs')
        if os.path.exists(docs_dir):
            self.strengths.append('Documentation directory exists')
        
        # Check for config management
        if self._component_exists('config_manager'):
            self.strengths.append('Configuration management system exists')
        
        return results
    
    def _component_exists(self, component_name: str) -> bool:
        """Check if a component file exists in the project."""
        for root, dirs, files in os.walk(self.project_root):
            for file in files:
                if component_name in file.lower() and file.endswith('.py'):
                    return True
        return False
    
    def _code_contains(self, component_pattern: str, search_term: str) -> bool:
        """Check if any Python file contains a specific term."""
        for root, dirs, files in os.walk(self.project_root):
            # Skip venv and cache directories
            if '.venv' in root or '__pycache__' in root or '.git' in root:
                continue
                
            for file in files:
                if component_pattern == '*' or component_pattern in file.lower():
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                if search_term.lower() in f.read().lower():
                                    return True
                        except (IOError, OSError):
                            pass  # File read error - skip this file
        return False
    
    def generate_report(self, audit_results: Dict) -> str:
        """Generate comprehensive audit report."""
        
        report_lines = [
            "="*100,
            "MARK5 COMPREHENSIVE SYSTEM AUDIT REPORT",
            "="*100,
            f"\nTimestamp: {audit_results['timestamp']}",
            f"Project Root: {audit_results['project_root']}",
            f"\nOVERALL STATUS: {audit_results['overall_status']}",
            f"Critical Issues: {audit_results['critical_issues']}",
            f"Warnings: {audit_results['warnings']}",
            f"Passed Checks: {audit_results['passed_checks']}",
            "\n" + "="*100,
            "STRENGTHS",
            "="*100
        ]
        
        for strength in self.strengths:
            report_lines.append(f"✅ {strength}")
        
        if audit_results['critical_issues'] > 0:
            report_lines.extend([
                "\n" + "="*100,
                "CRITICAL ISSUES (MUST FIX IMMEDIATELY)",
                "="*100
            ])
            
            for issue in [i for i in self.issues if i['severity'] == 'CRITICAL']:
                report_lines.append(f"\n❌ [{issue['component'].upper()}] {issue['issue']}")
                report_lines.append(f"   Fix: {issue['fix']}")
        
        if audit_results['warnings'] > 0:
            report_lines.extend([
                "\n" + "="*100,
                "WARNINGS",
                "="*100
            ])
            
            for issue in [i for i in self.issues if i['severity'] == 'WARNING']:
                report_lines.append(f"\n⚠️  [{issue['component'].upper()}] {issue['issue']}")
                report_lines.append(f"   Fix: {issue['fix']}")
        
        if self.missing_features:
            report_lines.extend([
                "\n" + "="*100,
                "MISSING FEATURES",
                "="*100
            ])
            
            for feature in sorted(self.missing_features, key=lambda x: x['priority'], reverse=True):
                priority_icon = "🔴" if feature['priority'] == 'CRITICAL' else "🟡" if feature['priority'] == 'HIGH' else "🟢"
                report_lines.append(f"\n{priority_icon} [{feature['priority']}] {feature['feature']}")
                report_lines.append(f"   {feature['description']}")
        
        if self.optimization_opportunities:
            report_lines.extend([
                "\n" + "="*100,
                "OPTIMIZATION OPPORTUNITIES",
                "="*100
            ])
            
            for opp in sorted(self.optimization_opportunities, key=lambda x: x['impact'], reverse=True):
                impact_icon = "⭐" if opp['impact'] == 'HIGH' else "💡"
                report_lines.append(f"\n{impact_icon} [{opp['area']}] {opp['opportunity']}")
                report_lines.append(f"   Impact: {opp['impact']}")
        
        report_lines.extend([
            "\n" + "="*100,
            "PRIORITY ACTION PLAN",
            "="*100,
            "\nIMMEDIATE (Week 1):"
        ])
        
        # Generate priority action plan
        immediate_actions = [i for i in self.issues if i['severity'] == 'CRITICAL']
        immediate_actions.extend([f for f in self.missing_features if f['priority'] == 'CRITICAL'])
        
        if immediate_actions:
            for idx, action in enumerate(immediate_actions[:5], 1):
                if 'issue' in action:
                    report_lines.append(f"{idx}. Fix: {action['issue']}")
                else:
                    report_lines.append(f"{idx}. Implement: {action['feature']}")
        else:
            report_lines.append("✅ No critical issues")
        
        report_lines.append("\nSHORT-TERM (Week 2-4):")
        short_term = [f for f in self.missing_features if f['priority'] == 'HIGH']
        for idx, feature in enumerate(short_term[:5], 1):
            report_lines.append(f"{idx}. Implement: {feature['feature']}")
        
        report_lines.append("\n" + "="*100)
        
        return "\n".join(report_lines)


def main():
    """Run system audit."""
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    auditor = SystemAuditor()
    results = auditor.audit_full_system()
    
    report = auditor.generate_report(results)
    print(report)
    
    # Save to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f"SYSTEM_AUDIT_REPORT_{timestamp}.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n📄 Full report saved to: {report_path}")
    
    # Save JSON results
    json_path = f"SYSTEM_AUDIT_RESULTS_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"📊 JSON results saved to: {json_path}")
    
    return results


if __name__ == "__main__":
    main()
