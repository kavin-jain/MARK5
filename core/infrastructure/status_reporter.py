#!/usr/bin/env python3
"""
MARK3 System Status Report Generator v3.0
Generate comprehensive system health and performance reports
"""

import os
import sys
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

class SystemStatusReport:
    """Generate comprehensive system status reports"""
    
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_path = os.path.join(self.base_dir, 'database/main/mark3.db')
    
    def get_model_status(self) -> Dict:
        """Get status of all trained models"""
        models_dir = os.path.join(self.base_dir, 'models')
        
        status = {
            'total_models': 0,
            'by_type': {},
            'by_ticker': {},
            'oldest_model': None,
            'newest_model': None,
            'avg_age_hours': 0
        }
        
        ages = []
        oldest_time = float('inf')
        newest_time = 0
        
        for model_type in ['xgboost', 'lightgbm', 'random_forest', 'catboost', 'ensemble']:
            type_dir = os.path.join(models_dir, model_type)
            if not os.path.exists(type_dir):
                continue
            
            files = [f for f in os.listdir(type_dir) if f.endswith('.pkl')]
            status['by_type'][model_type] = len(files)
            status['total_models'] += len(files)
            
            for file in files:
                ticker = file.split('_')[0]
                if ticker not in status['by_ticker']:
                    status['by_ticker'][ticker] = 0
                status['by_ticker'][ticker] += 1
                
                file_path = os.path.join(type_dir, file)
                mtime = os.path.getmtime(file_path)
                age = (datetime.now().timestamp() - mtime) / 3600
                ages.append(age)
                
                if mtime < oldest_time:
                    oldest_time = mtime
                    status['oldest_model'] = {
                        'file': file,
                        'age_hours': age
                    }
                
                if mtime > newest_time:
                    newest_time = mtime
                    status['newest_model'] = {
                        'file': file,
                        'age_hours': age
                    }
        
        if ages:
            status['avg_age_hours'] = sum(ages) / len(ages)
        
        return status
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        if not os.path.exists(self.db_path):
            return {'error': 'Database not found'}
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {
            'size_mb': os.path.getsize(self.db_path) / 1024 / 1024,
            'tables': {},
            'total_records': 0
        }
        
        # Get table counts
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                stats['tables'][table] = count
                stats['total_records'] += count
            except:
                stats['tables'][table] = 'Error'
        
        conn.close()
        return stats
    
    def get_recent_performance(self) -> Dict:
        """Get recent model performance summary"""
        if not os.path.exists(self.db_path):
            return {}
        
        conn = sqlite3.connect(self.db_path)
        
        # Get recent performance (last 7 days)
        cutoff = (datetime.now() - timedelta(days=7)).isoformat()
        
        query = '''
            SELECT ticker, model_type, accuracy, timestamp
            FROM model_performance_history
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
        '''
        
        cursor = conn.execute(query, (cutoff,))
        records = cursor.fetchall()
        
        if not records:
            conn.close()
            return {'message': 'No recent performance data'}
        
        # Aggregate by ticker
        by_ticker = {}
        for ticker, model_type, accuracy, timestamp in records:
            if ticker not in by_ticker:
                by_ticker[ticker] = []
            if accuracy:
                by_ticker[ticker].append(accuracy * 100)
        
        performance = {}
        for ticker, accuracies in by_ticker.items():
            if accuracies:
                performance[ticker] = {
                    'avg_accuracy': sum(accuracies) / len(accuracies),
                    'max_accuracy': max(accuracies),
                    'min_accuracy': min(accuracies),
                    'samples': len(accuracies)
                }
        
        conn.close()
        return performance
    
    def get_system_files(self) -> Dict:
        """Get count of system files"""
        stats = {
            'python_modules': 0,
            'config_files': 0,
            'log_files': 0,
            'data_files': 0,
            'total_size_mb': 0
        }
        
        for root, dirs, files in os.walk(self.base_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    size = os.path.getsize(file_path)
                    stats['total_size_mb'] += size / 1024 / 1024
                    
                    if file.endswith('.py'):
                        stats['python_modules'] += 1
                    elif file.endswith('.json') or file.endswith('.yaml'):
                        stats['config_files'] += 1
                    elif file.endswith('.log'):
                        stats['log_files'] += 1
                    elif file.endswith('.csv') or file.endswith('.pkl'):
                        stats['data_files'] += 1
                except:
                    pass
        
        return stats
    
    def generate_report(self, output_format: str = 'text') -> str:
        """
        Generate comprehensive status report
        
        Args:
            output_format: 'text' or 'json'
        """
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'models': self.get_model_status(),
            'database': self.get_database_stats(),
            'performance': self.get_recent_performance(),
            'files': self.get_system_files()
        }
        
        if output_format == 'json':
            return json.dumps(report_data, indent=2)
        
        # Text format
        lines = []
        lines.append("=" * 70)
        lines.append("MARK3 SYSTEM STATUS REPORT")
        lines.append("=" * 70)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Models Section
        models = report_data['models']
        lines.append("## TRAINED MODELS")
        lines.append("-" * 70)
        lines.append(f"Total Models: {models['total_models']}")
        lines.append("")
        
        if models['by_type']:
            lines.append("By Model Type:")
            for model_type, count in models['by_type'].items():
                lines.append(f"  {model_type:15} {count:3} models")
            lines.append("")
        
        if models['by_ticker']:
            lines.append("By Ticker:")
            for ticker, count in sorted(models['by_ticker'].items()):
                lines.append(f"  {ticker:15} {count:3} models")
            lines.append("")
        
        if models['avg_age_hours'] > 0:
            lines.append(f"Average Model Age: {models['avg_age_hours']:.1f} hours ({models['avg_age_hours']/24:.1f} days)")
            
            if models['oldest_model']:
                lines.append(f"Oldest Model: {models['oldest_model']['file']} ({models['oldest_model']['age_hours']/24:.1f} days)")
            
            if models['newest_model']:
                lines.append(f"Newest Model: {models['newest_model']['file']} ({models['newest_model']['age_hours']:.1f} hours)")
        
        lines.append("")
        
        # Database Section
        db = report_data['database']
        lines.append("## DATABASE")
        lines.append("-" * 70)
        
        if 'error' in db:
            lines.append(f"Status: {db['error']}")
        else:
            lines.append(f"Size: {db['size_mb']:.2f} MB")
            lines.append(f"Total Records: {db['total_records']:,}")
            lines.append("")
            lines.append("Table Record Counts:")
            for table, count in sorted(db['tables'].items()):
                if isinstance(count, int):
                    lines.append(f"  {table:25} {count:>10,}")
                else:
                    lines.append(f"  {table:25} {count}")
        
        lines.append("")
        
        # Performance Section
        perf = report_data['performance']
        lines.append("## RECENT PERFORMANCE (Last 7 Days)")
        lines.append("-" * 70)
        
        if 'message' in perf:
            lines.append(perf['message'])
        elif perf:
            lines.append(f"{'Ticker':<15} {'Avg Acc':>8} {'Max Acc':>8} {'Min Acc':>8} {'Samples':>8}")
            lines.append("-" * 70)
            for ticker, stats in sorted(perf.items()):
                lines.append(
                    f"{ticker:<15} "
                    f"{stats['avg_accuracy']:>7.1f}% "
                    f"{stats['max_accuracy']:>7.1f}% "
                    f"{stats['min_accuracy']:>7.1f}% "
                    f"{stats['samples']:>8}"
                )
        else:
            lines.append("No performance data available")
        
        lines.append("")
        
        # Files Section
        files = report_data['files']
        lines.append("## SYSTEM FILES")
        lines.append("-" * 70)
        lines.append(f"Python Modules: {files['python_modules']}")
        lines.append(f"Config Files: {files['config_files']}")
        lines.append(f"Log Files: {files['log_files']}")
        lines.append(f"Data Files: {files['data_files']}")
        lines.append(f"Total Size: {files['total_size_mb']:.2f} MB")
        lines.append("")
        
        # Recommendations
        lines.append("## RECOMMENDATIONS")
        lines.append("-" * 70)
        
        recommendations = []
        
        if models['avg_age_hours'] > 168:  # 1 week
            recommendations.append("⚠ Models are older than 1 week - consider retraining")
        
        if models['total_models'] < 10:
            recommendations.append("⚠ Low number of trained models - run training")
        
        if db.get('size_mb', 0) > 1000:  # 1GB
            recommendations.append("ℹ Database is large - consider archiving old data")
        
        if files['log_files'] > 100:
            recommendations.append("ℹ Many log files - consider cleanup")
        
        if not recommendations:
            recommendations.append("✓ System is in good health")
        
        for rec in recommendations:
            lines.append(f"  {rec}")
        
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def save_report(self, filename: str = None, output_format: str = 'text'):
        """Save report to file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            ext = 'json' if output_format == 'json' else 'txt'
            filename = f'system_status_{timestamp}.{ext}'
        
        report = self.generate_report(output_format)
        
        output_path = os.path.join(os.path.dirname(self.base_dir), filename)
        with open(output_path, 'w') as f:
            f.write(report)
        
        return output_path


if __name__ == '__main__':
    reporter = SystemStatusReport()
    
    # Generate and display report
    report = reporter.generate_report('text')
    print(report)
    
    # Optionally save to file
    if '--save' in sys.argv:
        output_path = reporter.save_report()
        print(f"\n✓ Report saved to: {output_path}")
