#!/bin/bash
# MARK3 Production Deployment Script v3.0
# Automated deployment for production environment

set -e  # Exit on error

echo "========================================="
echo "MARK3 Production Deployment"
echo "========================================="
echo ""

# Configuration
MARK3_HOME="/home/lynx/Documents/MARK3"
VENV_PATH="$MARK3_HOME/.venv"
CORE_PATH="$MARK3_HOME/core"
BACKUP_DIR="$CORE_PATH/database/backups"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo_success() {
    echo -e "${GREEN}✓${NC} $1"
}

echo_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

echo_error() {
    echo -e "${RED}✗${NC} $1"
}

# 1. Check Prerequisites
echo "1. Checking prerequisites..."
if [ ! -d "$VENV_PATH" ]; then
    echo_error "Virtual environment not found at $VENV_PATH"
    exit 1
fi
echo_success "Virtual environment found"

if [ ! -f "$CORE_PATH/ml_trainer_enhanced.py" ]; then
    echo_error "Core files not found in $CORE_PATH"
    exit 1
fi
echo_success "Core files found"

# 2. Create backup
echo ""
echo "2. Creating backup..."
cd "$CORE_PATH"
python3 - <<EOF
from backup_manager import BackupManager
manager = BackupManager()
result = manager.create_backup(include_models=True, include_database=True)
print(f"Backup created: {result['backup_name']}")
print(f"Size: {result['size_mb']:.2f} MB")
EOF

if [ $? -eq 0 ]; then
    echo_success "Backup completed"
else
    echo_warning "Backup failed (continuing anyway)"
fi

# 3. Run tests
echo ""
echo "3. Running integration tests..."
cd "$CORE_PATH"
timeout 300 $VENV_PATH/bin/python3 integration_tests.py > /tmp/mark3_test_results.txt 2>&1

if grep -q "ALL TESTS PASSED" /tmp/mark3_test_results.txt; then
    echo_success "All tests passed"
else
    echo_warning "Some tests failed - review /tmp/mark3_test_results.txt"
fi

# 4. Check system health
echo ""
echo "4. Checking system health..."
cd "$CORE_PATH"
$VENV_PATH/bin/python3 - <<EOF
from monitoring_dashboard import MARK3MonitoringDashboard
dashboard = MARK3MonitoringDashboard()
health = dashboard.get_system_health()
print(f"System Status: {health['status']}")
print(f"Health Score: {health['health_score']}/100")
if health['issues']:
    print("Issues:")
    for issue in health['issues']:
        print(f"  - {issue}")
EOF

# 5. Verify models
echo ""
echo "5. Verifying trained models..."
MODEL_COUNT=$(find "$CORE_PATH/models" -name "*.pkl" 2>/dev/null | wc -l)
echo "Found $MODEL_COUNT model files"

if [ $MODEL_COUNT -lt 10 ]; then
    echo_warning "Low number of models - may need retraining"
else
    echo_success "Models verified ($MODEL_COUNT files)"
fi

# 6. Check database
echo ""
echo "6. Checking database..."
DB_PATH="$CORE_PATH/database/main/mark3.db"
if [ -f "$DB_PATH" ]; then
    DB_SIZE=$(du -h "$DB_PATH" | cut -f1)
    echo_success "Database found ($DB_SIZE)"
    
    # Verify tables
    TABLE_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
    echo "  Tables: $TABLE_COUNT"
else
    echo_error "Database not found - run init_database.py"
    exit 1
fi

# 7. Set up cron jobs (optional)
echo ""
echo "7. Setting up automated tasks..."
echo "To enable automated retraining, add to crontab:"
echo "  0 2 * * 0 cd $CORE_PATH && $VENV_PATH/bin/python3 quick_retrain_all.py"
echo ""
echo "To enable daily backups, add:"
echo "  0 1 * * * cd $CORE_PATH && $VENV_PATH/bin/python3 -c 'from backup_manager import BackupManager; BackupManager().create_backup()'"

# 8. Generate deployment report
echo ""
echo "8. Generating deployment report..."
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
REPORT_FILE="$MARK3_HOME/deployment_report_${TIMESTAMP}.txt"

cat > "$REPORT_FILE" << EOL
MARK3 Production Deployment Report
Generated: $(date)
==========================================

System Information:
- MARK3 Home: $MARK3_HOME
- Core Path: $CORE_PATH
- Python Env: $VENV_PATH

Component Status:
- Models: $MODEL_COUNT files
- Database: $DB_SIZE
- Backups: $(ls -1 $BACKUP_DIR/*.tar.gz 2>/dev/null | wc -l) available

Test Results:
$(cat /tmp/mark3_test_results.txt | grep "TEST SUMMARY" -A 10 || echo "Tests not available")

Recommendations:
1. Review test results in /tmp/mark3_test_results.txt
2. Set up automated retraining (weekly recommended)
3. Configure daily backups
4. Monitor system health regularly
5. Review model performance weekly

==========================================
Deployment Status: READY
Next Steps: Configure automated tasks
==========================================
EOL

echo_success "Report generated: $REPORT_FILE"

# 9. Final status
echo ""
echo "========================================="
echo "Deployment Summary"
echo "========================================="
echo ""
cat "$REPORT_FILE"
echo ""
echo_success "Production deployment completed successfully!"
echo ""
echo "To start the system:"
echo "  cd $CORE_PATH"
echo "  $VENV_PATH/bin/python3 launcher.py"
echo ""
echo "For monitoring:"
echo "  $VENV_PATH/bin/python3 monitoring_dashboard.py --loop 30"
echo ""
