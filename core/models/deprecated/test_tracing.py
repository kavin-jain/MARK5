import sys
print("Starting...")
try:
    print("Importing TradeJournal...")
    from core.analytics.journal import TradeJournal
    print("Importing MARK5DatabaseManager...")
    from core.infrastructure.database_manager import MARK5DatabaseManager
    print("Importing RiskManager...")
    from core.trading.risk_manager import RiskManager, RiskAlerts, PortfolioRiskAnalyzer
    print("Importing ExecutionEngine...")
    from core.execution.execution_engine import ExecutionEngine
    print("Done")
except Exception as e:
    print(f"Error: {e}")
