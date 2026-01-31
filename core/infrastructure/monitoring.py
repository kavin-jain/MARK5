#!/usr/bin/env python3
"""
MARK3 COMMAND CENTER (TUI)
--------------------------
Professional HFT Dashboard using Curses.
Flicker-free, Low-Latency monitoring.
"""

import curses
import time
import psutil
import os
import sqlite3
import threading
from datetime import datetime
from core.infrastructure.redis_io import get_redis_manager

class DashboardData:
    """Thread-safe data container for the TUI"""
    def __init__(self):
        self.cpu = 0.0
        self.ram = 0.0
        self.pnl = 0.0
        self.trade_count = 0
        self.models = []
        self.system_status = "INITIALIZING"
        self.redis_ping = False
        self.lock = threading.Lock()

    def update_system(self):
        """Runs in background thread"""
        while True:
            try:
                # System Stats
                c = psutil.cpu_percent(interval=1)
                m = psutil.virtual_memory().percent
                
                # Redis Check
                r = get_redis_manager()
                rp = False
                try:
                    rp = r.client.ping()
                except: pass
                
                with self.lock:
                    self.cpu = c
                    self.ram = m
                    self.redis_ping = rp
                    self.system_status = "ONLINE" if rp else "REDIS DOWN"

            except Exception as e:
                with self.lock:
                    self.system_status = f"ERROR: {str(e)[:10]}"
            time.sleep(0.5)

    def update_db_stats(self, db_path):
        """Runs in background thread - Slower poll"""
        while True:
            try:
                # Use Read-Only mode explicitly
                uri = f"file:{db_path}?mode=ro"
                conn = sqlite3.connect(uri, uri=True, timeout=1)
                
                # Get PnL (Mock query based on your schema)
                # Ensure schema matches what we defined in DB manager
                try:
                    cur = conn.execute("SELECT COUNT(*) FROM trade_journal WHERE status='CLOSED'")
                    count = cur.fetchone()[0]
                    
                    # Mock PnL query - replace with actual sum
                    cur = conn.execute("SELECT SUM(net_pnl) FROM trade_journal WHERE status='CLOSED'")
                    res = cur.fetchone()[0]
                    pnl = float(res) if res else 0.0
                except:
                    count = 0
                    pnl = 0.0

                conn.close()
                
                with self.lock:
                    self.trade_count = count
                    self.pnl = pnl
            except:
                pass
            time.sleep(2)

def draw_bar(stdscr, y, x, val, width, label, color_pair):
    """Draws a progress bar"""
    stdscr.addstr(y, x, f"{label}: ", curses.A_BOLD)
    
    # Calculate bar length
    bar_width = width - len(label) - 10
    fill = int((val / 100.0) * bar_width)
    
    stdscr.addstr("[")
    stdscr.attron(color_pair)
    stdscr.addstr("|" * fill)
    stdscr.attroff(color_pair)
    stdscr.addstr("-" * (bar_width - fill))
    stdscr.addstr(f"] {val:5.1f}%")

def main(stdscr):
    # Setup Curses
    curses.curs_set(0) # Hide cursor
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN, -1)
    curses.init_pair(2, curses.COLOR_RED, -1)
    curses.init_pair(3, curses.COLOR_YELLOW, -1)
    curses.init_pair(4, curses.COLOR_CYAN, -1)

    data = DashboardData()
    
    # Locate DB
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming standard path, adjust if needed
    db_path = os.path.join(base_dir, '../../database/main/mark3.db')

    # Start Background Threads
    t1 = threading.Thread(target=data.update_system, daemon=True)
    t2 = threading.Thread(target=data.update_db_stats, args=(db_path,), daemon=True)
    t1.start()
    t2.start()

    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        
        # Header
        title = " MARK 3 | AUTONOMOUS TRADING ENGINE | SYSTEM MONITOR "
        stdscr.attron(curses.color_pair(4) | curses.A_REVERSE)
        stdscr.addstr(0, 0, title + " " * (w - len(title) - 1))
        stdscr.attroff(curses.color_pair(4) | curses.A_REVERSE)
        
        # System Health
        with data.lock:
            cpu = data.cpu
            ram = data.ram
            status = data.system_status
            redis_ok = data.redis_ping
            pnl = data.pnl
            trades = data.trade_count

        # Draw Stats
        stdscr.addstr(2, 2, f"SYSTEM STATUS: ", curses.A_BOLD)
        if status == "ONLINE":
            stdscr.addstr(status, curses.color_pair(1) | curses.A_BOLD)
        else:
            stdscr.addstr(status, curses.color_pair(2) | curses.A_BLINK)

        stdscr.addstr(2, 40, f"REDIS CONNECTION: ")
        if redis_ok:
            stdscr.addstr("ACTIVE", curses.color_pair(1))
        else:
            stdscr.addstr("FAILED", curses.color_pair(2))

        # Resource Bars
        col = curses.color_pair(1) if cpu < 50 else curses.color_pair(3) if cpu < 80 else curses.color_pair(2)
        draw_bar(stdscr, 4, 2, cpu, 50, "CPU CORE", col)
        
        col = curses.color_pair(1) if ram < 50 else curses.color_pair(3) if ram < 80 else curses.color_pair(2)
        draw_bar(stdscr, 5, 2, ram, 50, "RAM USAGE", col)

        # Financials (The important part)
        stdscr.hline(7, 2, curses.ACS_HLINE, w-4)
        stdscr.addstr(8, 2, "FINANCIAL METRICS", curses.color_pair(4) | curses.A_BOLD)
        
        stdscr.addstr(10, 4, f"TOTAL TRADES: {trades}")
        
        stdscr.addstr(10, 35, "NET PNL (DAY): ")
        pnl_str = f"₹ {pnl:,.2f}"
        if pnl >= 0:
            stdscr.addstr(pnl_str, curses.color_pair(1) | curses.A_BOLD)
        else:
            stdscr.addstr(pnl_str, curses.color_pair(2) | curses.A_BOLD)

        # Footer
        stdscr.addstr(h-2, 2, f"Time: {datetime.now().strftime('%H:%M:%S.%f')[:-3]} | Press Ctrl+C to Exit")

        stdscr.refresh()
        time.sleep(0.1)

if __name__ == "__main__":
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        print("Dashboard Closed.")
