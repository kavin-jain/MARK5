import os
import json
import pandas as pd

def audit_golden_army():
    root = "models"
    if not os.path.exists(root):
        print(f"Error: {root} directory not found.")
        return

    army = []
    for ticker in os.listdir(root):
        t_path = os.path.join(root, ticker)
        if not os.path.isdir(t_path): 
            continue
        
        # Get all version directories and sort them v1, v2, v3...
        try:
            versions = sorted(
                [v for v in os.listdir(t_path) if v.startswith('v') and os.path.isdir(os.path.join(t_path, v))], 
                key=lambda x: int(x[1:])
            )
        except (ValueError, IndexError):
            continue
            
        if not versions: 
            continue
        
        # Check the LATEST version for the gate pass
        latest_v = versions[-1]
        meta_path = os.path.join(t_path, latest_v, "metadata.json")
        
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                try:
                    m = json.load(f)
                    if m.get("passes_gate"):
                        army.append({
                            "Ticker": ticker,
                            "Version": m.get('version', latest_v),
                            "Timestamp": m.get('timestamp', 'N/A'),
                            "Gate": "READY"
                        })
                except json.JSONDecodeError:
                    continue
    
    if not army:
        print("\n⚠️ No symbols have passed the production gate yet.")
        return

    df = pd.DataFrame(army)
    print(f"\n🏆 MARK5 GOLDEN ARMY SUMMARY")
    print(f"Total Symbols with 'Gate-Pass': {len(df)}")
    print("-" * 50)
    print(df.to_string(index=False))
    print("-" * 50)

if __name__ == "__main__":
    audit_golden_army()
