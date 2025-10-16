"""
decision_persistence.py - Persistence för Beslutshistorik

Beskrivning:
    Lagrar och hämtar trading-beslut från fil/databas för långsiktig analys.
    Stödjer både JSON-fil och SQLite-databas.

Features:
    - Automatisk backup av beslut
    - Inkrementell save (inte hela history varje gång)
    - Load/Save med komprimering
    - Query API för historisk analys
    - Export till CSV/JSON

Användning:
    from modules.decision_persistence import DecisionPersistence
    
    persistence = DecisionPersistence('data/decisions.db')
    persistence.save_decision(decision, execution_result)
    history = persistence.load_decisions(symbol='AAPL', limit=100)
"""

import json
import sqlite3
import gzip
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import os


class DecisionPersistence:
    """Hanterar persistence av trading-beslut."""
    
    def __init__(self, db_path: str = 'data/decisions.db', use_sqlite: bool = True):
        """
        Initialiserar persistence layer.
        
        Args:
            db_path: Sökväg till databas-fil
            use_sqlite: Om True, använd SQLite, annars JSON
        """
        self.db_path = db_path
        self.use_sqlite = use_sqlite
        
        # Skapa data directory om den inte finns
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else 'data', exist_ok=True)
        
        if self.use_sqlite:
            self._init_sqlite()
        else:
            self.json_path = db_path.replace('.db', '.json.gz')
    
    def _init_sqlite(self):
        """Initialiserar SQLite-databas med schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Decisions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                confidence REAL,
                indicators TEXT,
                execution_success BOOLEAN,
                executed_price REAL,
                profit REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Indexes för snabbare queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol ON decisions(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON decisions(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_action ON decisions(action)')
        
        conn.commit()
        conn.close()
    
    def save_decision(self, decision: Dict[str, Any], execution_result: Optional[Dict[str, Any]] = None):
        """
        Sparar ett beslut till persistence.
        
        Args:
            decision: Beslutsobjekt med symbol, action, quantity, etc.
            execution_result: Optional execution resultat
        """
        timestamp = datetime.now().isoformat()
        
        if self.use_sqlite:
            self._save_to_sqlite(decision, execution_result, timestamp)
        else:
            self._save_to_json(decision, execution_result, timestamp)
    
    def _save_to_sqlite(self, decision: Dict, execution_result: Optional[Dict], timestamp: str):
        """Sparar till SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO decisions 
            (timestamp, symbol, action, quantity, price, confidence, indicators,
             execution_success, executed_price, profit)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp,
            decision.get('symbol', ''),
            decision.get('action', 'HOLD'),
            decision.get('quantity', 0),
            decision.get('price', 0.0),
            decision.get('confidence', 0.0),
            json.dumps(decision.get('indicators', {})),
            execution_result.get('success', False) if execution_result else None,
            execution_result.get('executed_price') if execution_result else None,
            execution_result.get('profit') if execution_result else None
        ))
        
        conn.commit()
        conn.close()
    
    def _save_to_json(self, decision: Dict, execution_result: Optional[Dict], timestamp: str):
        """Sparar till komprimerad JSON-fil."""
        # Ladda befintlig data
        existing_data = []
        if os.path.exists(self.json_path):
            with gzip.open(self.json_path, 'rt') as f:
                existing_data = json.load(f)
        
        # Lägg till nytt beslut
        record = {
            'timestamp': timestamp,
            'decision': decision,
            'execution': execution_result
        }
        existing_data.append(record)
        
        # Spara tillbaka
        with gzip.open(self.json_path, 'wt') as f:
            json.dump(existing_data, f)
    
    def load_decisions(self, 
                      symbol: Optional[str] = None,
                      action: Optional[str] = None,
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None,
                      limit: int = 1000) -> List[Dict]:
        """
        Laddar beslut från persistence med filtering.
        
        Args:
            symbol: Filtrera på symbol (None = alla)
            action: Filtrera på action (BUY/SELL/HOLD)
            start_date: Från datum
            end_date: Till datum
            limit: Max antal resultat
            
        Returns:
            Lista av beslut
        """
        if self.use_sqlite:
            return self._load_from_sqlite(symbol, action, start_date, end_date, limit)
        else:
            return self._load_from_json(symbol, action, start_date, end_date, limit)
    
    def _load_from_sqlite(self, symbol, action, start_date, end_date, limit) -> List[Dict]:
        """Laddar från SQLite med filters."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = 'SELECT * FROM decisions WHERE 1=1'
        params = []
        
        if symbol:
            query += ' AND symbol = ?'
            params.append(symbol)
        
        if action:
            query += ' AND action = ?'
            params.append(action)
        
        if start_date:
            query += ' AND timestamp >= ?'
            params.append(start_date.isoformat())
        
        if end_date:
            query += ' AND timestamp <= ?'
            params.append(end_date.isoformat())
        
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        
        columns = [desc[0] for desc in cursor.description]
        results = []
        
        for row in cursor.fetchall():
            record = dict(zip(columns, row))
            # Parse JSON indicators
            if record.get('indicators'):
                record['indicators'] = json.loads(record['indicators'])
            results.append(record)
        
        conn.close()
        return results
    
    def _load_from_json(self, symbol, action, start_date, end_date, limit) -> List[Dict]:
        """Laddar från JSON med filters."""
        if not os.path.exists(self.json_path):
            return []
        
        with gzip.open(self.json_path, 'rt') as f:
            all_data = json.load(f)
        
        # Filter
        filtered = all_data
        
        if symbol:
            filtered = [d for d in filtered if d['decision'].get('symbol') == symbol]
        
        if action:
            filtered = [d for d in filtered if d['decision'].get('action') == action]
        
        if start_date:
            filtered = [d for d in filtered if datetime.fromisoformat(d['timestamp']) >= start_date]
        
        if end_date:
            filtered = [d for d in filtered if datetime.fromisoformat(d['timestamp']) <= end_date]
        
        # Sort and limit
        filtered.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return filtered[:limit]
    
    def get_statistics(self, symbol: Optional[str] = None, days: int = 30) -> Dict:
        """
        Hämtar statistik för beslut.
        
        Args:
            symbol: Symbol att analysera (None = alla)
            days: Antal dagar bakåt
            
        Returns:
            Dict med statistik
        """
        start_date = datetime.now() - timedelta(days=days)
        decisions = self.load_decisions(symbol=symbol, start_date=start_date, limit=10000)
        
        if not decisions:
            return {
                'total_decisions': 0,
                'buy_count': 0,
                'sell_count': 0,
                'hold_count': 0,
                'success_rate': 0.0,
                'avg_profit': 0.0
            }
        
        total = len(decisions)
        buy_count = sum(1 for d in decisions if d.get('action') == 'BUY')
        sell_count = sum(1 for d in decisions if d.get('action') == 'SELL')
        hold_count = sum(1 for d in decisions if d.get('action') == 'HOLD')
        
        successful = sum(1 for d in decisions if d.get('execution_success'))
        success_rate = successful / total if total > 0 else 0.0
        
        profits = [d.get('profit', 0) for d in decisions if d.get('profit') is not None]
        avg_profit = sum(profits) / len(profits) if profits else 0.0
        
        return {
            'total_decisions': total,
            'buy_count': buy_count,
            'sell_count': sell_count,
            'hold_count': hold_count,
            'success_rate': success_rate,
            'avg_profit': avg_profit,
            'period_days': days
        }
    
    def export_to_csv(self, output_path: str, symbol: Optional[str] = None, limit: int = 10000):
        """
        Exporterar beslut till CSV.
        
        Args:
            output_path: Sökväg till output CSV-fil
            symbol: Optional symbol filter
            limit: Max antal rader
        """
        import csv
        
        decisions = self.load_decisions(symbol=symbol, limit=limit)
        
        if not decisions:
            print("Inga beslut att exportera")
            return
        
        with open(output_path, 'w', newline='') as csvfile:
            # Bestäm kolumner baserat på första posten
            if self.use_sqlite:
                fieldnames = ['timestamp', 'symbol', 'action', 'quantity', 'price', 
                             'confidence', 'execution_success', 'executed_price', 'profit']
            else:
                fieldnames = ['timestamp', 'symbol', 'action', 'quantity', 'price', 'confidence']
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            
            for decision in decisions:
                if self.use_sqlite:
                    writer.writerow(decision)
                else:
                    row = {'timestamp': decision['timestamp']}
                    row.update(decision['decision'])
                    writer.writerow(row)
        
        print(f"✓ Exporterade {len(decisions)} beslut till {output_path}")
    
    def cleanup_old_decisions(self, days_to_keep: int = 90):
        """
        Tar bort gamla beslut (äldre än X dagar).
        
        Args:
            days_to_keep: Antal dagar att behålla
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        if self.use_sqlite:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM decisions WHERE timestamp < ?', (cutoff_date.isoformat(),))
            deleted = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            print(f"✓ Tog bort {deleted} gamla beslut (äldre än {days_to_keep} dagar)")
        else:
            if not os.path.exists(self.json_path):
                return
            
            with gzip.open(self.json_path, 'rt') as f:
                all_data = json.load(f)
            
            filtered = [d for d in all_data 
                       if datetime.fromisoformat(d['timestamp']) >= cutoff_date]
            
            with gzip.open(self.json_path, 'wt') as f:
                json.dump(filtered, f)
            
            deleted = len(all_data) - len(filtered)
            print(f"✓ Tog bort {deleted} gamla beslut (äldre än {days_to_keep} dagar)")


# CLI för testing
if __name__ == "__main__":
    print("=" * 60)
    print("Decision Persistence Test")
    print("=" * 60)
    
    # Test SQLite
    print("\n1. Test SQLite persistence...")
    persistence = DecisionPersistence('data/test_decisions.db', use_sqlite=True)
    
    # Spara test-beslut
    for i in range(5):
        decision = {
            'symbol': 'AAPL' if i % 2 == 0 else 'TSLA',
            'action': 'BUY' if i % 3 == 0 else 'SELL',
            'quantity': i + 1,
            'price': 150.0 + i,
            'confidence': 0.7 + (i * 0.05),
            'indicators': {'RSI': 50 + i}
        }
        
        execution = {
            'success': True,
            'executed_price': 150.0 + i + 0.1,
            'profit': i * 2.5
        }
        
        persistence.save_decision(decision, execution)
    
    print("✓ Sparade 5 test-beslut")
    
    # Ladda beslut
    all_decisions = persistence.load_decisions(limit=10)
    print(f"✓ Laddade {len(all_decisions)} beslut")
    
    # Statistik
    stats = persistence.get_statistics(days=1)
    print(f"\n📊 Statistik:")
    print(f"   Total: {stats['total_decisions']}")
    print(f"   BUY: {stats['buy_count']}, SELL: {stats['sell_count']}")
    print(f"   Success rate: {stats['success_rate']:.1%}")
    print(f"   Avg profit: ${stats['avg_profit']:.2f}")
    
    # Export
    persistence.export_to_csv('data/test_export.csv')
    
    print("\n✅ Test completed!")
