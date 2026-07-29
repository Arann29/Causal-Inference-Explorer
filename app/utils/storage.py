"""
Storage utilities for results persistence
"""
import json
import sqlite3
import os
from datetime import datetime
from typing import Dict, Any, Optional

class ResultStorage:
    """Manage storage of analysis results"""
    
    def __init__(self, db_path: str = "/app/results/results.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with required tables"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Clustering results table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS clustering_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id TEXT NOT NULL,
                    method TEXT NOT NULL,
                    params TEXT NOT NULL,
                    result TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
            
            # Causal analysis results table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS causal_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id TEXT NOT NULL,
                    method TEXT NOT NULL,
                    clustering_id INTEGER,
                    result TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (clustering_id) REFERENCES clustering_results (id)
                )
            """)
            
            # LLM responses table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    response TEXT NOT NULL,
                    context TEXT,
                    model TEXT,
                    timestamp TEXT NOT NULL
                )
            """)
    
    def save_clustering_result(self, dataset_id: str, method: str, params: Dict[str, Any], result: Dict[str, Any]) -> int:
        """Save clustering result and return ID"""
        timestamp = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO clustering_results 
                (dataset_id, method, params, result, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (dataset_id, method, json.dumps(params), json.dumps(result), timestamp))
            
            return cursor.lastrowid
    
    def save_causal_result(self, dataset_id: str, method: str, result: Dict[str, Any], clustering_id: Optional[int] = None) -> int:
        """Save causal analysis result and return ID"""
        timestamp = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO causal_results 
                (dataset_id, method, clustering_id, result, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (dataset_id, method, clustering_id, json.dumps(result), timestamp))
            
            return cursor.lastrowid
    
    def save_llm_response(self, dataset_id: str, prompt: str, response: str, context: Optional[Dict[str, Any]] = None, model: str = "unknown") -> int:
        """Save LLM response and return ID"""
        timestamp = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO llm_responses 
                (dataset_id, prompt, response, context, model, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (dataset_id, prompt, response, json.dumps(context) if context else None, model, timestamp))
            
            return cursor.lastrowid
    
    def get_latest_results(self, dataset_id: str) -> Dict[str, Any]:
        """Get latest results for a dataset"""
        results = {}
        
        with sqlite3.connect(self.db_path) as conn:
            # Latest clustering
            cursor = conn.execute("""
                SELECT * FROM clustering_results 
                WHERE dataset_id = ? 
                ORDER BY timestamp DESC LIMIT 1
            """, (dataset_id,))
            clustering = cursor.fetchone()
            if clustering:
                results['clustering'] = {
                    'id': clustering[0],
                    'method': clustering[2],
                    'params': json.loads(clustering[3]),
                    'result': json.loads(clustering[4]),
                    'timestamp': clustering[5]
                }
            
            # Latest causal results
            cursor = conn.execute("""
                SELECT * FROM causal_results 
                WHERE dataset_id = ? 
                ORDER BY timestamp DESC
            """, (dataset_id,))
            causal_results = cursor.fetchall()
            results['causal'] = []
            for row in causal_results:
                results['causal'].append({
                    'id': row[0],
                    'method': row[2],
                    'result': json.loads(row[4]),
                    'timestamp': row[5]
                })
        
        return results