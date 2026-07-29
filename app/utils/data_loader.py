"""
Data loading utilities - Connected to existing DATA/datasets.py
"""
import os
import pandas as pd
import sys
from typing import Dict, List, Any

# Add paths for existing code imports
sys.path.append('/app')
sys.path.append('/app/DATA')
sys.path.append('/app/app/utils')

from DATA.datasets import Custom

class DatasetService:
    """Service to load datasets using existing Custom loader"""
    
    def __init__(self, data_path: str = '/app/DATA/custom_pairs'):
        self.data_path = data_path
    
    def get_available_datasets(self) -> List[Dict[str, Any]]:
        """Get list of available datasets from custom_pairs directory"""
        datasets = []
        
        if not os.path.exists(self.data_path):
            return datasets
            
        try:
            for file in os.listdir(self.data_path):
                if file.endswith('.txt') and file != 'pairs.txt':
                    dataset_id = file.replace('.txt', '')
                    datasets.append({
                        'id': dataset_id,
                        'name': dataset_id.replace('_', ' ').title(),
                        'file': file
                    })
        except Exception as e:
            print(f"Error scanning datasets directory: {e}")
            
        return datasets

    def load_dataset(self, dataset_id: str) -> pd.DataFrame:
        """Load a specific dataset using existing Custom loader"""
        try:
            dataset = Custom(pair_id=dataset_id, path=self.data_path)
            return dataset.dataframe
        except Exception as e:
            raise Exception(f"Failed to load dataset {dataset_id}: {e}")
    
    def get_dataset_info(self, dataset_id: str) -> Dict[str, Any]:
        """Get detailed information about a dataset"""
        try:
            dataset = Custom(pair_id=dataset_id, path=self.data_path)
            df = dataset.dataframe
            
            return {
                'id': dataset_id,
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'summary': df.describe().to_dict()
            }
        except Exception as e:
            return {'error': str(e)}


# Global instance for easy import
dataset_service = DatasetService()