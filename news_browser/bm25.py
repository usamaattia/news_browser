import pandas as pd

class bm():
    def data(self):
        df = pd.read_json('News_Dataset.json')
        
        return df