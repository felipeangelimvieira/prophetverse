from prophetverse.examples.repository.repositories import load_dataset, save_dataset
import pandas as pd

class Preprocess:
    def __init__(self):
        self.load_layer = 'raw'
        self.save_layer = 'refined'
    
    def brazilian_unemployment_ibge(self, save=True):
        df = (
            load_dataset(
                'brazilian_unemployment_ibge', 
                self.load_layer,
                 )
              .query("D3C=='4099'")
              .loc[:, ['D2C', 'D1N' , 'V']]
              .rename(
                  columns=  {
                        'V':'Unemployment Rate',
                        'D1N':'Country',
                        'D2C':'Date'
                    }
              )
          
              )
        df['Date'] = pd.to_datetime(df['Date'].astype(str), format='%Y%m')
        df = df.set_index('Date')
        if save:
            save_dataset(df, 'brazilian_unemployment_ibge', index=False)


