from prophetverse.examples.repository import load_dataset


class Preprocess:
    def __init__(self):
        self.load_layer = 'raw'
        self.save_layer = 'refined'
    
    def brazilian_unemployment_ibge(self):
        parse_dates
        date_formats
        columns
        #TODO
        df = (
            load_dataset(
                'brazilian_unemployment_ibge', 
                load_layer,
                #  dtype={'V': float}
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


        
        

        (df
         .loc[''])
'