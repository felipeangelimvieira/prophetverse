from prophetverse.examples.repository import load_dataset, list_datasets


df = load_dataset('brazilian_unemployment_ibge')

from sidrapy import get_table

data = get_table(
   table_code="4093",
   territorial_level="1",
   ibge_territorial_code="all",
   period="all",
   category="39426"
).query("D3C=='4099'")

data