### This data came from brazilian official government API.

```
 from sidrapy import get_table


 data = get_table(
   table_code="4093",
   territorial_level="1",
   ibge_territorial_code="all",
   period="all",
   category="39426"
 )
```