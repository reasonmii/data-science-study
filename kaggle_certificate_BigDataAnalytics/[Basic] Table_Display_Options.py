
import pandas as pd

# Method1
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Method2
pd.options.display.max_columns = None
pd.options.display.max_rows = None
