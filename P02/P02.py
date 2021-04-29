# Javier Rodríguez @doblerodriguez
# IN - GII - UGR 2020/2021

from pathlib import Path

import pandas as pd

filename = "mamografias.csv"
df = pd.read_csv(Path(__file__).parent / f"{filename}", na_values=['?'])
