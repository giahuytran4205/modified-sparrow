import json
import pandas as pd
from decimal import Decimal, getcontext

pd.set_option('display.float_format', '{:.12f}'.format)

# Set precision for Decimal
getcontext().prec = 25
scale_factor = Decimal('1e15')

path = r'output\final_square_1.22.json'
with open(path) as f:
    obj = json.loads(f.read())

print(obj)

SCALE = 1

N = obj['items'][0]['demand']

df = pd.DataFrame()
df['id'] = [str(N).rjust(3, '0') + f'_{i}' for i in range(N)]
df['x'] = [Decimal(i['transformation']['translation'][0]) / SCALE for i in obj['solution']['layout']['placed_items']]
df['y'] = [Decimal(i['transformation']['translation'][1]) / SCALE for i in obj['solution']['layout']['placed_items']]
df['deg'] = [Decimal(i['transformation']['rotation']) for i in obj['solution']['layout']['placed_items']]

cols = ['x', 'y', 'deg']

for col in cols:
    df[col] = df[col].astype(float).round(decimals=6)

for col in cols:
    df[col] = 's' + df[col].astype('string')

df.to_csv('submission.csv', index=False)
print(df)