import pandas as pd

FILEPATH = 'results.csv'
pd.read_csv(FILEPATH).to_html('results.html', escape=False)