from pathlib import Path
import pandas as pd
import yaml

ROOT = Path(__name__).parent

data_dir = ROOT / 'data' / 'synthetic_data_example_1'
visu_dir = ROOT / 'visuals'

df = pd.read_csv(data_dir / 'data.csv')
with open(data_dir / 'parameters.yaml', 'r') as file: params = yaml.safe_load(file)

