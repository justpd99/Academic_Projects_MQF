import pandas as pd
import glob

# Merge all CE and PE files for Maruti
ce_files = sorted(glob.glob('OPTSTK_MARUTI_CE_*.csv'))
pe_files = sorted(glob.glob('OPTSTK_MARUTI_PE_*.csv'))

all_files = ce_files + pe_files

merged_df = pd.DataFrame()
for file in all_files:
    df = pd.read_csv(file)
    merged_df = pd.concat([merged_df, df], ignore_index=True)

merged_df.to_csv('MARUTI_OPTIONS_MERGED.csv', index=False)
print('Merged options data saved to MARUTI_OPTIONS_MERGED.csv') 