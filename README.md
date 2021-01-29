# kp-anonymity
Data Protection and Privacy - final project, (k,P)-anonymity

## Data

The folder `data` contains all the testing files.

- https://www.kaggle.com/rohanrao/nifty50-stock-market-data?select=BRITANNIA.csv

- `stock_data_full.csv` - stock data, a few rows (50), many columns (1940)
- `weekly_transaction_full.csv` - transactions in weeks, many rows (811), a few columns (52)

- `...reduced` - one of the previous files with a reduced number of rows/columns

## Run

Examples of how to run the program:

### Run single algorithm on a single file with possible visualization

```shell
pip3 install -r requirements.txt
```

```shell
python3 src/kp_anonymity.py -k 4 -p 2 -a top-down -i data/stock_data_reduced.csv -o data/anonymized_table.csv -s
python3 src/kp_anonymity.py -k 4 -p 2 -a bottom-up -i data/stock_data_reduced.csv -o data/anonymized_table.csv -s
python3 src/kp_anonymity.py -k 4 -p 2 -a kapra -i data/stock_data_reduced.csv -o data/anonymized_table.csv -s
```

Parameters:

- `-k`, `--k-anonymity` - k value
- `-p`, `--p-anonymity` - p value (cannot be greater than k)
- `-l`, `--PR-length` - default 5, length of p-anonymity string
- `-m`, `--max-level` - default 5, max level in p-anonymity
- `-s`, `--show-plots` - if set, the envelopes will be visualized
- `-i`, `--input-file` - path to the input file
- `-o`, `--output-file` - path to the output file
- `-a`, `--algorithm` - must be one of the following: `top-down`, `bottom-up`, `kapra`
- `-v`, `--verbose` - if set, the verbose lines will be printed
- `-d`, `--debug` - if set, the debug lines will be printed

### Performance tests:

```shell
python3 src/performance_tests.py data/weekly_transactions_full.csv
```
