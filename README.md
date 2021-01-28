# kp-anonymity
Data Protection and Privacy - final project, (k,P)-anonymity

## Data


- https://www.kaggle.com/rohanrao/nifty50-stock-market-data?select=BRITANNIA.csv
- stolen from github 

- `stock_data_full.csv` - stock data, a few rows (50), many columns (1940)
  
- `weekly_transaction_full.csv` - transactions in weeks, many rows (811), a few columns (52)


## Run

Examples of how to run the program:


,,pr_value,2012-12-28,2013-02-15
1,(100,130),(100,400)
2,(100,130),(100,200)
3,(100,130),(100,500)


```shell
pip3 install -r requirements.txt
```

```shell
python3 src/kp_anonymity.py -k 4 -p 2 -a top-down
python3 src/kp_anonymity.py -k 4 -p 2 -a bottom-up
python3 src/kp_anonymity.py -k 4 -p 2 -a kapra
```

How to use 


