Project in the course Machine Learning for Portfolio Management and Trading

to install required packages run 

```bash
py -m pip install -r requirements.txt
```

To download market data (currency returns) and macro indicators time seirs from the FRED API

```bash
py -m GetMarketdata
```


To download FOMC event dates

```bash
py -m FomcGetCalendarData
```

To run the model 

```bash
py -m main
```

If everything runs smoothly, you will be able to run the notebook to analyze the results
