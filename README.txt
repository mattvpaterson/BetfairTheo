#Betfair_Static

This repo contains introductory work done on developing a pricing/research/backtesting framework for horse-racing markets on Betfair Exchange.

There are two section:

1. Book_Strategies
An attempt to develop HFT style order book trading strategies that are indiferrent to racing fundamentals.
Using historical market-by-level marketdata packets to construct order book snapshots.
Training and testing return predictions on constructed features to create signals for favourable predicted returns.
Aggresive strategy backtesting for out of sample profitability consideration.

2. Theo_Construction
The goal of this Pricing section is to create a system to calculate theoretical odds prices for horse/jockey combos listed in upcoming races. 
With these we can take expiry positions on incorrect prices or use theoretical values as a midpoint for market making strategies.
Training and testing statistical models with available meta data and devloping an appropriate scoring system to influence method changes.

In each section there are expositionary .ipynb files which explain the work and attached code in better detail/visualisation 
