import yfinance as yf
import pandas as pd
from datetime import datetime


symbol = 'THYAO.IS'


stock_data = yf.download(symbol, start='2002-01-01')


turkish_holidays = [
    '01-01',  
    '04-23',  
    '05-01',  
    '05-19',  
    '07-15',  
    '08-30', 
    '10-29',  
]


ramazan_bayrami_tarihleri = [
    '2002-03-13', '2003-03-03', '2004-02-22', '2005-02-10', '2006-01-30', '2007-09-13',
    '2008-09-30', '2009-09-20', '2010-09-10', '2011-08-30', '2012-08-19', '2013-08-08',
    '2014-07-28', '2015-07-17', '2016-07-05', '2017-06-24', '2018-06-14', '2019-06-04',
    '2020-05-24', '2021-05-13', '2022-05-02', '2023-04-21'
]

kurban_bayrami_tarihleri = [
    '2002-03-24', '2003-03-13', '2004-03-02', '2005-02-21', '2006-02-09', '2007-09-24',
    '2008-10-10', '2009-09-30', '2010-09-19', '2011-09-06', '2012-08-26', '2013-08-15',
    '2014-09-04', '2015-09-24', '2016-09-12', '2017-09-01', '2018-08-21', '2019-08-11',
    '2020-07-31', '2021-07-20', '2022-07-09', '2023-06-28'
]


df = stock_data[~((stock_data.index.weekday >= 5) |  
                 (stock_data.index.strftime('%m-%d').isin(turkish_holidays)) |  # Resmi tatiller
                 (stock_data.index.strftime('%Y-%m-%d').isin(ramazan_bayrami_tarihleri)) |  # Ramazan Bayramı
                 (stock_data.index.strftime('%Y-%m-%d').isin(kurban_bayrami_tarihleri)))]  # Kurban Bayramı


last_close_price = df['Close'].iloc[-1]
last_date = df.index[-1].date()


first_close_price = df['Close'].iloc[0]
percentage_change = ((last_close_price - first_close_price) / first_close_price) * 100


df = pd.DataFrame({
    'Tarih': df.index,
    'Kapanış Fiyatı': df['Close']
})
df['Yüzde Değişim'] = (df['Kapanış Fiyatı'].pct_change() * 100).fillna(0)


df.to_csv('thyao.csv', index=False)


print(f"Hisse: {symbol}")
print(f"İlk Kapanış Fiyatı: {first_close_price}")
print(f"Son Kapanış Fiyatı: {last_close_price}")
print(f"Son 10 Yıllık Yüzde Değişim: {percentage_change}%")
print(f"Tarih: {last_date}")
