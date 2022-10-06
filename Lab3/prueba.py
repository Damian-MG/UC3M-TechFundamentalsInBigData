import pyjstat

url = 'http://ec.europa.eu/eurostat/wdds/rest/data/v2.1/json/en/ei_bsco_m?indic=BS-CSMCI&precision=1&unit=BAL&s_adj=SA'

dataset = pyjstat.Dataset.read(url)

df = dataset.write('dataframe')
print(df)