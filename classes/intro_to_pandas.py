import pandas as pd
print("Pandas version is", pd.__version__)

# Series are single column
citySeries = pd.Series(["San Francisco", "San Jose", "Sacramento"])
print("City")
print(citySeries)

# DataFrame are like relational data tables, can have 1+ series with labels.
population = pd.Series([852469, 1015785, 485199])
cityPopulation = pd.DataFrame({'City Name': citySeries, 'Population': population})
print("Data Frame")
print(cityPopulation)

california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")
print("Describe the data")
print(california_housing_dataframe.describe())

print("Print the first few results")
print(california_housing_dataframe.head())

cities = pd.DataFrame({ 'City name': citySeries, 'Population': population })
print(type(cities['City name']))
print(cities['City name'])

cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['Population density'] = cities['Population'] / cities['Area square miles']

# If City starts with San, and Population Density is 50k+
cities['saint'] = citySeries.apply(lambda val: val[0:3] == "San")
cities['50 sqmile'] = cities['Population density'].apply(lambda val: val > 50)
cities['bool'] = cities['saint'] & cities['50 sqmile']
print(cities)

# One Line Solution
cities['Is wide and has saint name'] = (cities['Area square miles'] > 50) & cities['City name'].apply(lambda name: name.startswith('San'))
