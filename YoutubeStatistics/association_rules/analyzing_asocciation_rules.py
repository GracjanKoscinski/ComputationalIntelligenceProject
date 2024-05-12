import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Odczytanie danych
df = pd.read_csv('./../preprocessing/youtubeChannelsPreprocessed.csv')

# Przekształcenie kolumn tekstowych na format binarny
text_columns = ['youtuber', 'category', 'title', 'country', 'abbreviation', 'channel_type']
for col in text_columns:
    df[col] = df[col].astype(str).apply(lambda x: 1 if x else 0).astype(bool)


numeric_columns = ['rank', 'subscribers', 'video_views', 'video_views_rank', 'country_rank',
                   'video_views_for_the_last_30_days',
                   'lowest_monthly_earnings', 'highest_monthly_earnings', 'lowest_yearly_earnings',
                   'highest_yearly_earnings',
                   'subscribers_for_last_30_days', 'gross_tertiary_education_enrollment', 'population',
                   'unemployment_rate',
                   'urban_population']

for col in numeric_columns:
    df[col] = df[col].astype(str).apply(lambda x: 1 if x else 0).astype(bool)
# Stworzenie DataFrame z dummy variables dla kolumn tekstowych oraz numerycznych
basket = pd.get_dummies(df[text_columns + numeric_columns], drop_first=True)

# Zastosowanie algorytmu Apriori
frequent_itemsets = apriori(basket, min_support=0.11, use_colnames=True)

# Generowanie reguł asocjacyjnych
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head()

# Zapisanie reguł do pliku CSV
rules.to_csv('association_rules.csv', index=False)

# Wyniki bez sensu, bo za mało danych
