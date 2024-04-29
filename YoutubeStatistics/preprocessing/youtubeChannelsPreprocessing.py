import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
df = pd.read_csv("Global YouTube Statistics.csv", encoding='latin-1')
pd.set_option('display.max_columns', 200)
# print(df.head())
# Złączenie kolumn 'created_year', 'created_month', 'created_date' w jedną kolumnę 'created_at'
df['created_year'] = df['created_year'].astype(str).str.split('.').str[0]
df['created_month'] = df['created_month'].astype(str)
df['created_date'] = df['created_date'].astype(str).str.split('.').str[0]
df['created_at'] = df['created_date'] + '-' + df['created_month'] + '-' + df['created_year']
df['created_at'] = df['created_at'].replace("nan-nan-nan", np.nan)
df['created_at'] = pd.to_datetime(df['created_at'])
df.drop(['created_year', 'created_month', 'created_date'], axis=1, inplace=True)


# zmiana nazwy na nazwę bez spacji
df.rename(columns={"video views": "video_views", "Unemployment rate": "unemployment_rate"}, inplace=True)
df.rename(columns={"Gross tertiary education enrollment (%)": "gross_tertiary_education_enrollment"}, inplace=True)
# zamiana, aby każda kolumna zaczynała się małą literą
df.columns = [col.lower() for col in df.columns]

# Obsługa wartości nan
# print(df.isnull().sum())
# rank                                     0
# youtuber                                 0
# subscribers                              0
# video_views                              0
# category                                46
# title                                    0
# uploads                                  0
# country                                 25
# abbreviation                            26
# channel_type                            30
# video_views_rank                         1
# country_rank                           116
# channel_type_rank                       33
# video_views_for_the_last_30_days        56
# lowest_monthly_earnings                  0
# highest_monthly_earnings                 0
# lowest_yearly_earnings                   0
# highest_yearly_earnings                  0
# subscribers_for_last_30_days           337
# gross_tertiary_education_enrollment    123
# population                             123
# unemployment_rate                      123
# urban_population                       123
# latitude                               123
# longitude                              123
# created_at                               5

# Więc wszystko gdzie występuje nan, trzeba jakoś przerobić lub usunąć

# pozbycie się latitude i longitude
df.drop(['latitude', 'longitude'], axis=1, inplace=True)
# pozbycie się channel_type_rank
df.drop(['channel_type_rank'], axis=1, inplace=True)
# usunięcie kolumn gdzie video_views lub uploades jest 0
df = df[df['video_views'] != 0.0]
df = df[df['uploads'] != 0.0]
# usunięcie tam gdzie country_rank jest nan
df = df.dropna(subset=['country_rank'])
# usuń te kanały gdzie w youtuber lub title jest coś z "ý", lub "½", lub "¿", lub "ï", lub "ż"
df = df[~df['youtuber'].str.contains('ý')]
df = df[~df['youtuber'].str.contains('½')]
df = df[~df['youtuber'].str.contains('¿')]
df = df[~df['youtuber'].str.contains('ï')]
df = df[~df['youtuber'].str.contains('ż')]
df = df[~df['title'].str.contains('ý')]
df = df[~df['title'].str.contains('½')]
df = df[~df['title'].str.contains('¿')]
df = df[~df['title'].str.contains('ï')]
df = df[~df['title'].str.contains('ż')]

# użycie logistic regression do przewidzenia wartości dla 'category' i 'channel_type'

label_encoder = LabelEncoder()
df['encoded_youtuber'] = label_encoder.fit_transform(df['youtuber'])
df['encoded_title'] = label_encoder.fit_transform(df['title'])

train_df = df.dropna(subset=['category'])
test_df = df[df['category'].isnull()]

features = ['encoded_youtuber', 'encoded_title']
target = 'category'

logistic_regression = LogisticRegression(max_iter=1000000)
logistic_regression.fit(train_df[features], train_df[target])
predictions = logistic_regression.predict(test_df[features])
df.loc[df['category'].isnull(), 'category'] = predictions

train_df = df.dropna(subset=['channel_type'])
test_df = df[df['channel_type'].isnull()]
features = ['encoded_youtuber', 'encoded_title']
target = 'channel_type'
logistic_regression = LogisticRegression(max_iter=1000000)
logistic_regression.fit(train_df[features], train_df[target])
predictions = logistic_regression.predict(test_df[features])
df.loc[df['channel_type'].isnull(), 'channel_type'] = predictions

# użycie LinearRegression do przewidzenia wartości dla 'subscribers_for_last_30_days'
# oraz 'video_views_for_the_last_30_days'
df['encoded_subscribers'] = label_encoder.fit_transform(df['subscribers'])
df['encoded_video_views'] = label_encoder.fit_transform(df['video_views'])

features = ['encoded_subscribers', 'encoded_video_views']
target = 'subscribers_for_last_30_days'

linear_regression = LinearRegression()
train_df = df.dropna(subset=[target])
test_df = df[df[target].isnull()]

linear_regression.fit(train_df[features], train_df[target])
predictions = linear_regression.predict(test_df[features])

df.loc[df[target].isnull(), target] = predictions

features = ['encoded_subscribers', 'encoded_video_views']
target = 'video_views_for_the_last_30_days'

linear_regression = LinearRegression()
train_df = df.dropna(subset=[target])
test_df = df[df[target].isnull()]

linear_regression.fit(train_df[features], train_df[target])
predictions = linear_regression.predict(test_df[features])

df.loc[df[target].isnull(), target] = predictions

# usuń Andorra i Puerto Rico i Turkey i Latvia i Cuba i Peru
df = df[~df['country'].str.contains('Andorra')]
df = df[~df['country'].str.contains('Puerto Rico')]
df = df[~df['country'].str.contains('Turkey')]
df = df[~df['country'].str.contains('Latvia')]
df = df[~df['country'].str.contains('Cuba')]
df = df[~df['country'].str.contains('Peru')]
# zastąp wartościami ze słownika
# gross_tertiary_education_enrollment, population, unemployment_rate, urban_population
# Mexico - 88.2 - 126014024.0 - 3.42 - 102626859.0
# United States - 88.2 - 328239523.0 - 14.7 - 270663028.0
# Brazil - 88.2 - 212559417.0 - 12.08 - 183241641.0
# India - 94.3 - 1366417754.0 - 5.36 - 471031528.0
# Philippines - 88.2 - 108116615.0 - 2.15 - 50975903.0
# Indonesia - 28.1 - 270203917.0 - 4.69 - 151509724.0
# Pakistan - 28.1 - 216565318.0 - 4.45 - 79927762.0
# Kuwait - 90.0 - 4207083.0 - 2.18 - 4207083.0
# Spain - 28.1 - 47076781.0 - 13.96 - 37927409.0
# United Kingdom - 88.2 - 66834405.0 - 3.85 - 55908316.0
# Thailand - 28.1 - 69625582.0 - 0.75 - 35294600.0
# Colombia - 51.3 - 50339443.0 - 9.71 - 40827302.0
# Sweden - 28.1 - 10285453.0 - 6.48 - 9021165.0
# South Korea - 51.3 - 51709098.0 - 4.15 - 42106719.0
# Saudi Arabia - 49.3 - 34268528.0 - 5.93 - 28807838.0
# Germany - 28.1 - 83132799.0 - 3.04 - 64324835.0
# Argentina - 28.1 - 44938712.0 - 9.79 - 41339571.0
# Australia - 36.3 - 25766605.0 - 5.27 - 21844756.0
# Finland - 51.3 - 5520314.0 - 6.59 - 4716888.0
# Russia - 88.2 - 144373535.0 - 4.59 - 107683889.0
# France - 28.1 - 67059887.0 - 8.43 - 54123364.0
# Samoa - 88.2 - 202506.0 - 8.36 - 35588.0
# Italy - 60.0 - 60297396.0 - 9.89 - 42651966.0
# Chile - 88.2 - 18952038.0 - 7.09 - 16610135.0
# Ecuador - 63.2 - 17373662.0 - 3.97 - 11116711.0
# Vietnam - 28.1 - 96462106.0 - 2.01 - 35332140.0
# United Arab Emirates - 36.8 - 9770529.0 - 2.35 - 8479744.0
# Switzerland - 90.0 - 8574832.0 - 4.58 - 6332428.0
# El Salvador - 88.2 - 6453553.0 - 4.11 - 4694702.0
# Canada - 88.2 - 36991981.0 - 5.56 - 30628482.0
# Bangladesh - 51.3 - 167310838.0 - 4.19 - 60987417.0
# Ukraine - 81.9 - 44385155.0 - 8.88 - 30835699.0
# Malaysia - 88.9 - 32447385.0 - 3.32 - 24475766.0
# Singapore - 28.1 - 5703569.0 - 4.11 - 5703569.0
# Netherlands - 60.0 - 17332850.0 - 3.2 - 15924729.0
# Japan - 55.3 - 126226568.0 - 2.29 - 115782416.0
# Barbados - 35.5 - 287025.0 - 10.33 - 89431.0
# Venezuela - 28.1 - 28515829.0 - 8.8 - 25162368.0
# Jordan - 88.2 - 10101694.0 - 14.72 - 9213048.0

df.loc[df['country'] == 'Mexico', 'gross_tertiary_education_enrollment'] = 88.2
df.loc[df['country'] == 'Mexico', 'population'] = 126014024.0
df.loc[df['country'] == 'Mexico', 'unemployment_rate'] = 3.42
df.loc[df['country'] == 'Mexico', 'urban_population'] = 102626859.0

df.loc[df['country'] == 'United States', 'gross_tertiary_education_enrollment'] = 88.2
df.loc[df['country'] == 'United States', 'population'] = 328239523.0
df.loc[df['country'] == 'United States', 'unemployment_rate'] = 14.7
df.loc[df['country'] == 'United States', 'urban_population'] = 270663028.0

df.loc[df['country'] == 'Brazil', 'gross_tertiary_education_enrollment'] = 88.2
df.loc[df['country'] == 'Brazil', 'population'] = 212559417.0
df.loc[df['country'] == 'Brazil', 'unemployment_rate'] = 12.08
df.loc[df['country'] == 'Brazil', 'urban_population'] = 183241641.0

df.loc[df['country'] == 'India', 'gross_tertiary_education_enrollment'] = 94.3
df.loc[df['country'] == 'India', 'population'] = 1366417754.0
df.loc[df['country'] == 'India', 'unemployment_rate'] = 5.36
df.loc[df['country'] == 'India', 'urban_population'] = 471031528.0

df.loc[df['country'] == 'Philippines', 'gross_tertiary_education_enrollment'] = 88.2
df.loc[df['country'] == 'Philippines', 'population'] = 108116615.0
df.loc[df['country'] == 'Philippines', 'unemployment_rate'] = 2.15
df.loc[df['country'] == 'Philippines', 'urban_population'] = 50975903.0

df.loc[df['country'] == 'Indonesia', 'gross_tertiary_education_enrollment'] = 28.1
df.loc[df['country'] == 'Indonesia', 'population'] = 270203917.0
df.loc[df['country'] == 'Indonesia', 'unemployment_rate'] = 4.69
df.loc[df['country'] == 'Indonesia', 'urban_population'] = 151509724.0

df.loc[df['country'] == 'Pakistan', 'gross_tertiary_education_enrollment'] = 28.1
df.loc[df['country'] == 'Pakistan', 'population'] = 216565318.0
df.loc[df['country'] == 'Pakistan', 'unemployment_rate'] = 4.45
df.loc[df['country'] == 'Pakistan', 'urban_population'] = 79927762.0

df.loc[df['country'] == 'Kuwait', 'gross_tertiary_education_enrollment'] = 90.0
df.loc[df['country'] == 'Kuwait', 'population'] = 4207083.0
df.loc[df['country'] == 'Kuwait', 'unemployment_rate'] = 2.18
df.loc[df['country'] == 'Kuwait', 'urban_population'] = 4207083.0

df.loc[df['country'] == 'Spain', 'gross_tertiary_education_enrollment'] = 28.1
df.loc[df['country'] == 'Spain', 'population'] = 47076781.0
df.loc[df['country'] == 'Spain', 'unemployment_rate'] = 13.96
df.loc[df['country'] == 'Spain', 'urban_population'] = 37927409.0

df.loc[df['country'] == 'United Kingdom', 'gross_tertiary_education_enrollment'] = 88.2
df.loc[df['country'] == 'United Kingdom', 'population'] = 66834405.0
df.loc[df['country'] == 'United Kingdom', 'unemployment_rate'] = 3.85
df.loc[df['country'] == 'United Kingdom', 'urban_population'] = 55908316.0

df.loc[df['country'] == 'Thailand', 'gross_tertiary_education_enrollment'] = 28.1
df.loc[df['country'] == 'Thailand', 'population'] = 69625582.0
df.loc[df['country'] == 'Thailand', 'unemployment_rate'] = 0.75
df.loc[df['country'] == 'Thailand', 'urban_population'] = 35294600.0

df.loc[df['country'] == 'Colombia', 'gross_tertiary_education_enrollment'] = 51.3
df.loc[df['country'] == 'Colombia', 'population'] = 50339443.0
df.loc[df['country'] == 'Colombia', 'unemployment_rate'] = 9.71
df.loc[df['country'] == 'Colombia', 'urban_population'] = 40827302.0

df.loc[df['country'] == 'Sweden', 'gross_tertiary_education_enrollment'] = 28.1
df.loc[df['country'] == 'Sweden', 'population'] = 10285453.0
df.loc[df['country'] == 'Sweden', 'unemployment_rate'] = 6.48
df.loc[df['country'] == 'Sweden', 'urban_population'] = 9021165.0

df.loc[df['country'] == 'South Korea', 'gross_tertiary_education_enrollment'] = 51.3
df.loc[df['country'] == 'South Korea', 'population'] = 51709098.0
df.loc[df['country'] == 'South Korea', 'unemployment_rate'] = 4.15
df.loc[df['country'] == 'South Korea', 'urban_population'] = 42106719.0

df.loc[df['country'] == 'Saudi Arabia', 'gross_tertiary_education_enrollment'] = 49.3
df.loc[df['country'] == 'Saudi Arabia', 'population'] = 34268528.0
df.loc[df['country'] == 'Saudi Arabia', 'unemployment_rate'] = 5.93
df.loc[df['country'] == 'Saudi Arabia', 'urban_population'] = 28807838.0

df.loc[df['country'] == 'Germany', 'gross_tertiary_education_enrollment'] = 28.1
df.loc[df['country'] == 'Germany', 'population'] = 83132799.0
df.loc[df['country'] == 'Germany', 'unemployment_rate'] = 3.04

df.loc[df['country'] == 'Argentina', 'gross_tertiary_education_enrollment'] = 28.1
df.loc[df['country'] == 'Argentina', 'population'] = 44938712.0
df.loc[df['country'] == 'Argentina', 'unemployment_rate'] = 9.79
df.loc[df['country'] == 'Argentina', 'urban_population'] = 41339571.0

df.loc[df['country'] == 'Australia', 'gross_tertiary_education_enrollment'] = 36.3
df.loc[df['country'] == 'Australia', 'population'] = 25766605.0
df.loc[df['country'] == 'Australia', 'unemployment_rate'] = 5.27
df.loc[df['country'] == 'Australia', 'urban_population'] = 21844756.0

df.loc[df['country'] == 'Finland', 'gross_tertiary_education_enrollment'] = 51.3
df.loc[df['country'] == 'Finland', 'population'] = 5520314.0
df.loc[df['country'] == 'Finland', 'unemployment_rate'] = 6.59
df.loc[df['country'] == 'Finland', 'urban_population'] = 4716888.0

df.loc[df['country'] == 'Russia', 'gross_tertiary_education_enrollment'] = 88.2
df.loc[df['country'] == 'Russia', 'population'] = 144373535.0
df.loc[df['country'] == 'Russia', 'unemployment_rate'] = 4.59
df.loc[df['country'] == 'Russia', 'urban_population'] = 107683889.0

df.loc[df['country'] == 'France', 'gross_tertiary_education_enrollment'] = 28.1
df.loc[df['country'] == 'France', 'population'] = 67059887.0
df.loc[df['country'] == 'France', 'unemployment_rate'] = 8.43
df.loc[df['country'] == 'France', 'urban_population'] = 54123364.0

df.loc[df['country'] == 'Samoa', 'gross_tertiary_education_enrollment'] = 88.2
df.loc[df['country'] == 'Samoa', 'population'] = 202506.0
df.loc[df['country'] == 'Samoa', 'unemployment_rate'] = 8.36
df.loc[df['country'] == 'Samoa', 'urban_population'] = 35588.0

df.loc[df['country'] == 'Italy', 'gross_tertiary_education_enrollment'] = 60.0
df.loc[df['country'] == 'Italy', 'population'] = 60297396.0
df.loc[df['country'] == 'Italy', 'unemployment_rate'] = 9.89
df.loc[df['country'] == 'Italy', 'urban_population'] = 42651966.0

df.loc[df['country'] == 'Chile', 'gross_tertiary_education_enrollment'] = 88.2
df.loc[df['country'] == 'Chile', 'population'] = 18952038.0
df.loc[df['country'] == 'Chile', 'unemployment_rate'] = 7.09
df.loc[df['country'] == 'Chile', 'urban_population'] = 16610135.0

df.loc[df['country'] == 'Ecuador', 'gross_tertiary_education_enrollment'] = 63.2
df.loc[df['country'] == 'Ecuador', 'population'] = 17373662.0
df.loc[df['country'] == 'Ecuador', 'unemployment_rate'] = 3.97
df.loc[df['country'] == 'Ecuador', 'urban_population'] = 11116711.0

df.loc[df['country'] == 'Vietnam', 'gross_tertiary_education_enrollment'] = 28.1
df.loc[df['country'] == 'Vietnam', 'population'] = 96462106.0
df.loc[df['country'] == 'Vietnam', 'unemployment_rate'] = 2.01
df.loc[df['country'] == 'Vietnam', 'urban_population'] = 35332140.0

df.loc[df['country'] == 'United Arab Emirates', 'gross_tertiary_education_enrollment'] = 36.8
df.loc[df['country'] == 'United Arab Emirates', 'population'] = 9770529.0
df.loc[df['country'] == 'United Arab Emirates', 'unemployment_rate'] = 2.35
df.loc[df['country'] == 'United Arab Emirates', 'urban_population'] = 8479744.0

df.loc[df['country'] == 'Switzerland', 'gross_tertiary_education_enrollment'] = 90.0
df.loc[df['country'] == 'Switzerland', 'population'] = 8574832.0
df.loc[df['country'] == 'Switzerland', 'unemployment_rate'] = 4.58
df.loc[df['country'] == 'Switzerland', 'urban_population'] = 6332428.0

df.loc[df['country'] == 'El Salvador', 'gross_tertiary_education_enrollment'] = 88.2
df.loc[df['country'] == 'El Salvador', 'population'] = 6453553.0
df.loc[df['country'] == 'El Salvador', 'unemployment_rate'] = 4.11
df.loc[df['country'] == 'El Salvador', 'urban_population'] = 4694702.0

df.loc[df['country'] == 'Canada', 'gross_tertiary_education_enrollment'] = 88.2
df.loc[df['country'] == 'Canada', 'population'] = 36991981.0
df.loc[df['country'] == 'Canada', 'unemployment_rate'] = 5.56
df.loc[df['country'] == 'Canada', 'urban_population'] = 30628482.0

df.loc[df['country'] == 'Bangladesh', 'gross_tertiary_education_enrollment'] = 51.3
df.loc[df['country'] == 'Bangladesh', 'population'] = 167310838.0
df.loc[df['country'] == 'Bangladesh', 'unemployment_rate'] = 4.19
df.loc[df['country'] == 'Bangladesh', 'urban_population'] = 60987417.0

df.loc[df['country'] == 'Ukraine', 'gross_tertiary_education_enrollment'] = 81.9
df.loc[df['country'] == 'Ukraine', 'population'] = 44385155.0
df.loc[df['country'] == 'Ukraine', 'unemployment_rate'] = 8.88
df.loc[df['country'] == 'Ukraine', 'urban_population'] = 30835699.0

df.loc[df['country'] == 'Malaysia', 'gross_tertiary_education_enrollment'] = 88.9
df.loc[df['country'] == 'Malaysia', 'population'] = 32447385.0
df.loc[df['country'] == 'Malaysia', 'unemployment_rate'] = 3.32
df.loc[df['country'] == 'Malaysia', 'urban_population'] = 24475766.0

df.loc[df['country'] == 'Singapore', 'gross_tertiary_education_enrollment'] = 28.1
df.loc[df['country'] == 'Singapore', 'population'] = 5703569.0
df.loc[df['country'] == 'Singapore', 'unemployment_rate'] = 4.11
df.loc[df['country'] == 'Singapore', 'urban_population'] = 5703569.0

df.loc[df['country'] == 'Netherlands', 'gross_tertiary_education_enrollment'] = 60.0
df.loc[df['country'] == 'Netherlands', 'population'] = 17332850.0
df.loc[df['country'] == 'Netherlands', 'unemployment_rate'] = 3.2
df.loc[df['country'] == 'Netherlands', 'urban_population'] = 15924729.0

df.loc[df['country'] == 'Japan', 'gross_tertiary_education_enrollment'] = 55.3
df.loc[df['country'] == 'Japan', 'population'] = 126226568.0
df.loc[df['country'] == 'Japan', 'unemployment_rate'] = 2.29
df.loc[df['country'] == 'Japan', 'urban_population'] = 115782416.0

df.loc[df['country'] == 'Barbados', 'gross_tertiary_education_enrollment'] = 35.5
df.loc[df['country'] == 'Barbados', 'population'] = 287025.0
df.loc[df['country'] == 'Barbados', 'unemployment_rate'] = 10.33
df.loc[df['country'] == 'Barbados', 'urban_population'] = 89431.0

df.loc[df['country'] == 'Venezuela', 'gross_tertiary_education_enrollment'] = 28.1
df.loc[df['country'] == 'Venezuela', 'population'] = 28515829.0
df.loc[df['country'] == 'Venezuela', 'unemployment_rate'] = 8.8
df.loc[df['country'] == 'Venezuela', 'urban_population'] = 25162368.0

df.loc[df['country'] == 'Jordan', 'gross_tertiary_education_enrollment'] = 88.2
df.loc[df['country'] == 'Jordan', 'population'] = 10101694.0
df.loc[df['country'] == 'Jordan', 'unemployment_rate'] = 14.72
df.loc[df['country'] == 'Jordan', 'urban_population'] = 9213048.0

# zapisz plik jako 'youtubeChannelsPreprocessed.csv'
df.drop(['encoded_youtuber', 'encoded_title', 'encoded_subscribers', 'encoded_video_views'], axis=1, inplace=True)
print(df.isnull().sum())
df.to_csv("youtubeChannelsPreprocessed.csv", index=False)
