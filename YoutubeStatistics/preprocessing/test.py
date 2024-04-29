import pandas as pd
df = pd.read_csv("youtubeChannelsPreprocessed.csv", encoding='latin-1')
for index, row in df.iterrows():
    print(f"{row['country']} - {row['gross_tertiary_education_enrollment']} - {row['population']} - {row['unemployment_rate']} - {row['urban_population']}")

# Utwórz pusty słownik
country_dict = {}

# # Iteruj przez DataFrame
# for index, row in df.iterrows():
#     # Jeśli kraj nie jest już w słowniku, dodaj go
#     if row['country'] not in country_dict:
#         if not pd.isna(row['gross_tertiary_education_enrollment']):
#             country_dict[row['country']] = [row['gross_tertiary_education_enrollment'], row['population'], row['unemployment_rate'], row['urban_population']]
#
# # Wydrukuj słownik
# for country, values in country_dict.items():
#     print(f"{country} - {values[0]} - {values[1]} - {values[2]} - {values[3]}")
# # import pandas as pd
#
# # Wczytaj pliki CSV
# df_global = pd.read_csv("Global YouTube Statistics.csv", encoding='latin-1')
# df_correct = pd.read_csv("poprawny.csv", encoding='latin-1')
#
# # Upewnij się, że oba DataFrame'y są posortowane w taki sam sposób
# df_global = df_global.sort_values('Youtuber')
# df_correct = df_correct.sort_values('Youtuber')
#
# # Zastąp kolumnę "unemployment_rate"
# df_global['Gross tertiary education enrollment (%)'] = df_correct['Gross tertiary education enrollment (%)']
#
# # Zapisz zmieniony DataFrame z powrotem do pliku CSV
# df_global.to_csv("Global YouTube Statistics.csv", index=False, encoding='latin-1')