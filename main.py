import pandas as pd

df = pd.read_excel('IRminiProj#2_dataset.xlsx')

# for column,value in df.items():
#     if (column == 'Abstract'):
#         print(value)

for ind in df.index:
    print(df['Abstract'][ind])