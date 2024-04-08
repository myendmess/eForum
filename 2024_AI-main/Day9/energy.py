import pandas as pd

# Read CSV file into DataFrame
df = pd.read_csv("DCCV.csv")

# Display first two rows of DataFrame
print(df.head(2))

# Drop specified columns to create a new DataFrame
df1 = df.drop(columns=["ITTER107", "Territorio", "TIPO_DATO4", "Tipo Dato", "Disponibilità","Flag codes", "Flags"], axis=1)

# Print information about the new DataFrame
print(df1.info())

# Calcolo della disponibilità
disponibilita = df1[df1["SETTORE_USO"] == "totale"]["Value"].sum()

# Calcolo del consumo interno lordo
consumo_interno_lordo = df1[df1["SETTORE_USO"] == "consumo interno lordo"]["Value"].sum()

print("Disponibilità totale:", disponibilita)
print("Consumo interno lordo totale:", consumo_interno_lordo)
