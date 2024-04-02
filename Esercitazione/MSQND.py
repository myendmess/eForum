import numpy as np

# Esempio di array di dati
data = np.array([22, 55, 77, 99, 12, 5, 98])

# Calcolo di media, deviazione standard e numero di dati
media = np.mean(data)
deviazione_standard = np.std(data)
numero_dati = data.size

# Stampa dei risultati
print(f"Media: {media}")
print(f"Deviazione standard: {deviazione_standard}")
print(f"Numero di dati: {numero_dati}")

# Salvataggio dei risultati in un array
risultati = np.array([media, deviazione_standard, numero_dati])

# Salvataggio dell'array su file
np.save("risultati_statistiche.npy", risultati)

print("Statistiche salvate correttamente nel file 'risultati_statistiche.npy'")
