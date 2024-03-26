import string
import json
import pickle
import socket


import json

#creazione oggetto
class Contatto:
    def __init__(self, nome, cognome, numero):
        self.nome = nome
        self.cognome = cognome
        self.numero = numero

    def to_dict(self):
        return {
            'nome': self.nome,
            'cognome': self.cognome,
            'numero': self.numero
        }
    

# lista contatti
contatti = [
    Contatto("Sara", "Anelli", "+39 3678940675"),
    Contatto("Antonio", "Corradi", "+39 8970553148"),
    Contatto("Walter", "White", "+1 8970553148"),
]

# Convertire la lista di oggetti Contatto in una lista di dizionari
contatti_dict = [contatto.to_dict() for contatto in contatti]

# Codifica della lista di dizionari in formato JSON
contatti_json = json.dumps(contatti_dict, indent=3)
print(contatti_json)
# Codifica contatti in binario con pickle
with open('contatti.pickle', 'wb') as file:
    pickle.dump(contatti_dict, file)

print("Contacts have been encrypted to binary format and saved as 'contatti.pickle'.")