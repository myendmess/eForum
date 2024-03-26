import string
import json
import pickle
import socket


class Contatto:
    def __init__(self, nome, cognome, numero):
        self.nome = nome
        self.cognome = cognome
        self.numero = numero

    def contatto_rubrica(self):
        return f"""
            Contatto Personale
            Nome: {self.nome}
            Cognome: {self.cognome}
            Numero: {self.numero}
            """

PORTS_DATA_FILE = "./common_ports.json"

def extract_json_data(self):
    with open(self, "r") as file:
        data = json.load(file)
    return data

#lista contatti
contatti = [
    Contatto("Sara", "Anelli", "+39 3678940675"),
    Contatto("Antonio", "Corradi", "+39 8970553148"),
    Contatto("Walter", "White", "+1 8970553148"),
]

def stampa_oggetto(oggetto):
    print(f"\n{oggetto.__class__.__name__}")
    for attributo, valore in oggetto.__dict__.items():
        print(f" - {attributo}: {valore}")

for contatto in contatti:
    stampa_oggetto(contatto)
