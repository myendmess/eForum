import string
import secrets
import os

class Student:
    def __init__(self, first_name='', second_name='', DOB='03/09/2002'): 
        self.first_name = first_name
        self.second_name = second_name
        self.DOB = DOB
        self.password = self.generate_random_password()

    def __str__(self):
        return f"{self.first_name} {self.second_name} {self.DOB}, password {self.password}"

    @staticmethod
    def generate_random_password():
        chars = string.ascii_letters + string.digits + string.punctuation
        length = 12
        return ''.join(secrets.choice(chars) for _ in range(length))

class GraduateStudent(Student):
    def __init__(self, first_name='', second_name='', DOB='15/05/2001'):
        super().__init__(first_name, second_name, DOB)

def main():
    Leo = GraduateStudent('Leonardo')
    Marco = Student('Marco')

    print(Leo)
    print(Marco)

if __name__ == "__main__":
    main()
