class BankAccount:
    def __init__(self, balace):
        self.balance = balace

    def deposit(self, amount):
        self.balance += amount
        return self.balance


accouint = BankAccount(1000)
print(accouint.balance)
accouint.deposit(500)
print(accouint.balance)
