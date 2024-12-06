import requests

# API key and endpoint
api_key = "AIzaSyCMI2ZmiBgZGxc2cPLsvBvR5mGgvGTry3o"
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"

prompt = """
take the below code and give answer in strictly yes or no, wheather the code has bug or not\n

class Employee:
    def __init__(self, name, salary):
        # Incorrectly storing mutable default argument (Bug 1)
        self.name = name
        self.salary = salary
        self.skills = []  # This is fine, but it could have been passed as an argument with a bug

    def add_skill(self, skill):
        # Doesn't check if skill is already added (Bug 2)
        self.skills.append(skill)

    def display(self):
        # Accessing non-existent attribute (Bug 3)
        print(f"Employee Name: {self.name}, Age: {self.age}")

def calculate_bonus(employees):
    # May cause ZeroDivisionError if no employees exist (Bug 4)
    total_salary = sum(emp.salary for emp in employees)
    average_salary = total_salary / len(employees)

    # Gives a bonus to all employees, but could apply incorrect logic (Bug 5)
    for emp in employees:
        if emp.salary < average_salary:
            emp.salary += 1000
        else:
            emp.salary -= 1000  # Risky behavior (Bug 6)
    return employees

# Main code
if __name__ == "__main__":
    emp1 = Employee("John", 5000)
    emp2 = Employee("Doe", 6000)

    emp1.add_skill("Python")
    emp2.add_skill("Java")

    # Doesn't handle exceptions (Bug 7)
    try:
        emp1.display()
    except AttributeError as e:
        print(f"Error in displaying employee: {e}")

    employees = [emp1, emp2]
    updated_employees = calculate_bonus(employees)

    # Prints details but may fail due to bugs above
    for emp in updated_employees:
        print(f"Name: {emp.name}, Salary: {emp.salary}, Skills: {emp.skills}")

"""

# Headers and payload
headers = {
    "Content-Type": "application/json"
}
payload = {
    "contents": [
        {
            "parts": [
                {"text": prompt}
            ]
        }
    ]
}

# Make the POST request
response = requests.post(url, headers=headers, json=payload)

# Print the response
if response.status_code == 200:
    print("Response:")
    print(response.json()["candidates"][0]["content"]["parts"][0]["text"])
else:
    print(f"Error: {response.status_code}")
    print(response.text)
