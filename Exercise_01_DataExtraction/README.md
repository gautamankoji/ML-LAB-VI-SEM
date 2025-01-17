# Exercise 01

## Extracting Data from the Database using Python

### Aim  

To extract and display data from a database table using Python.

### Procedure/Program  

SQLite Example:  

```python
import sqlite3

connection = sqlite3.connect('student_data.db')
cursor = connection.cursor()
cursor.execute("SELECT * FROM students")
rows = cursor.fetchall()

print("ID\tName\t\tAge\tDepartment")
for row in rows:
    print(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}")

connection.close()
```

MySQL Example:  

```python
import mysql.connector

connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="password",
    database="student_data"
)
cursor = connection.cursor()
cursor.execute("SELECT * FROM students")
rows = cursor.fetchall()

print("ID\tName\t\tAge\tDepartment")
for row in rows:
    print(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}")

connection.close()
```

### Output/Explanation  

- Output:  

The program connects to the database, executes a query, and prints the data in the following format:  

```bash
ID    Name        Age    Department
1     Alice       21     CSE
2     Bob         22     ECE
3     Charlie     23     EEE
```

- Explanation:  
  - The script establishes a connection to the database using Python's libraries.  
  - It executes a SQL `SELECT` query to retrieve all rows from the `students` table.  
  - The results are displayed in a tabular format, demonstrating successful data extraction.
