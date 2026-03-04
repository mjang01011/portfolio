- ### Basic SQL

  #### What is a Database?

  A **database** is a structured collection of data stored in a computer system, designed to make data easily accessible and manageable. 

  #### Relational Databases

  A **relational database** is a specific type of database that organizes data into tables, where relationships between different pieces of data can be established. Each table consists of rows and columns, with each row representing a unique record and each column representing a specific attribute of that record.

  For example, imagine a table in a relational database that stores information about books. Each row might represent a single book, and columns could include attributes like the title, author, and publication year. The power of relational databases comes from their ability to link tables together based on shared data, making it easy to cross-reference and access related information.

  

  #### Introduction to SQL

  **SQL** (Structured Query Language) is the standard programming language used to interact with relational databases. Below are some of the most commonly used SQL commands:

  - **`CREATE TABLE`**: Creates a new table in the database.
  - **`INSERT INTO`**: Adds a new row to a table.
  - **`SELECT`**: Queries and retrieves data from a table.
  - **`ALTER TABLE`**: Modifies an existing table, such as adding or removing columns.
  - **`UPDATE`**: Edits an existing record in a table.
  - **`DELETE FROM`**: Removes records from a table.

  ##### Example: Creating a Table

  ```
  CREATE TABLE books (
    id INTEGER PRIMARY KEY,
    title TEXT UNIQUE,
    author TEXT NOT NULL,
    published_year INTEGER,
    genre TEXT
  );
  ```

  In this example, we're creating a `books` table with columns for `id`, `title`, `author`, `published_year`, and `genre`. Each column is defined with a specific data type, and constraints like `PRIMARY KEY`, `UNIQUE`, and `NOT NULL` help ensure data integrity.

  #### Indexes in Databases

  An **index** is like a pointer to data in a table, allowing for faster retrieval of specific information without having to scan the entire table. Indexes are particularly useful for improving the performance of database queries.

  Imagine you have a large table storing millions of records about books. If you want to quickly find all books written by a specific author, an index on the `author` column would make this query much faster.

  ##### Example: Creating an Index

  ```
  CREATE INDEX idx_author
  ON books (author);
  ```

  Here, we're creating an index named `idx_author` on the `author` column of the `books` table.

  #### Querying Data with SQL

  When working with databases, retrieving data is a common task. SQL provides powerful commands to filter, sort, and manipulate query results:

  - **`SELECT`**: The basic command for querying data.
  - **`AS`**: Renames a column or table in the result set.
  - **`DISTINCT`**: Returns unique values, removing duplicates.
  - **`WHERE`**: Filters the results based on specific conditions.
  - **`LIKE`** and **`BETWEEN`**: Special operators for pattern matching and range queries.
  - **`AND`** and **`OR`**: Combine multiple conditions.
  - **`ORDER BY`**: Sorts the result set by one or more columns.
  - **`LIMIT`**: Restricts the number of rows returned by the query.
  - **`CASE`**: Creates conditional logic within queries.

  ##### Example: Selecting Data with a Filter

  ```
  SELECT title, author
  FROM books
  WHERE genre = 'Science Fiction'
  ORDER BY published_year DESC;
  ```

  This query retrieves the titles and authors of all books in the `Science Fiction` genre, sorted by the publication year in descending order.

  #### Aggregate Functions

  **Aggregate functions** in SQL allow you to perform calculations on multiple rows and return a single value that represents a summary of the data. Some common aggregate functions include:

  - **`COUNT()`**: Counts the number of rows in a query.
  - **`SUM()`**: Calculates the sum of values in a column.
  - **`MAX()`**/**`MIN()`**: Finds the maximum or minimum value in a column.
  - **`AVG()`**: Calculates the average value of a column.
  - **`ROUND()`**: Rounds numeric values to a specified number of decimal places.

  ##### Example: Grouping Data with Aggregate Functions

  ```
  SELECT genre, COUNT(*) AS num_books
  FROM books
  GROUP BY genre
  HAVING COUNT(*) > 10;
  ```

  In this query, we group books by genre and count the number of books in each genre. The `HAVING` clause filters the results to include only genres with more than 10 books.