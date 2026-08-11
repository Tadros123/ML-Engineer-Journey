-- Day 3: SQL JOINs and Multi-Table Analysis

CREATE TABLE Customers (
    customer_id INT PRIMARY KEY,
    name VARCHAR(50),
    city VARCHAR(50)
);

INSERT INTO Customers (customer_id, name, city) VALUES
(1, 'Ahmad', 'Amman'),
(2, 'Sara', 'Zarqa'),
(3, 'Omar', 'Amman'),
(4, 'Lina', 'Irbid'),
(5, 'Dana', 'Amman');


CREATE TABLE Orders (
    order_id INT PRIMARY KEY,
    customer_id INT,
    amount DECIMAL(10,2),
    FOREIGN KEY (customer_id) REFERENCES Customers(customer_id)
);

INSERT INTO Orders (order_id, customer_id, amount) VALUES
(101, 1, 120),
(102, 2, 300),
(103, 1, 200),
(104, 3, 150),
(105, 2, 100),
(106, 3, 400);

-- Task 1: Customers who placed orders

SELECT Customers.name AS customer_name,
       Orders.order_id,
       Orders.amount
FROM Customers
JOIN Orders
ON Customers.customer_id = Orders.customer_id;


-- Task 2: All customers, including customers with no orders

SELECT Customers.name AS customer_name,
       Orders.order_id,
       Orders.amount
FROM Customers
LEFT JOIN Orders
ON Customers.customer_id = Orders.customer_id;


-- Task 3: Customers who never placed an order

SELECT Customers.name AS customer_name
FROM Customers
LEFT JOIN Orders
ON Customers.customer_id = Orders.customer_id
WHERE Orders.customer_id IS NULL;


-- Task 4: Total spending by customer

SELECT Customers.name AS customer_name,
       SUM(Orders.amount) AS total_spent
FROM Customers
JOIN Orders
ON Customers.customer_id = Orders.customer_id
GROUP BY Customers.customer_id, Customers.name;

-- Task 5: Customers whose total spending is greater than 350

SELECT Customers.name AS customer_name,
       SUM(Orders.amount) AS total_spent
FROM Customers
JOIN Orders
ON Customers.customer_id = Orders.customer_id
GROUP BY Customers.customer_id, Customers.name
HAVING SUM(Orders.amount) > 350;

-- Task 6: All customers with total spending, including customers with no orders

SELECT Customers.name AS customer_name,
       COALESCE(SUM(Orders.amount), 0) AS total_spent
FROM Customers
LEFT JOIN Orders
ON Customers.customer_id = Orders.customer_id
GROUP BY Customers.customer_id, Customers.name;


-- Task 7: Number of orders and total spending for every customer

SELECT Customers.name AS customer_name,
       COUNT(Orders.order_id) AS number_of_orders,
       COALESCE(SUM(Orders.amount), 0) AS total_spent
FROM Customers
LEFT JOIN Orders
ON Customers.customer_id = Orders.customer_id
GROUP BY Customers.customer_id, Customers.name;


-- Task 8: Number of orders, total spending, and average order value

SELECT Customers.name AS customer_name,
       COUNT(Orders.order_id) AS number_of_orders,
       SUM(Orders.amount) AS total_spent,
       AVG(Orders.amount) AS average_order_value
FROM Customers
JOIN Orders
ON Customers.customer_id = Orders.customer_id
GROUP BY Customers.customer_id, Customers.name;


-- Task 9: Highest-spending customer overall

SELECT Customers.name AS customer_name,
       SUM(Orders.amount) AS total_spent
FROM Customers
JOIN Orders
ON Customers.customer_id = Orders.customer_id
GROUP BY Customers.customer_id, Customers.name
ORDER BY total_spent DESC
LIMIT 1;


-- Task 10: Highest-spending customer within each city

SELECT city,
       customer_name,
       total_spent
FROM (
    SELECT city,
           customer_name,
           total_spent,
           ROW_NUMBER() OVER (
               PARTITION BY city
               ORDER BY total_spent DESC
           ) AS rn
    FROM (
        SELECT Customers.city,
               Customers.name AS customer_name,
               SUM(Orders.amount) AS total_spent
        FROM Customers
        JOIN Orders
        ON Customers.customer_id = Orders.customer_id
        GROUP BY Customers.customer_id,
                 Customers.name,
                 Customers.city
    ) AS customer_totals
) AS ranked
WHERE rn = 1;


-- Task 11: Rank all customers by spending within each city

SELECT city,
       customer_name,
       total_spent,
       ROW_NUMBER() OVER (
           PARTITION BY city
           ORDER BY total_spent DESC
       ) AS city_rank
FROM (
    SELECT Customers.city,
           Customers.name AS customer_name,
           SUM(Orders.amount) AS total_spent
    FROM Customers
    JOIN Orders
    ON Customers.customer_id = Orders.customer_id
    GROUP BY Customers.customer_id,
             Customers.name,
             Customers.city
) AS customer_totals;
