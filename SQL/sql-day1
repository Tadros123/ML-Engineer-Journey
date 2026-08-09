```sql
-- ============================================================
-- DAY 01 — SQL FUNDAMENTALS
-- Data Science / ML Engineer Journey
-- ============================================================

-- Assumed table:
-- Employees
-- Columns:
-- name
-- department
-- salary


-- ============================================================
-- 1. EMPLOYEES ABOVE THEIR DEPARTMENT AVERAGE
-- ============================================================
-- Find employees whose salary is higher than the
-- average salary of their own department.

SELECT e.name, e.department, e.salary
FROM Employees e
WHERE e.salary > (
    SELECT AVG(salary)
    FROM Employees
    WHERE department = e.department
);


-- ============================================================
-- 2. DEPARTMENTS WITH AT LEAST 2 EMPLOYEES
-- ============================================================
-- GROUP BY creates one group for each department.
-- HAVING filters the grouped results.

SELECT department, COUNT(*) AS employee_count
FROM Employees
GROUP BY department
HAVING COUNT(*) >= 2;


-- ============================================================
-- 3. SALARY CLASSIFICATION USING CASE
-- ============================================================
-- Classify employees based on salary.

SELECT
    name,
    department,
    salary,
    CASE
        WHEN salary >= 900 THEN 'High salary'
        ELSE 'Low salary'
    END AS salary_level
FROM Employees
WHERE salary > 899;


-- ============================================================
-- 4. DEPARTMENT AVERAGE USING A WINDOW FUNCTION
-- ============================================================
-- Calculate the average salary for each department
-- while keeping every employee row.

SELECT
    name,
    department,
    salary,
    AVG(salary) OVER (
        PARTITION BY department
    ) AS department_avg
FROM Employees;


-- ============================================================
-- 5. HIGHEST-PAID EMPLOYEE IN EACH DEPARTMENT
-- ============================================================
-- ROW_NUMBER() ranks employees within each department.
-- rn = 1 represents the highest-paid employee.

SELECT
    department,
    name,
    salary
FROM (
    SELECT
        department,
        name,
        salary,
        ROW_NUMBER() OVER (
            PARTITION BY department
            ORDER BY salary DESC
        ) AS rn
    FROM Employees
) AS ranked
WHERE rn = 1;


-- ============================================================
-- 6. BASIC AGGREGATION — AVERAGE SALARY
-- ============================================================
-- Calculate the average salary across all employees.

SELECT AVG(salary) AS average_salary
FROM Employees;


-- ============================================================
-- 7. AVERAGE SALARY BY DEPARTMENT
-- ============================================================
-- Calculate one average salary for each department.

SELECT
    department,
    AVG(salary) AS average_salary
FROM Employees
GROUP BY department;


-- ============================================================
-- 8. BASIC FILTERING
-- ============================================================
-- Find employees earning more than 800.

SELECT
    name,
    department,
    salary
FROM Employees
WHERE salary > 800;


-- ============================================================
-- DAY 01 SQL CONCEPTS PRACTICED
-- ============================================================
-- SELECT
-- WHERE
-- AVG()
-- COUNT()
-- GROUP BY
-- HAVING
-- CASE
-- Correlated subqueries
-- Window functions
-- PARTITION BY
-- ROW_NUMBER()
-- ORDER BY
-- Filtering ranked results
-- ============================================================
```
