-- Day 6 - Advanced SQL practice

-- monthly sales and profit
WITH calculations AS (
    SELECT
        DATE_FORMAT(order_date, '%Y-%m') AS Months,
        SUM(net_sales) AS Monthly_net_sales,
        SUM(profit) AS Monthly_profit
    FROM orders_table
    GROUP BY Months
)
SELECT
    Months,
    Monthly_net_sales,
    Monthly_profit,
    100.0 * Monthly_profit / NULLIF(Monthly_net_sales, 0) AS Profit_margin,
    LAG(Monthly_profit) OVER (ORDER BY Months) AS Previous_month_profit,
    Monthly_profit - LAG(Monthly_profit) OVER (ORDER BY Months) AS Profit_change,
    100.0 * (
        Monthly_profit - LAG(Monthly_profit) OVER (ORDER BY Months)
    ) / NULLIF(
        LAG(Monthly_profit) OVER (ORDER BY Months), 0
    ) AS Profit_change_pct,
    SUM(Monthly_net_sales) OVER (
        ORDER BY Months
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS Running_net_sales,
    AVG(Monthly_profit) OVER (
        ORDER BY Months
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS Three_month_profit_avg
FROM calculations
ORDER BY Months;


-- return rate for every channel
SELECT
    channel AS Channel,
    COUNT(*) AS Total_orders,
    SUM(CASE WHEN returned = 'Yes' THEN 1 ELSE 0 END) AS Returned_orders,
    100.0 * SUM(CASE WHEN returned = 'Yes' THEN 1 ELSE 0 END)
        / COUNT(*) AS Return_rate
FROM orders_table
GROUP BY channel;


-- returned vs non-returned profit
SELECT
    channel AS Channel,
    SUM(profit) AS Total_profit,
    SUM(CASE WHEN returned = 'Yes' THEN profit ELSE 0 END) AS Returned_profit,
    SUM(CASE WHEN returned = 'No' THEN profit ELSE 0 END) AS Nonreturned_profit
FROM orders_table
GROUP BY channel;


-- rank channels by total profit
WITH ranking_channel AS (
    SELECT
        channel AS Channel,
        SUM(profit) AS Total_profit_by_channel
    FROM orders_table
    GROUP BY channel
)
SELECT
    Channel,
    Total_profit_by_channel,
    DENSE_RANK() OVER (
        ORDER BY Total_profit_by_channel DESC
    ) AS ranking
FROM ranking_channel;


-- rank products inside each region
WITH product_profit AS (
    SELECT
        region,
        product,
        SUM(profit) AS total_profit
    FROM orders_table
    GROUP BY region, product
)
SELECT
    region,
    product,
    total_profit,
    DENSE_RANK() OVER (
        PARTITION BY region
        ORDER BY total_profit DESC
    ) AS product_rank
FROM product_profit;


-- top product in each region
WITH calculations AS (
    SELECT
        region,
        product,
        SUM(profit) AS total_profit
    FROM orders_table
    GROUP BY region, product
),
ranking AS (
    SELECT
        region,
        product,
        total_profit,
        DENSE_RANK() OVER (
            PARTITION BY region
            ORDER BY total_profit DESC
        ) AS The_Ranking
    FROM calculations
)
SELECT
    region,
    product,
    total_profit,
    The_Ranking
FROM ranking
WHERE The_Ranking = 1;
