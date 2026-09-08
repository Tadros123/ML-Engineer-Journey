# Sales & Profitability Dashboard

Power BI project built to analyze sales performance, profitability, discount behavior, and returns from a transactional business dataset.

## Project goal

The main business question was:

> Sales look okay, but profit seems weaker than expected. What is happening, and what should management pay attention to?

The report focuses on Net Sales, Profit, Profit Margin, discounts, returns, channel performance, and monthly profitability.

## Report pages

### Sales Overview

![Sales Overview](images/sales_overview.png)

Main view for overall business performance and profitability trends.

### Return Analysis

![Return Analysis](images/return_analysis.png)

Focused view for return frequency and the financial impact of returned orders.

### Channel Drilldown

Interactive page for looking at channel-level Net Sales and Profit Margin using a channel slicer.

## Data model

![Data Model](images/data_model.png)

The model uses a simple star-schema pattern:

- `day04_business_sales` — fact table containing transaction-level sales data
- `DateTable` — date dimension related to the sales table with a one-to-many relationship

`Month Name` is sorted by `Month Number`, and the Date table is used for monthly reporting.

## Main KPIs

| Metric | Result |
|---|---:|
| Total Net Sales | 132.07K |
| Total Profit | 46.58K |
| Profit Margin | 35.27% |
| Total Orders | 650 |
| Average Order Value | 203.19 |
| Returned Orders | 31 |
| Return Rate | 4.77% |
| Returned Profit | -3.24K |
| Average Profit per Returned Order | -104.37 |

## Key findings

- Online produced the highest Net Sales, but had the lowest channel Profit Margin at about 32.9%.
- Wholesale had the highest channel Profit Margin at about 39.3%.
- Higher discount levels were strongly associated with lower Profit Margin in this dataset, from about 44.8% at 0% discount to about 12.3% at 30% discount.
- Returned orders represented 4.77% of orders but generated about -3.24K in Profit, averaging roughly -104 per returned order.
- Online had the highest observed channel Return Rate at about 6.13%, followed by Retail at about 5.38%; no Wholesale returns were observed in this dataset.
- Monthly Profit Margin weakened from January through April, recovered partially in May, and fell again in June.

These are descriptive findings from this dataset. They show associations and observed patterns, not proof of causation.

## Power Query work

The dataset was prepared in Power Query before visualization:

- removed exact duplicate rows
- handled missing categorical values with `Unknown`
- corrected data types
- kept valid negative-profit observations instead of treating them automatically as bad data

## DAX measures

```DAX
Total Net Sales =
SUM(day04_business_sales[net_sales])
```

```DAX
Total Profit =
SUM(day04_business_sales[profit])
```

```DAX
Profit Margin =
DIVIDE(
    [Total Profit],
    [Total Net Sales]
)
```

```DAX
Total Orders =
DISTINCTCOUNT(day04_business_sales[order_id])
```

```DAX
Average Order Value =
DIVIDE(
    [Total Net Sales],
    [Total Orders]
)
```

```DAX
Returned Orders =
COALESCE(
    CALCULATE(
        [Total Orders],
        day04_business_sales[returned] = "Yes"
    ),
    0
)
```

```DAX
Return Rate =
DIVIDE(
    [Returned Orders],
    [Total Orders]
)
```

```DAX
Returned Profit =
CALCULATE(
    [Total Profit],
    day04_business_sales[returned] = "Yes"
)
```

```DAX
Average Profit per Returned Order =
DIVIDE(
    [Returned Profit],
    [Returned Orders]
)
```

```DAX
Total Orders All Channels =
CALCULATE(
    [Total Orders],
    REMOVEFILTERS(day04_business_sales[channel])
)
```

```DAX
Order Share % =
DIVIDE(
    [Total Orders],
    [Total Orders All Channels]
)
```

## Power BI features used

- Power Query cleaning and transformation
- DAX measures
- filter context
- `CALCULATE`, `DIVIDE`, `DISTINCTCOUNT`, `REMOVEFILTERS`, and `COALESCE`
- slicers and cross-filtering
- custom tooltips
- visual interactions
- page navigation
- date dimension and one-to-many relationship
- chronological month sorting

## Files

- `day05_sales_profitability_dashboard.pbix` — Power BI report
- `images/sales_overview.png` — Sales Overview preview
- `images/return_analysis.png` — Return Analysis preview
- `images/data_model.png` — model preview

The source CSV is kept separately in the repository under `Data/day04_business_sales.csv`.

## Open the report

Open `day05_sales_profitability_dashboard.pbix` with Power BI Desktop.
