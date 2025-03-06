import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import statsmodels.api as sm

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")


def generate_sales_data(n_records=1000):
    """Generate synthetic sales data with seasonal patterns."""
    end_date = pd.Timestamp('2024-12-31')
    start_date = pd.Timestamp('2022-01-01')
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    weights = np.linspace(0.5, 1.0, len(dates))
    sampled_dates = np.random.choice(
        dates, size=n_records, p=weights/weights.sum())

    categories = ['Electronics', 'Clothing', 'Home Goods', 'Books', 'Toys']
    product_ids = [f'P{i:03d}' for i in range(1, 51)]

    data = {
        'date': sampled_dates,
        'product_id': np.random.choice(product_ids, size=n_records),
        'category': np.random.choice(categories, size=n_records),
        'quantity': np.random.randint(1, 10, size=n_records),
        'unit_price': np.random.uniform(10, 500, size=n_records).round(2),
        'customer_id': np.random.randint(1000, 5000, size=n_records),
        'store_id': np.random.choice(['S001', 'S002', 'S003', 'S004'], size=n_records)
    }

    # Add seasonal effects (holiday season and summer)
    month = pd.DatetimeIndex(sampled_dates).month
    holiday_boost = np.where(np.isin(month, [11, 12]), 1.5, 1.0)
    summer_boost = np.where(np.isin(month, [6, 7, 8]), 1.2, 1.0)

    data['unit_price'] = data['unit_price'] * holiday_boost * summer_boost
    data['total_sales'] = data['quantity'] * data['unit_price']

    df = pd.DataFrame(data)

    # Add missing values to demonstrate cleaning
    mask = np.random.random(size=n_records) < 0.05
    df.loc[mask, 'unit_price'] = np.nan

    return df


def clean_data(df):
    """Clean the dataset by handling missing values and adding time features."""
    # Fill missing unit prices with median price for that product
    df['unit_price'] = df.groupby('product_id')['unit_price'].transform(
        lambda x: x.fillna(x.median()))

    # Recalculate total sales after filling missing values
    df['total_sales'] = df['quantity'] * df['unit_price']

    # Add time-based features for analysis
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter

    return df


def analyze_monthly_trends(df):
    """Analyze and visualize monthly sales trends."""
    monthly_sales = df.groupby(['year', 'month'])[
        'total_sales'].sum().reset_index()
    monthly_sales['year_month'] = monthly_sales['year'].astype(
        str) + '-' + monthly_sales['month'].astype(str).str.zfill(2)

    plt.figure(figsize=(12, 6))
    plt.plot(monthly_sales['year_month'],
             monthly_sales['total_sales'], marker='o')
    plt.title('Monthly Sales Trend')
    plt.xlabel('Year-Month')
    plt.ylabel('Total Sales ($)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('monthly_sales_trend.png')

    return monthly_sales


def analyze_category_performance(df):
    """Analyze sales performance by product category."""
    category_sales = df.groupby(
        'category')['total_sales'].sum().sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=category_sales.index, y=category_sales.values)
    plt.title('Total Sales by Product Category')
    plt.xlabel('Category')
    plt.ylabel('Total Sales ($)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('category_sales.png')

    return category_sales


def analyze_top_products(df):
    """Identify and visualize top-selling products."""
    top_products = df.groupby('product_id')['total_sales'].sum(
    ).sort_values(ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_products.index, y=top_products.values)
    plt.title('Top 10 Products by Sales')
    plt.xlabel('Product ID')
    plt.ylabel('Total Sales ($)')
    plt.tight_layout()
    plt.savefig('top_products.png')

    return top_products


def analyze_weekly_patterns(df):
    """Analyze sales patterns by day of week."""
    day_names = ['Monday', 'Tuesday', 'Wednesday',
                 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_sales = df.groupby('day_of_week')['total_sales'].sum()
    day_sales.index = [day_names[i] for i in day_sales.index]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=day_sales.index, y=day_sales.values)
    plt.title('Sales by Day of Week')
    plt.xlabel('Day')
    plt.ylabel('Total Sales ($)')
    plt.tight_layout()
    plt.savefig('day_of_week_sales.png')

    return day_sales


def analyze_seasonal_trends(df):
    """Analyze quarterly sales patterns to identify seasonality."""
    quarterly_sales = df.groupby(['year', 'quarter'])[
        'total_sales'].sum().reset_index()
    quarterly_sales['year_quarter'] = quarterly_sales['year'].astype(
        str) + '-Q' + quarterly_sales['quarter'].astype(str)

    plt.figure(figsize=(12, 6))
    plt.plot(quarterly_sales['year_quarter'],
             quarterly_sales['total_sales'], marker='o')
    plt.title('Quarterly Sales Trend')
    plt.xlabel('Year-Quarter')
    plt.ylabel('Total Sales ($)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('quarterly_sales.png')

    return quarterly_sales


def analyze_store_performance(df):
    """Compare performance across different stores."""
    store_sales = df.groupby('store_id')[
        'total_sales'].sum().sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=store_sales.index, y=store_sales.values)
    plt.title('Sales by Store')
    plt.xlabel('Store ID')
    plt.ylabel('Total Sales ($)')
    plt.tight_layout()
    plt.savefig('store_sales.png')

    return store_sales


def perform_time_series_analysis(df):
    """Decompose time series to identify trend and seasonality components."""
    daily_sales = df.groupby('date')['total_sales'].sum()
    daily_sales = daily_sales.asfreq('D', fill_value=0)

    decomposition = sm.tsa.seasonal_decompose(
        daily_sales, model='additive', period=30)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
    decomposition.observed.plot(ax=ax1)
    ax1.set_title('Observed')
    decomposition.trend.plot(ax=ax2)
    ax2.set_title('Trend')
    decomposition.seasonal.plot(ax=ax3)
    ax3.set_title('Seasonality')
    decomposition.resid.plot(ax=ax4)
    ax4.set_title('Residuals')
    plt.tight_layout()
    plt.savefig('time_series_decomposition.png')

    return decomposition


def generate_insights(category_sales, top_products, store_sales, day_sales, monthly_sales, quarterly_sales):
    """Generate key business insights from the analysis."""
    insights = {
        "top_category": {"name": category_sales.index[0], "sales": category_sales.values[0]},
        "top_product": {"id": top_products.index[0], "sales": top_products.values[0]},
        "best_store": {"id": store_sales.index[0], "sales": store_sales.values[0]},
        "best_day": {"name": day_sales.idxmax(), "sales": day_sales.max()}
    }

    # Calculate monthly growth rate
    monthly_growth = monthly_sales['total_sales'].pct_change() * 100
    insights["avg_monthly_growth"] = monthly_growth.mean()

    # Analyze seasonal patterns
    q4_sales = quarterly_sales[quarterly_sales['quarter']
                               == 4]['total_sales'].mean()
    other_quarters = quarterly_sales[quarterly_sales['quarter']
                                     != 4]['total_sales'].mean()
    insights["q4_increase"] = ((q4_sales / other_quarters) - 1) * 100

    return insights


def main():
    """Main function to execute the sales analysis pipeline."""
    np.random.seed(42)

    # Generate dataset
    sales_df = generate_sales_data(5000)

    # Clean and preprocess data
    sales_df = clean_data(sales_df)

    # Perform analyses
    monthly_sales = analyze_monthly_trends(sales_df)
    category_sales = analyze_category_performance(sales_df)
    top_products = analyze_top_products(sales_df)
    day_sales = analyze_weekly_patterns(sales_df)
    quarterly_sales = analyze_seasonal_trends(sales_df)
    store_sales = analyze_store_performance(sales_df)

    # Time series analysis
    decomposition = perform_time_series_analysis(sales_df)

    # Generate insights
    insights = generate_insights(
        category_sales, top_products, store_sales,
        day_sales, monthly_sales, quarterly_sales
    )

    # Print key findings
    print("\n--- Key Findings ---")
    print(
        f"1. Top selling category: {insights['top_category']['name']} (${insights['top_category']['sales']:.2f})")
    print(
        f"2. Top selling product: {insights['top_product']['id']} (${insights['top_product']['sales']:.2f})")
    print(
        f"3. Best performing store: {insights['best_store']['id']} (${insights['best_store']['sales']:.2f})")
    print(
        f"4. Best sales day: {insights['best_day']['name']} (${insights['best_day']['sales']:.2f})")
    print(
        f"5. Average monthly growth rate: {insights['avg_monthly_growth']:.2f}%")
    print(
        f"6. Q4 sales are {insights['q4_increase']:.2f}% higher than other quarters on average")

    # Save cleaned dataset
    sales_df.to_csv('cleaned_sales_data.csv', index=False)


if __name__ == "__main__":
    main()
