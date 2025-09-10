import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

'''TASK 1: Data Quality and Preparation
● Validate and clean the datasets.
● Check for missing values, duplicate entries, and inconsistent data.
● Standardize date formats and remove irrelevant columns.
● Handle outliers in numeric columns like Loan_Amount, Interest_Rate, and Default_Amount.'''

# loading the data
applications_df =  pd.read_csv('Downloads\\applications.csv',low_memory=False)
branches_df =  pd.read_csv('Downloads\\branches.csv')
customers_df =  pd.read_csv('Downloads\\customers.csv')
defaults_df =  pd.read_csv('Downloads\\defaults.csv')
loans_df =  pd.read_csv('Downloads\\loans.csv')
transactions_df =  pd.read_csv('Downloads\\transactions.csv')

'''
print("Applications DF")
applications_df.info() ## there is null in loan id, approval date, and rejected reasons
print("Branches DF")
branches_df.info()   ## there is no null value in this df
print("Customers DF")
customers_df.info()  ## there is no null value in this df
print("Defaults DF")
defaults_df.info()   ## recovery status nulls
print("Loans DF")
loans_df.info()      ## Collateral_Details nulls
print("Transactions DF")
transactions_df.info() ## there is no null value in this df
'''

# cleaning the data - removing duplicates
applications_df.drop_duplicates(inplace=True)
branches_df.drop_duplicates(inplace=True)
customers_df.drop_duplicates(inplace=True)
defaults_df.drop_duplicates(inplace=True)
loans_df.drop_duplicates(inplace=True)
transactions_df.drop_duplicates(inplace=True)

# Standardizing date
applications_df["Application_Date"] = pd.to_datetime(applications_df["Application_Date"])
applications_df["Approval_Date"] = pd.to_datetime(applications_df["Approval_Date"])
defaults_df["Default_Date"] = pd.to_datetime(defaults_df["Default_Date"])
loans_df["Disbursal_Date"] = pd.to_datetime(loans_df["Disbursal_Date"])
loans_df["Repayment_Start_Date"] = pd.to_datetime(loans_df["Repayment_Start_Date"])
loans_df["Repayment_End_Date"] = pd.to_datetime(loans_df["Repayment_End_Date"])
transactions_df["Transaction_Date"] = pd.to_datetime(transactions_df["Transaction_Date"])

# handle missing values
applications_df["Rejection_Reason"].fillna("Not Applicable", inplace= True)

'''TASK 2. Descriptive Analysis
● Summarize and visualize key metrics:
o Distribution of Loan_Amount, EMI_Amount, and Credit_Score.
o Regional trends in loan disbursement and defaults.
o Monthly trends in loan approvals and disbursements.
'''

df = customers_df.merge(applications_df, on="Customer_ID", how="left")
df = df.merge(loans_df, on='Loan_ID', how='left')
df = df.merge(defaults_df, on='Loan_ID', how='left')
region_agg = branches_df.groupby('Region').agg({'Branch_ID': 'count', 'Total_Customers': 'sum', 
                                                'Total_Active_Loans': 'sum', 'Delinquent_Loans': 'sum', 
                                                'Loan_Disbursement_Amount': 'sum','Avg_Processing_Time': 'mean', 
                                                'Relationship_Manager_Count': 'sum'}).reset_index()
df = df.merge(region_agg, on='Region', how='left')


## distribution of loan amount, emi amount and credit score
plt.figure(figsize=(8, 4))
plt.hist(df['Loan_Amount'], bins=100, color='blue', edgecolor='black')
plt.title(f"Distribution of Loan Amount")
plt.xlabel('Loan_Amount')
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8, 4))
plt.hist(df['EMI_Amount'], bins=100, color='blue', edgecolor='black')
plt.title(f"Distribution of EMI Amount")
plt.xlabel('EMI_Amount')
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8, 4))
plt.hist(df['Credit_Score'], bins=100, color='blue', edgecolor='black')
plt.title(f"Distribution of Credit Score")
plt.xlabel('Credit_Score')
plt.ylabel("Frequency")
plt.show()

# Regional trends in loan disbursement and defaults.
#if 'Region' in df.columns:
regional_data = df.groupby('Region').agg(
        Total_Loans=('Loan_ID', 'count'),
        Disbursed=('Approval_Status', lambda x: (x == 'Approved').sum()),
        Defaults=('Default_ID', 'count')
    ).reset_index()

plt.figure(figsize=(8, 5))
plt.bar(regional_data['Region'], regional_data['Disbursed'], color='green', label='Disbursed')
plt.bar(regional_data['Region'], regional_data['Defaults'], color='red', label='Defaults')
plt.title("Regional Loan Disbursement & Defaults")
plt.xlabel("Region")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.legend()
plt.show()

# Monthly trends in loan approvals and disbursements.
if 'Approval_Date' in df.columns:
    df['Approval_Month'] = df['Approval_Date'].dt.to_period('M')
    monthly_data = df.groupby('Approval_Month').agg(
        Approvals=('Approval_Status', lambda x: (x == 'Approved').sum()),
        Total_Loans=('Loan_ID', 'count')
    ).reset_index()

plt.figure(figsize=(10, 6))
plt.plot(monthly_data['Approval_Month'].astype(str), monthly_data['Approvals'],
             marker='o', label='Approvals')
plt.plot(monthly_data['Approval_Month'].astype(str), monthly_data['Total_Loans'],
             marker='x', label='Total Loans')
plt.title("Monthly Loan Approvals & Disbursements")
plt.xlabel("Month")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


'''TASK 3. Default Risk Analysis
● Correlation Between Loan Attributes and Defaults:
o Calculate correlations between Loan_Amount, Interest_Rate,Credit_Score, and Default_Flag (a binary indicator for default).
● Pairwise Correlation Analysis:
o Create a heatmap to visualize the correlations between key
variables, such as EMI_Amount, Overdue_Amount, and
Default_Amount.
● Correlation Between Branch Metrics and Defaults: o Analyze the relationship between branch performance metrics (e.g.,
Delinquent_Loans, Loan_Disbursement_Amount) and default rates.
'''
df['Default_Flag'] = df['Default_Amount'].apply(lambda x: 1 if x > 0 else 0)

# Correlation Between Loan Attributes and Defaults:
corr_matrix = df[['Loan_Amount', 'Interest_Rate', 'Credit_Score', 'Default_Flag']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title
plt.show()

# Pairwise Correlation Analysis:
corr_matrix = df[['EMI_Amount', 'Overdue_Amount', 'Default_Amount']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Correlation Between Branch Metrics and Defaults:
corr_matrix = df[['Delinquent_Loans', 'Loan_Disbursement_Amount', 'Default_Flag']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

'''
TASK 4. Branch and Regional Performance
● Rank branches by:
o Loan disbursement volume.
o Processing time efficiency.
o Default rates and recovery rates.
● Compare branch performance across regions.
'''

branch_rank_by_disbursement = branches_df.groupby('Branch_Name')['Loan_Disbursement_Amount'].sum().sort_values(ascending=False).reset_index()
branch_rank_by_disbursement.display()

branch_rank_by_processing_time = branches_df.groupby('Branch_Name')['Avg_Processing_Time'].mean().sort_values(ascending=True).reset_index()
branch_rank_by_processing_time.display()

df_region_metrics = branches_df.merge(df.groupby('Region').agg({'Default_Flag': 'sum','Recovery_Amount': 'sum'}).reset_index(), on='Region', how='left')

branch_rank_by_default_flag = df_region_metrics.groupby('Branch_Name')['Default_Flag'].sum().sort_values(ascending=False).reset_index()
branch_rank_by_default_flag.display()

branch_rank_by_recovery_rate = df_region_metrics.groupby('Branch_Name')['Recovery_Amount'].sum().sort_values(ascending=True).reset_index()
branch_rank_by_recovery_rate.display()

branch_performance_region = branches_df.groupby(['Branch_Name', 'Region'])['Total_Active_Loans'].sum().sort_values(ascending=False).reset_index()
branch_performance_region.display()

''' TASK 5. Customer Segmentation
● Segment customers by income, credit score, and loan status.
● Identify high-risk and high-value customer groups.
● Analyze repayment behavior across segments'''

customer_segmentation = df.groupby(['Annual_Income', 'Credit_Score', 'Loan_Status'])['Customer_ID'].count().reset_index()
customer_segmentation.display()

high_risk_customers = df[df['Default_Flag'] == 1].groupby(['Annual_Income', 'Credit_Score'])['Customer_ID'].count().reset_index()
high_risk_customers.display()

high_value_customers = df[df['Loan_Amount'] > df['Loan_Amount'].quantile(0.9)].groupby(['Annual_Income', 'Credit_Score'])['Customer_ID'].count().reset_index()
high_value_customers.display()

repayment_behavior = df.groupby(['Annual_Income', 'Credit_Score', 'Loan_Status'])['EMI_Amount'].mean().reset_index()
repayment_behavior.display()


'''TASK 6. Advanced Statistical Analysis
1. Correlation Analysis for Default Risks:
o Examine the correlation between Credit_Score, Loan_Amount, Interest_Rate, Overdue_Amount, and Default_Flag.
2. Pairwise Correlation Heatmap:
o Generate a heatmap to visualize correlations among key variables like EMI_Amount, Recovery_Rate, and Default_Amount.
3. Branch-Level Correlation:
o Explore the relationship between branch performance metrics (Delinquent_Loans, Loan_Disbursement_Amount, Recovery_Rate)
and overall efficiency.'''

# correlation between Credit_Score, Loan_Amount, Interest_Rate, Overdue_Amount, and Default_Flag
corr_matrix = df[['Loan_Amount', 'Interest_Rate', 'Credit_Score', 'Loan_Amount', 'Default_Flag']].corr()
corr_matrix.display()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title
plt.show()

df['Recovery_Rate'] = df['Recovery_Amount']/df['Default_Amount']

# Pairwise Correlation Heatmap:
corr_matrix = df[['EMI_Amount', 'Recovery_Rate', 'Default_Amount']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

df['overall_efficiency'] = (df['Recovery_Amount']/df['Loan_Disbursement_Amount']) * 100

# Branch level Correlation:
corr_matrix = df[['Delinquent_Loans', 'Loan_Disbursement_Amount', 'Recovery_Rate','overall_efficiency']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

'''TASK 7. Transaction and Recovery Analysis
● Analyze penalty payments and overdue trends.
● Evaluate recovery rates by Default_Reason and Legal_Action.
● Compare recovery rates across regions and branches.'''

df_trx = transactions_df[transactions_df['Payment_Type']=='Penalty'].groupby(transactions_df['Transaction_Date'].dt.year).agg({'Amount': 'sum','Overdue_Fee': 'sum'}).reset_index()
df_trx.display()

plt.plot(df_trx['Transaction_Date'], df_trx['Amount'])
plt.xlabel('Year')
plt.ylabel('Total Amount')
plt.show()

plt.plot(df_trx['Transaction_Date'], df_trx['Overdue_Fee'])
plt.xlabel('Year')
plt.ylabel('Total Overdue_Fee')
plt.show()

df.groupby(['Default_Reason', 'Legal_Action'])['Recovery_Rate'].mean().reset_index().display()

df_region_metrics = branches_df.merge(df.groupby('Region').agg({'Recovery_Rate': 'mean'}).reset_index(), on='Region', how='left')

df_region_metrics.groupby(['Region', 'Branch_Name'])['Recovery_Rate'].mean().reset_index().display()

'''TASK 8. EMI Analysis
● Analyze the relationship between EMI amounts and default probabilities.
● Identify thresholds for EMI amounts where defaults are most likely.
● Compare EMI trends across loan types. '''

# Relationship btw EMI amount and DEfault probabilities
df['EMI_group'] = pd.cut(df['EMI_Amount'], bins=10)

default_probability = df.groupby('EMI_group')['Default_Flag'].mean().reset_index()
default_probability.display()

plt.figure(figsize=(8, 6))
plt.bar(default_probability['EMI_group'].astype(str), default_probability['Default_Flag'])
plt.xticks(rotation=45)
plt.xlabel('EMI Group')
plt.ylabel('Default Probability')
plt.title('Default Probability by EMI Group')
plt.show()

# EMI thresholds for default probability-
if 'EMI_Amount' in df.columns and 'Default_Flag' in df.columns:
    threshold_analysis = df.groupby('EMI_group')['Default_Flag'].mean().reset_index()
    threshold_analysis = threshold_analysis.sort_values(by='Default_Flag', ascending=False)

    print("\nTop EMI ranges with highest default probability:")
    print(threshold_analysis.head())

# EMI trends across loan types
df.groupby(['Loan_Purpose', 'EMI_group'])['EMI_Amount'].mean().reset_index().display()
plt.figure(figsize=(8, 6))
sns.boxplot(x='Loan_Purpose', y='EMI_Amount', data=df)
plt.xticks(rotation=45)
plt.xlabel('Loan Purpose')
plt.ylabel('EMI Amount')
plt.title('EMI Amount by Loan Purpose')
plt.show()

'''TASK 9. Loan Application Insights
● Calculate approval and rejection rates for loan applications.
● Identify the most common reasons for loan rejection.
● Compare application processing fees between approved and rejected
applications.'''

# Approval and Rejection Rate
approval_reject_count = df.groupby('Approval_Status')['Application_ID'].count().reset_index()
approval_reject_count.display()

print("Approval Rate:", approval_reject_count.loc[approval_reject_count['Approval_Status'] == 'Approved', 'Application_ID'].values[0] / len(df))
print("Rejection Rate:", approval_reject_count.loc[approval_reject_count['Approval_Status'] == 'Rejected', 'Application_ID'].values[0] / len(df))

print("Most Common Reasons of Loan Rejection:")
rejection_reason = df[df['Approval_Status'] == 'Rejected'].groupby('Rejection_Reason')['Application_ID'].count().reset_index().sort_values(by='Application_ID', ascending=False)
rejection_reason.display()

processing_fee = df.groupby('Approval_Status')['Processing_Fee'].mean().reset_index()
processing_fee.display()

'''TASK 10. Recovery Effectiveness
● Determine the effectiveness of recovery efforts by calculating the ratio of Recovery_Amount to Default_Amount.
● Compare recovery rates for defaults with and without legal actions.
● Analyze branch-wise recovery performance.'''

# effectiveness of recovery efforts by calculating the ratio of Recovery_Amount to Default_Amount.
df['Recovery_Rate'] = df['Recovery_Amount'] / df['Default_Amount']
print("Overall Recovery Rate:", df['Recovery_Rate'].mean())

# Comparison btw recovery rates for defaults with and without legal actions.
recovery_rate_by_action = df.groupby('Legal_Action')['Recovery_Rate'].mean().reset_index()
print("Recovery Rate by Legal Action:")
recovery_rate_by_action.display()

# Branch-wise recovery performance
branch_recovery_df = branches_df.merge(df[['Region','Recovery_Rate']], on='Region', how='inner')
print("Branch-wise Recovery Performance:")
branch_recovery_df.groupby('Branch_Name')['Recovery_Rate'].mean().reset_index().sort_values(by='Recovery_Rate', ascending=False).display()


'''TASK 11. Loan Disbursement Efficiency
● Analyze the time from application to loan disbursement and identify bottlenecks.
● Compare average processing times across branches.
● Evaluate disbursement trends by loan purpose and region.'''

# time from application to loan disbursement and identify bottlenecks.
df['Disbursement_Time'] = (pd.to_datetime(df['Disbursal_Date']) - pd.to_datetime(df['Application_Date'])).dt.days
print("Average Disbursement Time:", df['Disbursement_Time'].mean())

plt.figure(figsize=(10, 6))
sns.histplot(df['Disbursement_Time'], bins=30, kde=True, color='blue')
plt.xlabel('Disbursement Time (Days)')
plt.ylabel('Frequency')
plt.title('Distribution of Disbursement Time')
plt.show()

# average processing time accross branches-
branch_recovery_df = branches_df.merge(df[['Region','Disbursement_Time']], on='Region', how='inner')
print("Average Processing Time by Branch:")
branch_recovery_df.groupby('Branch_Name')['Disbursement_Time'].mean().reset_index().sort_values(by='Disbursement_Time', ascending=False).display()

# Disbursements trends by loan purose and region
df_loan_disburse_metrics = df[df['Disbursal_Date'].notnull()].groupby(['Loan_Purpose', 'Region'])['Disbursement_Time'].mean().reset_index().pivot(index='Loan_Purpose', columns='Region', values='Disbursement_Time')
print("Average Disbursement Time by Loan Purpose and Region")
df_loan_disburse_metrics.display()
df_loan_disburse_metrics.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.xlabel('Loan Purpose')
plt.ylabel('Number of Loans')
plt.title('Loan Disbursement Trends by Purpose and Region')
plt.show()


'''TASK 12. Profitability Analysis
● Calculate the total interest income generated across all loans.
● Identify the most profitable loan purposes based on interest earnings.
● Compare profitability metrics for branches across regions.'''

# total interest income generated across all loans
df['Interest_Income'] = (df['Loan_Amount'] * df['Interest_Rate'])/100
print("Total Interest Income:", df['Interest_Income'].sum())

# Most profitable loan based on interest earning
print("Most Profitable Loan Purpose:")
df.groupby('Loan_Purpose')['Interest_Income'].sum().reset_index().sort_values(by='Interest_Income', ascending=False).head(1).display()

# Compare profitability metrics for branches across regions
branch_profit_df = branches_df.merge(df[['Region','Interest_Income']], on='Region', how='inner')
print("Branch-wise Profitability:")
branch_profit_df.groupby('Branch_Name')['Interest_Income'].sum().reset_index().sort_values(by='Interest_Income', ascending=False).display()

'''TASK 13. Geospatial Analysis
● Map the distribution of active loans across regions.
● Compare default rates across different geographic regions.
● Visualize the loan disbursement trends for rural vs. urban areas.'''

# Map the distribution of active loans across regions
loan_distribution = df.groupby('Region')['Loan_ID'].count().reset_index()
loan_distribution.display()
sns.boxplot(x='Region', y='Loan_ID', data=loan_distribution, palette='Reds')
plt.xlabel('Region')
plt.xticks(rotation=45)
plt.ylabel('Number of Loans')
plt.title('Distribution of Active Loans by Region')
plt.show()

# Comparison of default rates across diff regions
default_rate = df.groupby('Region')['Default_Flag'].mean().reset_index()
default_rate['Default_Flag'] = default_rate['Default_Flag'] * 100
default_rate = default_rate.rename(columns={'Default_Flag': 'Default_rate_percent'})
default_rate.display()
sns.boxplot(x='Region', y='Default_rate_percent', data=default_rate, palette='Blues')
plt.xlabel('Region')
plt.xticks(rotation=45)
plt.ylabel('Default Rate Percentage')
plt.title('Default Rate by Region')
plt.show()

# Visualize the loan disbursement trends for rural vs. urban areas -- there is no field showing Rural/Urban region

'''TASK 14. Default Trends
● Analyze the number of defaults over time to identify patterns.
● Calculate the average default amount for different loan purposes.
● Compare default rates across customer income categories.
'''

# number of defaults over time to identify patterns
df['Default_Year'] = df['Default_Date'].dt.year
df['Default_Month'] = df['Default_Date'].dt.month
defaults_by_year = df.groupby(['Default_Year','Default_Month'])['Default_ID'].count().reset_index()
defaults_by_year.display()

sns.lineplot(x='Default_Month', y='Default_ID', hue='Default_Year', data=defaults_by_year)
plt.xlabel('Month')
plt.ylabel('Number of Defaults')
plt.title('Number of Defaults Over Time')
plt.show()

# average default amount for different loan purposes
avg_default_amount = df.groupby('Loan_Purpose')['Default_Amount'].mean().reset_index()
print("Average Default Amount by Loan Purpose:")
avg_default_amount.display()

# default rates across customer income categories
df['Income_Category'] = pd.cut(df['Annual_Income'], bins=10)
default_rates_by_income = df.groupby('Income_Category')['Default_Flag'].mean().reset_index()
print("Default Rates by Income Category:")
default_rates_by_income['Default_Flag'] = default_rates_by_income['Default_Flag'] * 100
default_rates_by_income.rename(columns={'Default_Flag': 'Default_Rate'}, inplace=True)
default_rates_by_income = default_rates_by_income.sort_values(by='Default_Rate', ascending=False)
default_rates_by_income.display()

'''TASK 15. Branch Efficiency
● Calculate the average loan disbursement time for each branch.
● Identify branches with the highest number of rejected applications.
● Compare branch efficiency based on customer satisfaction metrics (if available).
'''

# Average loan disbursement time for each branch
region_avg_disbursement_time = df.groupby('Region')['Disbursement_Time'].mean().reset_index()
avg_disburesement_time_by_branch = branches_df.merge(region_avg_disbursement_time, on='Region', how='inner').groupby('Branch_Name')['Disbursement_Time'].mean().reset_index()
avg_disburesement_time_by_branch.display()

# Branches with highest number of Rejected applications-
df['Rejected_Flag'] = df['Rejection_Reason'].apply(lambda x: 1 if x != 'Not Applicable' else 0)
rejected_applications = df.groupby('Region')['Rejected_Flag'].sum().reset_index().merge(branches_df, on= 'Region', how='inner')
rejected_applications= rejected_applications.groupby('Branch_Name')['Rejected_Flag'].sum().reset_index().sort_values(by='Rejected_Flag', ascending=False)
rejected_applications.display()


'''TASK 16. Time-Series Analysis
● Analyze monthly loan disbursement trends over the last 5 years.
● Identify seasonal patterns in loan applications and disbursements.
● Compare monthly default rates across regions.
'''
# monthly loan disbursement trends over the last 5 years.
df['Disbursement_Year'] = df['Disbursal_Date'].dt.year
df['Disbursement_Month'] = df['Disbursal_Date'].dt.month
monthly_disburse = df[df['Disbursement_Year'] >= 2020].groupby(['Disbursement_Year', 'Disbursement_Month'])['Loan_ID'].count().reset_index()
monthly_disburse.display()
sns.lineplot(x='Disbursement_Month', y='Loan_ID', hue='Disbursement_Year', data=monthly_disburse)
plt.xlabel('Month')
plt.ylabel('Number of Loans Disbursed') 
plt.title('Monthly Loan Disbursement Trends')
plt.show()

# seasonal pattern in loan applications and disbursement
df['Application_month'] = df['Application_Date'].dt.month
df['Disbursement_month'] = df['Disbursal_Date'].dt.month
seasonal_pattern = df.groupby(['Application_month', 'Disbursement_month'])['Loan_ID'].count().reset_index()
seasonal_pattern.display()
sns.lineplot(x='Application_month', y='Loan_ID', data=seasonal_pattern, markers="o", label='Application')
sns.lineplot(x='Disbursement_month', y='Loan_ID', data=seasonal_pattern, markers="o", label='Disbursement')
plt.legend(['Application', 'Disbursement'])
plt.ylabel('Number of Loans')
plt.xlabel('Month')
plt.title('Seasonal Patterns in Loan Applications and Disbursements')
plt.show()

# Monthly Default Rates across regions
df['Default_Month'] = df['Default_Date'].dt.month
monthly_default_rates = df.groupby(['Region', 'Default_Month'])['Default_ID'].count().reset_index()
monthly_default_rates.display()
sns.lineplot(x='Default_Month', y='Default_ID', hue='Region', data=monthly_default_rates)
plt.xlabel('Month')
plt.ylabel('Number of Defaults') 
plt.title('Monthly Default Rates across Regions')
plt.show()

'''TASK 17. Customer Behavior Analysis
● Categorize customers based on their repayment behavior (e.g., always on time, occasional defaulters, frequent defaulters).
● Analyze patterns in loan approval and rejection reasons segmented by customer demographics.
● Identify high-value customers with consistent repayment histories.'''

# Customer Repayment behaviour
customer_behaviour = df.groupby('Customer_ID')['Default_Flag'].agg(total_loans = 'count', total_defaults = 'sum').reset_index()
def category(x):
    if x['total_defaults'] == 0:
        return 'Always on time'
    elif x['total_defaults'] / x['total_loans'] * 100 < 5:
        return 'Occasional Defaulters'
    else:
        return 'Frequent Defaulters'

customer_behaviour['Repayment_Category'] = customer_behaviour.apply(category, axis=1)
customer_behaviour.display()

# Loan approval and rejection reasons segmented by customer demographics
if 'Approval_Status' in df.columns and 'Rejection_Reason' in df.columns:
    # Example demographics: Age, Gender, Income_Category (if available)
    demo_cols = [col for col in ['Gender','Income_Category','Age'] if col in df.columns]

    for col in demo_cols:
        plt.figure(figsize=(10,5))
        sns.countplot(x=col, hue='Approval_Status', data=df, palette="coolwarm")
        plt.title(f"Loan Approvals & Rejections by {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.legend(title="Approval Status")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Rejection reasons distribution
    rejection_reasons = df[df['Approval_Status']=="Rejected"]['Rejection_Reason'].value_counts().reset_index()
    rejection_reasons.columns = ['Reason','Count']
    print("\nMost Common Rejection Reasons:")
    print(rejection_reasons.head())

    plt.figure(figsize=(10,5))
    sns.barplot(x='Count', y='Reason', data=rejection_reasons, palette="Reds_r")
    plt.title("Most Common Loan Rejection Reasons")
    plt.xlabel("Number of Rejections")
    plt.ylabel("Reason")
    plt.tight_layout()
    plt.show()

    # high value customer with consistent repayment history
    if 'Loan_Amount' in df.columns:
        high_value_customers = df[df['Loan_Amount'].notnull()].groupby('Customer_ID').agg(Total_Loan_Amount=('Loan_Amount','sum'),Total_Loans=('Loan_ID','count'),Defaults=('Default_Flag','sum')).reset_index()

    # Consistent payers = no defaults, high loan amounts
    high_value_customers = high_value_customers[
        (high_value_customers['Defaults']==0) & (high_value_customers['Total_Loan_Amount'] > df['Loan_Amount'].median())
    ]

    print("\nHigh-Value Customers with Consistent Repayment:")
    print(high_value_customers.head())

    plt.figure(figsize=(8,5))
    sns.histplot(high_value_customers['Total_Loan_Amount'], bins=30, kde=True, color="green")
    plt.title("Distribution of Loan Amounts - High Value Consistent Customers")
    plt.xlabel("Total Loan Amount")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.show()


'''TASK 18. Risk Assessment
● Develop a risk matrix for loan products based on Default_Amount,Loan_Term, and Interest_Rate.
● Rank loan types by risk level and suggest mitigation strategies.
● Analyze high-risk customer segments by credit score a
'''

# DEVELOP RISK MATRIX (Loan-Level)
risk_df = df[['Loan_ID','Loan_Purpose','Default_Amount','Loan_Term','Interest_Rate']].dropna()

risk_df['Risk_Score'] = (
    risk_df['Default_Amount'].rank(pct=True) * 0.5 +
    risk_df['Loan_Term'].rank(pct=True) * 0.2 +
    risk_df['Interest_Rate'].rank(pct=True) * 0.3
)

loan_risk_matrix = risk_df.groupby('Loan_Purpose').agg(
    Avg_Default_Amount=('Default_Amount','mean'),
    Avg_Loan_Term=('Loan_Term','mean'),
    Avg_Interest_Rate=('Interest_Rate','mean'),
    Avg_Risk_Score=('Risk_Score','mean')
).reset_index()

# Rank loan types by risk
loan_risk_matrix['Risk_Rank'] = loan_risk_matrix['Avg_Risk_Score'].rank(ascending=False,method="dense")
loan_risk_matrix.display()

plt.figure(figsize=(8,6))
sns.heatmap(
    loan_risk_matrix.set_index('Loan_Purpose')[['Avg_Default_Amount','Avg_Loan_Term','Avg_Interest_Rate','Avg_Risk_Score']],
    annot=True, fmt=".2f", cmap="Reds"
)
plt.title("Risk Matrix for Loan Products")
plt.tight_layout()
plt.show()


# HIGH-RISK CUSTOMER SEGMENTS
if 'Credit_Score' in df.columns and 'Annual_Income' in df.columns:
    high_risk_customers = df.groupby('Customer_ID').agg(
        Avg_Credit_Score=('Credit_Score','mean'),
        Avg_Income=('Annual_Income','mean'),
        Total_Default_Amount=('Default_Amount','sum')
    ).reset_index()

    conditions = (
        (high_risk_customers['Avg_Credit_Score'] < df['Credit_Score'].quantile(0.25)) &
        (high_risk_customers['Total_Default_Amount'] > 0) &
        (high_risk_customers['Avg_Income'] < df['Annual_Income'].median())
    )
    high_risk_segment = high_risk_customers[conditions]

    print("\n🚨 High-Risk Customer Segment (Low Credit, Defaults, Low Income):")
    print(high_risk_segment.head())

    plt.figure(figsize=(8,6))
    sns.scatterplot(
        x='Avg_Income', y='Avg_Credit_Score',
        size='Total_Default_Amount', data=high_risk_customers,
        sizes=(20,200), hue=(conditions.map({True:'High Risk',False:'Low/Medium Risk'}))
    )
    plt.title("Customer Risk Segmentation by Credit Score & Income")
    plt.xlabel("Average Income")
    plt.ylabel("Average Credit Score")
    plt.legend(title="Risk Category")
    plt.tight_layout()
    plt.show()

    '''TASK 19. Time to Default Analysis
● Calculate the average time from loan disbursement to default for overdue loans.
● Identify loan purposes with the shortest time to default.
● Compare the time to default across customer demographics.'''

# average time from loan disbursement to default for overdue loans.
df['Time_to_Default'] = (df['Default_Date'] - df['Disbursal_Date']).dt.days
overdue_loans = df[df['Time_to_Default'] > 0]

avg_time_to_default = overdue_loans['Time_to_Default'].mean()
print(f"Average Time to Default for Overdue Loans: {avg_time_to_default:.2f} days")

# Loan purposes with shortes time to default
purpose_defaults = overdue_loans.groupby('Loan_Purpose')['Time_to_Default'].mean().sort_values()
print("\nLoan Purposes with Shortest Time to Default:")
print(purpose_defaults.head())

# Compare time to default across customer demographics
demographic_defaults = overdue_loans.groupby(['Gender', 'Employment_Status'])['Time_to_Default'].mean().reset_index()
print("\nTime to Default by Customer Demographics:")
print(demographic_defaults)

sns.barplot(x='Gender', y='Time_to_Default', hue='Employment_Status', data=demographic_defaults)
plt.title("Time to Default by Customer Demographics")
plt.xlabel("Gender")
plt.ylabel("Average Time to Default")
plt.legend(title="Employment Status")
plt.tight_layout()
plt.show()

'''TASK 20. Transaction Pattern Analysis
● Identify customers with irregular repayment patterns.
● Analyze penalty payments as a proportion of total transactions.
● Compare transaction amounts for overdue vs. non-overdue loans.'''

# customers with irregular repayments
df_new = customers_df.merge(loans_df, on='Customer_ID', how='left').merge(transactions_df, on='Loan_ID', how='left')
irregular_customers = df_new.groupby('Customer_ID_x')['Overdue_Fee'].sum().sort_values(ascending=False).head(5)
print("Customers with Irregular Repayment Patterns:")
print(irregular_customers)

# panelty payments as proportion of total transactions
panalty_ratio = df_new.groupby('Customer_ID_x').apply(lambda x: x[x['Payment_Type']=='Penalty']['Amount'].sum() /
                  x['Amount'].sum()
        if x['Amount'].sum() > 0 else 0
    ).reset_index(name='Penalty_Ratio')
print("\nPenalty Payments as Proportion of Total Transactions:")

print(panalty_ratio.head())

sns.histplot(panalty_ratio['Penalty_Ratio'], bins=30)
plt.title("Penalty Payments as Proportion of Total Transactions")
plt.xlabel("Penalty Ratio")
plt.ylabel("Number of Customers")  
plt.tight_layout()
plt.show()

# compare transaction amounts for overdue vs. non-overdue loans
df_new['Overdue_Flag'] = df_new['Overdue_Amount'].apply(lambda x: 1 if x > 0 else 0)

if 'Overdue_Flag' in df_new.columns:
    overdue_comparison = df_new.groupby('Overdue_Flag')['Amount'].mean().reset_index()

    print("\n💰 Avg Transaction Amount - Overdue vs Non-Overdue Loans:")
    print(overdue_comparison)

    plt.figure(figsize=(6,5))
    sns.boxplot(x='Overdue_Flag', y='Amount', data=df_new, palette="Set2")
    plt.title("Transaction Amounts: Overdue vs Non-Overdue Loans")
    plt.xlabel("Overdue Loan?")
    plt.ylabel("Transaction Amount")
    plt.tight_layout()
    plt.show()