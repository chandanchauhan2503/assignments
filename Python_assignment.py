'''
Task 1: Sales Data Summary
Objective: XYZ Retail wants to automate the process of calculating basic sales
metrics.
● Assign the number of units sold in two categories, `Category A` and `Category
B`, to variables.
● Calculate the total units sold, the difference between the categories, and the
ratio of units sold.
● Print these results clearly on the management team.
'''

Category_A = int(input("Enter the No of Units Sold in Category A:"))
Category_B = int(input("Enter the No of Units Sold in Category B:"))

total_units = Category_A + Category_B
difference_units = Category_A - Category_B if Category_A - Category_B > 0 else Category_B - Category_A
ratio_units = Category_A / Category_B

print("Sales Data Summary:")
print("Total Units Sold:",total_units)
print("Difference Between Categories:",difference_units)
print("Ratio of Category A to Category B:",ratio_units)

'''
Task 2: Customer Age Data
Objective: Understanding the age distribution of customers is crucial for marketing
strategies.
● Store a customer's name and age.
● Convert the age into a string and create a personalized marketing message
like "Dear John Doe, at 30, you’re eligible for our premium loyalty program."
● Print the message for use in email campaigns.
'''

customer_name = str(input("Enter the Customer's Name:"))
customer_age = str(input("Enter the Customer's Age:"))

print("Dear", customer_name, ", at ", customer_age,", you’re eligible for our premium loyalty program.")

'''
Task 3: Product List Management
Objective: Efficient management of the product list is essential for inventory control.
● Given a list of product prices, extract the highest and lowest prices.
● Create a new list with the mid-range products.
● Add a new premium product price to the list and print the updated list for the
inventory team.
'''

product_prices = [50, 120, 30, 200, 90, 150, 180]
premium_product_price = 250

highest_price = max(product_prices)
lowest_price = min(product_prices)
mid_range_prod = [i for i in product_prices if highest_price > i > lowest_price]
product_prices.append(premium_product_price)

print("Highest Price:", highest_price)
print("Lowest Price:", lowest_price)
print("Mid-Range Products:", mid_range_prod)
print("Updated Product List with Premium Price:",product_prices)

'''
Task 4: Inventory Lookup
Objective: Quick access to product details is important for customer service representatives.
● Create a dictionary storing key information about a product (e.g.,`product_name`, `SKU`, `price`, `category`).
● Print the product name and SKU when queried by a customer service representative.
'''

product_info = {"product_name": "Wireless Mouse", 
                "SKU": "WM-12345", 
                "price":25.99, 
                "category": "Electronics" }

print("Product Name:",product_info["product_name"], "    Product SKU:", product_info["SKU"])


'''
Task 5: Stock Level Alert System
Objective: Ensuring that stock levels are maintained is critical to avoid stockouts.
● Write the code that takes the stock level as input.
● If stock is below a certain threshold, print a "Reorder Now" alert. If stock is
above the threshold, print "Stock is sufficient."
'''

stock_level = int(input("Enter the Stock Level:"))
threshold = int(input("Enter the Threshold:"))

if(stock_level < threshold):
        print("Reorder Now")
else:
    print("Stock if sufficient")


'''
Task 6: Sales Report Formatting
Objective: Formatting the sales data for management reports is crucial.
● Given a list of products sold, print each product name in uppercase for better visibility in reports.
● Implement both a `for` loop and a `while` loop for this task to ensure code flexibility.
'''

products_sold = ["laptop", "mouse", "keyboard", "monitor", "printer"]

for i in products_sold:
    print(i.upper())

i = 0
while i < len(products_sold):
    print(products_sold[i].upper())
    i = i+1

'''
Task 7: Area Calculation for Store Layout
Objective: Accurate area calculations are needed to plan new store layouts.
● Create a function that calculates the area of a section of the store based on length and width.
● Use this function to calculate and print the area of several store sections.
'''

length = int(input("Enter the Length or the area:"))
width = int(input("Enter the width of the area:"))

def area(a,b):
    c = a * b
    return c

print(f"The area of section 1 is {area(length,width)} square meters.")


'''
Task 8: Customer Feedback Analysis
Objective: Analyzing customer feedback is vital to improving service.
● Write the code to count the number of vowels in a customer feedback message.
● Also, reverse the feedback message for a unique data presentation in reports.
'''

orig_feedback = "I loved the fast and friendly service!"
vowels = "aeiouAEIOU"

print("Customer Feedback Analysis:")
print("Original Feedback:",orig_feedback)
print(f"Number of Vowels: {sum(1 for i in orig_feedback if i in vowels)}")
print(f"Reversed Feedback: {orig_feedback[::-1]}")

'''
Task 9: Price Filtering Tool
Objective: Filtering product prices helps in creating targeted discounts.
● Use list to filter out products priced below a certain threshold from the product list.
● Print the list of eligible products for a discount campaign.
'''

product_prices = [150, 85, 300, 120, 45, 200]
discount_threshold = 100

print(f"Products eligible for the discount campaign: {[i for i in product_prices if i < discount_threshold]}")

'''
Task 10: Daily Sales Average
Objective: Calculate the average daily sales for the past week.
● Given a list of sales figures for the last 7 days, calculate the average sales.
● Print the average sales to help the finance team understand the weekly
performance.
'''

daily_sales = [1200, 1500, 1100, 1800, 1700, 1600, 1400]
print(f"Average Daily Sales for the Past Week: ${round(sum(daily_sales)/len(daily_sales),ndigits=2)}")

'''
Task 11: Customer Segmentation
Objective: Categorize customers based on their total spending.
● Create a list of customer spending amounts.
● Use a loop to categorize customers as "Low", "Medium", or "High" spenders based on their spending amount.
● Print the categorized results to assist in targeted marketing.
'''

customer_spendings = [200, 800, 1500, 3000, 450, 1200]     

print("Customer Categorization Criteria:")
for i in range(0,len(customer_spendings)):
    if customer_spendings[i] < 500:
        category = "Low"
    elif 500 <= customer_spendings[i] < 1500:
        category = "Medium"
    else:
        category = "High"
    print(f"Customer {i+1}: Spending = ${customer_spendings[i]} -> Category: {category}")

'''
Task 12: Discount Calculation
Objective: Automate the calculation of discounts for a promotional campaign.
● Write a code that calculates the final price after applying a discount percentage to a product’s original price.
● Test this function on a list of products with different discounts and print the final prices.
'''

products = [ 
    {"name": "Product A", 
     "original_price": 100, 
     "discount_percentage": 10},
{"name": "Product B", 
 "original_price": 250, 
 "discount_percentage": 20}, 
{"name": "Product C", 
 "original_price": 75, 
 "discount_percentage": 15}, 
 {"name": "Product D",
"original_price": 150, 
"discount_percentage": 5} ]


print("Final Prices After Discounts:")
for i in range(0,len(products)):
      price = int(products[i]["original_price"])
      discount = int(products[i]["discount_percentage"])
      final_price = price - round((price * discount /100), ndigits= 2)
      print(f"Product {i+1}: Original Price = ${price}, Discount = {discount}% Final Price = ${final_price}")

'''
Task 13: Customer Feedback Sentiment Analysis
Objective: Basic sentiment analysis of customer feedback.
● Write the Python code that checks if certain positive or negative words (e.g., "good", "bad", "happy", "disappointed") are present in customer feedback.
● Print "Positive" or "Negative" based on the words found in the feedback.
'''

Customer_Feedback = "I am very happy with the service. It was a good experience!"
Positive_Words = ["good", "happy", "excellent", "great"]
Negative_Words = ["bad", "disappointed", "poor", "terrible"]

Customer_Feedback = Customer_Feedback.lower()

for i in Positive_Words:
    if i in Customer_Feedback:
        print("Customer Feedback Sentiment: Positive")
        break

for i in Negative_Words:
    if i in Customer_Feedback:
        print("Customer Feedback Sentiment: Negative")
        break

'''
Task 14: Employee Salary Increment Calculator
Objective: Calculate the salary increment for employees based on their performance
rating.
● Create a dictionary that stores employee names and their performance ratings.
● Write the code that applies a different increment percentage based on the rating.
● Print the updated salary for each employee.
'''

employees = {
            "Alice": {"current_salary": 50000, "rating": "Excellent"}, 
            "Bob": {"current_salary": 40000, "rating": "Good"}, 
            "Charlie": {"current_salary": 45000, "rating": "Average"},
            "David": {"current_salary": 35000, "rating": "Poor"} 
             }
increments = {
            "Excellent": 20, 
            "Good": 15, 
            "Average": 10, 
            "Poor": 5 
              }

print("Updated Salaries After Increment:")
for key,value in employees.items():
    current_salary = value["current_salary"]
    rating = value["rating"]
    increment_percent = increments[rating]
    updated_salary = current_salary + (current_salary * increment_percent / 100)
    print(f"{key}: Current Salary =${current_salary}, Rating = {rating}, Updated Salary = ${updated_salary}")


'''
Task 15: Stock Replenishment Planning
Objective: Determine which products need replenishment based on sales data.
● Given a list of products and their current stock levels, compare these against a predefined threshold.
● Print a list of products that need to be reordered to maintain adequate stock levels.
'''

products = [ 
            {"product_name": "Product A", "stock": 50}, 
            {"product_name": "Product B", "stock": 150}, 
            {"product_name": "Product C", "stock": 30}, 
            {"product_name": "Product D", "stock": 75}, 
            {"product_name": "Product E", "stock": 20} 
            ]
threshold = 40

print("Products that need replenishment:")

for i in products:
    if i["stock"] <= threshold:
        print(i["product_name"])