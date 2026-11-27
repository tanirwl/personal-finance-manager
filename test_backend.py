

from backend import (
    initialize_database,
    add_transaction,
    get_transactions,
    get_summary,
    delete_transaction
)

# Step 1: Ensure database and table exist
initialize_database()

# Step 2: Add salary
add_transaction(
    transaction_type="income",
    amount=30000,
    category="Salary",
    description="Monthly salary"
)

# Step 3: Add food expense
add_transaction(
    transaction_type="expense",
    amount= 1500,
    category="Food",
    description="Food and groceries"
)

# Step 4: Display all transactions
print("\n--- All Transactions ---")
for transaction in get_transactions():
    print(transaction)

# Step 5: Display financial summary
print("\n--- Financial Summary ---")
print(get_summary())

from backend import delete_transaction


