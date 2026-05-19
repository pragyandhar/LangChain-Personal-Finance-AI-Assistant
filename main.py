from rag.loader import TransactionLoader
import os

def main():
    # Define paths
    csv_file = "data/transactions.csv"
    db_path = "data/transactions.db"
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
        
    # Initialize loader
    loader = TransactionLoader(db_path=db_path)
    
    print(f"Starting transaction sync from {csv_file}...")
    
    try:
        # Checking if CSV exists, if not, we cannot proceed
        if not os.path.exists(csv_file):
            print(f"Error: {csv_file} not found. Please place your transactions CSV in the data folder.")
            return []

        # Load and sync to DB
        transactions = loader.load_from_csv(csv_file)
        loader.sync_to_db(transactions)
        
        print("Sync completed successfully.")
        return transactions
        
    except Exception as e:
        print(f"Error during execution: {e}")
        return []

if __name__ == "__main__":
    main()
