import pymysql
from app import app, db
from dotenv import load_dotenv
import os

load_dotenv()

def create_database():
    try:
        # Connect to MySQL without selecting a database
        connection = pymysql.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            user=os.getenv('root'),
            password=os.getenv('')
        )
        
        with connection.cursor() as cursor:
            # Create database if it doesn't exist
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {os.getenv('DB_NAME')}")
            print(f"Database '{os.getenv('DB_NAME')}' created or already exists")
            
    except Exception as e:
        print(f" Error creating database: {e}")
    finally:
        connection.close()

def create_tables():
    with app.app_context():
        try:
            # Create all tables based on models
            db.create_all()
            print("All tables created successfully!")
            
            # Print created tables
            from sqlalchemy import inspect
            inspector = inspect(db.engine)
            tables = inspector.get_table_names()
            print("\nCreated tables:")
            for table in tables:
                print(f"- {table}")
                # Print table columns
                columns = inspector.get_columns(table)
                for column in columns:
                    print(f"  └─ {column['name']}: {column['type']}")
                    
        except Exception as e:
            print(f"Error creating tables: {e}")

if __name__ == "__main__":
    create_database()
    create_tables()