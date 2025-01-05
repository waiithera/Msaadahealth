from app import app, db
from sqlalchemy import inspect

def verify_database():
    try:
        # Verify database connection
        with app.app_context():
            db.engine.connect()
            print("✅ Database connection successful!")
            
            # Create tables
            db.create_all()
            print("✅ Tables created successfully!")
            
            # List all tables
            inspector = inspect(db.engine)
            tables = inspector.get_table_names()
            print("\nCreated tables:")
            for table in tables:
                print(f"- {table}")
                
    except Exception as e:
        print(f"❌ Database Error: {str(e)}")

if __name__ == "__main__":
    verify_database()