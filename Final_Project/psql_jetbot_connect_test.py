import psycopg2

# Replace these variables with your actual details
hostname = "192.168.1.214"  # IP address of the Ubuntu host
database = "test"
username = "postgres"
password = "ae8803_master_key"
# port_id = 5432  # Port forwarded on the host

try:
    conn = psycopg2.connect(
        host=hostname,
        dbname=database,
        user=username,
        password=password,  # , port=port_id
    )
    print("Connected to the database")
except (psycopg2.DatabaseError, Exception) as error:
    print(error)
else:
    # Perform database operations
    cur = conn.cursor()
    cur.execute("SELECT VERSION();")
    version = cur.fetchone()
    print("Database version:", version)

    # Close the cursor and connection
    cur.close()
    conn.close()
