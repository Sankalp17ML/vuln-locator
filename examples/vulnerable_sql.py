def get_user_profile(db_cursor, username):
    # Vulnerable: Raw string concatenation in SQL query
    query = "SELECT id, username, email FROM users WHERE username = '" + username + "'"
    db_cursor.execute(query)
    return db_cursor.fetchall()
