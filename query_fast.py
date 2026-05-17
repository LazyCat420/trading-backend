import psycopg

# Connect directly using the known hardcoded URL
conn = psycopg.connect("postgresql://localhost:5432/trading_bot")
cur = conn.cursor()
print('--- NEWS ---')
cur.execute("SELECT title, quality_reason FROM news_articles WHERE quality_status = 'discarded' LIMIT 5;")
for row in cur.fetchall():
    print(f'Title: {str(row[0])[:50]} | Reason: {str(row[1])}')
print('--- REDDIT ---')
cur.execute("SELECT title, quality_reason FROM reddit_posts WHERE quality_status = 'discarded' LIMIT 5;")
for row in cur.fetchall():
    print(f'Title: {str(row[0])[:50]} | Reason: {str(row[1])}')
cur.close()
conn.close()
