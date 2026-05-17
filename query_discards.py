from app.db.connection import get_db

def run():
    with get_db() as db:
        print('--- NEWS ---')
        res = db.execute("SELECT title, quality_reason FROM news_articles WHERE quality_status = 'discarded' LIMIT 10;").fetchall()
        for row in res:
            print(f'Title: {row[0][:50]} | Reason: {row[1]}')
        print('--- REDDIT ---')
        res = db.execute("SELECT title, quality_reason FROM reddit_posts WHERE quality_status = 'discarded' LIMIT 10;").fetchall()
        for row in res:
            print(f'Title: {row[0][:50]} | Reason: {row[1]}')

if __name__ == "__main__":
    run()
