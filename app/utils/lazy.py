from functools import cache

@cache
def get_bot_manager():
    from app.services import bot_manager
    return bot_manager
