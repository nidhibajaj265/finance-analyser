from src.config import SUPABASE_URL, SUPABASE_KEY
from supabase import create_client, Client
from src.signals import Signal
from loguru import logger
from datetime import datetime, timedelta, timezone

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def save_signals(entity_signals: list[Signal]) -> None:
    if not entity_signals:
        logger.warning("No signals to save to Database")
        return
    
    rows = [s.model_dump(mode='json') for s in entity_signals]
    supabase.table('signal').insert(rows).execute()
    logger.info(f"Saved {len(rows)} to DB")

def fetch_recent_signals(days: int = 7) ->list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(days)
    response = (
        supabase.table('signals')
        .select('*')
        .gte('timestamp', cutoff)
        .order('timestamp', desc=True)
        .execute)
    
    logger.info(f"Fetched {len(response.data)} signals from DB")