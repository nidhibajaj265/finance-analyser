import asyncio
import sys
from loguru import logger
from src.graph import app

logger.remove()
logger.add("app.log", level="INFO", rotation="100 MB", retention="10 days")

if __name__ == "__main__":
    result = asyncio.run(app.ainvoke({}))
    for signal in result["signals"]:
        logger.info(signal)
