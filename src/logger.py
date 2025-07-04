import logging
import os
from datetime import datetime
from project_config import PROJECT_ROOT

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(PROJECT_ROOT, "logs", f"{datetime.now().strftime('%m_%d_%Y_%H_%M')}")
os.makedirs(logs_path, exist_ok=True)
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
        handlers=[
            logging.FileHandler(LOG_FILE_PATH),
            logging.StreamHandler()
        ],
        format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO

)

# Test
# if __name__ == "__main__":
#     try:
#         a = 1 / 0
#     except Exception as e:
#         raise CustomException(e, sys)
#     logging.info("Testing loger, logger has started")
