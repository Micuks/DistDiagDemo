#!/usr/bin/env python3

import pymysql
import os
import logging
import sys
from typing import List
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
env_path = Path(__file__).resolve().parent.parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    logger.info(f"Loaded environment variables from {env_path}")
else:
    logger.warning(f".env file not found at {env_path}")

class IndexDropper:
    def __init__(self):
        # OceanBase connection configuration from .env
        self.ob_config = {
            'host': os.getenv('OB_HOST', '127.0.0.1'),
            'port': int(os.getenv('OB_PORT', '2881')),
            'user': os.getenv('OB_USER', 'root@sys'),
            'password': os.getenv('OB_PASSWORD', 'password'),
            'database': os.getenv('OB_DATABASE', 'oceanbase'),
        }
        logger.info(f"Using database configuration - host: {self.ob_config['host']}, port: {self.ob_config['port']}, user: {self.ob_config['user']}, database: {self.ob_config['database']}")
        self.ob_zones = ['zone1', 'zone2', 'zone3']

    def get_drop_commands(self) -> List[str]:
        """Get all drop index commands for both tpcc and sbtest databases"""
        tpcc_drop_commands = [
            "USE tpcc",
            # Drop customer table indexes
            "DROP INDEX idx_customer_1 ON customer",
            "DROP INDEX idx_customer_2 ON customer",
            "DROP INDEX idx_customer_3 ON customer",
            "DROP INDEX idx_customer_4 ON customer",
            "DROP INDEX idx_customer_5 ON customer",
            # Drop district table indexes
            "DROP INDEX idx_district_1 ON district",
            "DROP INDEX idx_district_2 ON district",
            "DROP INDEX idx_district_3 ON district",
            # Drop history table indexes
            "DROP INDEX idx_history_1 ON history",
            "DROP INDEX idx_history_2 ON history",
            "DROP INDEX idx_history_3 ON history",
            # Drop item table indexes
            "DROP INDEX idx_item_1 ON item",
            "DROP INDEX idx_item_2 ON item",
            "DROP INDEX idx_item_3 ON item",
            # Drop new_orders table indexes
            "DROP INDEX idx_new_orders_1 ON new_orders",
            "DROP INDEX idx_new_orders_2 ON new_orders",
            # Drop order_line table indexes
            "DROP INDEX idx_order_line_1 ON order_line",
            "DROP INDEX idx_order_line_2 ON order_line",
            "DROP INDEX idx_order_line_3 ON order_line",
            "DROP INDEX idx_order_line_4 ON order_line",
            # Drop orders table indexes
            "DROP INDEX idx_orders_1 ON orders",
            "DROP INDEX idx_orders_2 ON orders",
            "DROP INDEX idx_orders_3 ON orders",
            "DROP INDEX idx_orders_4 ON orders",
            # Drop stock table indexes
            "DROP INDEX idx_stock_1 ON stock",
            "DROP INDEX idx_stock_2 ON stock",
            "DROP INDEX idx_stock_3 ON stock",
            # Drop warehouse table indexes
            "DROP INDEX idx_warehouse_1 ON warehouse",
            "DROP INDEX idx_warehouse_2 ON warehouse"
        ]

        sbtest_drop_commands = [
            "USE sbtest"
        ]

        # Generate drop commands for all sbtest tables (1-10)
        for i in range(1, 11):
            sbtest_drop_commands.extend([
                f"DROP INDEX idx_sbtest{i}_1 ON sbtest{i}",
                f"DROP INDEX idx_sbtest{i}_2 ON sbtest{i}",
                f"DROP INDEX idx_sbtest{i}_3 ON sbtest{i}"
            ])

        return tpcc_drop_commands + sbtest_drop_commands

    def execute_drop_commands(self) -> None:
        """Execute all drop commands to remove indexes"""
        try:
            conn = pymysql.connect(**self.ob_config)
            with conn.cursor() as cursor:
                current_db = None
                commands = self.get_drop_commands()
                
                for cmd in commands:
                    # Check if command is for changing database
                    if cmd.lower().startswith("use "):
                        current_db = cmd.split()[1].strip()
                        logger.info(f"Switching to database: {current_db}")
                    
                    logger.info(f"Executing: {cmd}")
                    try:
                        cursor.execute(cmd)
                    except pymysql.err.OperationalError as e:
                        # Handle errors but continue with other commands
                        if "Unknown index" in str(e) or "doesn't exist" in str(e):
                            logger.warning(f"Index already dropped or doesn't exist: {cmd}")
                        else:
                            logger.error(f"Error executing command: {cmd}, error: {str(e)}")
                        continue
                    except Exception as e:
                        logger.error(f"Unexpected error executing command: {cmd}, error: {str(e)}")
                        continue

            conn.commit()
            conn.close()
            logger.info("Successfully completed index dropping operations")

        except Exception as e:
            logger.error(f"Failed to execute drop commands: {str(e)}")
            sys.exit(1)

def main():
    try:
        logger.info("Starting index dropping process...")
        dropper = IndexDropper()
        dropper.execute_drop_commands()
        logger.info("Successfully completed dropping all indexes")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 