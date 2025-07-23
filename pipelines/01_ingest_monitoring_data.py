from src.monitoring import monitoring_utils
from src.utils.logging import get_logger

logger = get_logger(__name__)


def update_monitoring_data():
    """Main function to run Cuba hurricane monitoring updates."""
    # Create monitor with IMERG rainfall processing
    monitor = monitoring_utils.create_cuba_hurricane_monitor(
        rainfall_source="raster"
    )

    # Update both observational and forecast monitoring

    # lets just see of fcast runs on gha while setting up 'obsv' better

    # monitor.update_monitoring("fcast", clobber=False)
    
    # Add some diagnostic logging
    logger.info("=== DIAGNOSTIC: Checking existing data ===")
    try:
        existing_data = monitor._load_existing_monitoring("obsv")
        logger.info(f"Found existing obsv data: {len(existing_data)} records")
        if not existing_data.empty:
            logger.info(f"Existing data storms: {existing_data['atcf_id'].nunique()}")
            logger.info(f"Existing data date range: {existing_data['issue_time'].min()} to {existing_data['issue_time'].max()}")
            
            # Check if Rafael has rainfall data
            if 'al182024' in existing_data['atcf_id'].values:
                rafael_data = existing_data[existing_data['atcf_id'] == 'al182024']
                rainfall_records = rafael_data[rafael_data['rainfall_source'] != 'none']
                logger.info(f"Rafael in existing data: {len(rafael_data)} records, {len(rainfall_records)} with rainfall")
    except Exception as e:
        logger.info(f"No existing data found: {e}")
    
    logger.info("=== Starting update_monitoring ===")
    monitor.update_monitoring("obsv", clobber=False)


if __name__ == "__main__":
    update_monitoring_data()
