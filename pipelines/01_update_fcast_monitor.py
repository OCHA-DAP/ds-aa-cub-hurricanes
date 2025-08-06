from src.monitoring import monitoring_utils
from src.utils.logging import get_logger
from src.email import plotting, update_emails
from src.constants import DRY_RUN, TEST_EMAIL, FORCE_ALERT

logger = get_logger(__name__)


def update_forecast_data():
    """Main function to run Cuba hurricane monitoring updates."""
    # Create monitor with IMERG rainfall processing
    monitor = monitoring_utils.create_cuba_hurricane_monitor(
        rainfall_source="raster"
    )

    # Update forecast data
    monitor.update_monitoring("fcast", clobber=False)


if __name__ == "__main__":
    # Log critical environment variable configuration
    logger.info("🚀 Starting Cuba Hurricane Forecast Pipeline")
    logger.info("📋 Environment Configuration:")
    logger.info(
        f"   DRY_RUN: {DRY_RUN} "
        f"({'Will simulate' if DRY_RUN else 'Will execute'} email operations)"
    )
    logger.info(
        f"   TEST_EMAIL: {TEST_EMAIL} "
        f"({'Test mode' if TEST_EMAIL else 'Production mode'} - "
        f"affects email records and recipients)"
    )
    logger.info(
        f"   FORCE_ALERT: {FORCE_ALERT} "
        f"({'Test alerts enabled' if FORCE_ALERT else 'Real alerts only'} - "
        f"adds PRUEBA prefix)"
    )
    logger.info("=" * 80)

    update_forecast_data()
    update_emails.update_fcast_trigger_emails()
    plotting.update_plots(fcast_obsv="fcast")
    update_emails.update_fcast_info_emails()
