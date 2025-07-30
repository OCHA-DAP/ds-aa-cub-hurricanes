from src.monitoring import monitoring_utils
from src.utils.logging import get_logger
from src.email import plotting, update_emails

logger = get_logger(__name__)


def update_observational_data():
    """Main function to run Cuba hurricane monitoring updates."""
    # Create monitor object.
    # raster is for IMERG data processing- but is ignored
    # for the forecast step.
    monitor = monitoring_utils.create_cuba_hurricane_monitor(
        rainfall_source="raster"
    )
    monitor.update_monitoring("obsv", clobber=False)


if __name__ == "__main__":
        logger.info("🚀 Starting Cuba Hurricane Observational Pipeline")
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
    update_observational_data()
    update_emails.update_obsv_trigger_emails()
    plotting.update_plots(fcast_obsv="obsv")
    update_emails.update_obsv_info_emails()