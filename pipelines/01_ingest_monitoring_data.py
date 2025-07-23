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

    monitor.update_monitoring("fcast", clobber=False)
    monitor.update_monitoring("obsv", clobber=False)


if __name__ == "__main__":
    update_monitoring_data()
