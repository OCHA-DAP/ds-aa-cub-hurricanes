from src.monitoring import monitoring_utils
from src.utils.logging import get_logger

logger = get_logger(__name__)


def update_forecast_data():
    """Main function to run Cuba hurricane monitoring updates."""
    # Create monitor with IMERG rainfall processing
    monitor = monitoring_utils.create_cuba_hurricane_monitor(
        rainfall_source="raster"
    )

    # Update observational data
    monitor.update_monitoring("fcast", clobber=False)


if __name__ == "__main__":
    update_forecast_data()
