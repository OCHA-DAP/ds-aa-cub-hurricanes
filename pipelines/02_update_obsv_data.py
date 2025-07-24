from src.monitoring import monitoring_utils
from src.utils.logging import get_logger

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
    update_observational_data()
