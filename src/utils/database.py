from sqlalchemy import (
    REAL,
    TEXT,
    Column,
    Date,
    MetaData,
    Table,
    UniqueConstraint,
)


def create_chirps_gefs_table(table_name, engine):
    """
    Create a table for storing flood exposure data in the database.

    Parameters
    ----------
    table_name : str
        The name of the dataset for which the table is being created.
    engine : sqlalchemy.engine.Engine
        The SQLAlchemy engine object used to connect to the database.

    Returns
    -------
    None
    """

    metadata = MetaData()
    columns = [
        Column("valid_date", Date),
        Column("issued_date", Date),
        Column("variable", TEXT),
        Column("value", REAL),
    ]

    unique_constraint_columns = ["valid_date", "issued_date", "variable"]

    Table(
        f"{table_name}",
        metadata,
        *columns,
        UniqueConstraint(
            *unique_constraint_columns,
            name=f"{table_name}_unique",
            postgresql_nulls_not_distinct=True,
        ),
        schema="projects",
    )

    metadata.create_all(engine)
    return
