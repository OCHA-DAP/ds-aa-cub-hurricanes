import ocha_stratus as stratus
import pandas as pd


def load_imerg(pcode: str):
    query = f"""
    SELECT * FROM public.imerg
    WHERE pcode = '{pcode}'
    """
    df = pd.read_sql(query, stratus.get_engine("prod"))
    return df
