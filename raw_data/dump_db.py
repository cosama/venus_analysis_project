#!/bin/env python3

import argparse
import sqlite3

import pandas as pd
import pyarrow as pa


def extract_table_names(conn: sqlite3.Connection) -> list[str]:
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names = list(map(lambda x: x[0], c.fetchall()))
    c.close()
    return table_names


if "__main__" == __name__:
    parser = argparse.ArgumentParser(
        description="Convert a database to another selected file type."
    )

    parser.add_argument(
        "--database", help="Relative path of the database file", required=True, type=str
    )
    parser.add_argument(
        "--output", help="Name of the ouput file", required=True, type=str
    )
    parser.add_argument(
        "--filetype",
        help="Filetype to convert to.",
        choices=["pq", "csv", "h5"],
        required=True,
        type=str,
    )

    args = parser.parse_args()

    with sqlite3.connect(args.database) as conn:
        for table in extract_table_names(conn):
            dataframe = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            if "csv" == args.filetype:
                dataframe.to_csv(f"{args.output}_{table}.csv")

            elif "h5" == args.filetype:
                dataframe.to_hdf(f"{args.output}.h5", table)

            elif "pq" == args.filetype:
                dataframe.to_parquet(f"{args.output}_{table}.pq")
