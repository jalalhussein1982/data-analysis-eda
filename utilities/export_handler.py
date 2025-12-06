"""Export handlers for data and configuration."""

import pandas as pd
import json
from typing import Optional
from io import BytesIO, StringIO


def export_dataframe(
    df: pd.DataFrame,
    format: str = "csv",
    filename: Optional[str] = None
) -> tuple:
    """
    Export DataFrame to various formats.

    Args:
        df: DataFrame to export
        format: Export format ('csv', 'xlsx', 'parquet', 'json')
        filename: Optional filename (without extension)

    Returns:
        Tuple of (data_bytes, filename_with_extension, mime_type)
    """
    if filename is None:
        filename = "exported_data"

    if format == "csv":
        buffer = StringIO()
        df.to_csv(buffer, index=False)
        data = buffer.getvalue().encode('utf-8')
        return data, f"{filename}.csv", "text/csv"

    elif format == "xlsx":
        buffer = BytesIO()
        df.to_excel(buffer, index=False, engine='openpyxl')
        data = buffer.getvalue()
        return data, f"{filename}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    elif format == "parquet":
        buffer = BytesIO()
        df.to_parquet(buffer, index=False)
        data = buffer.getvalue()
        return data, f"{filename}.parquet", "application/octet-stream"

    elif format == "json":
        data = df.to_json(orient='records', indent=2).encode('utf-8')
        return data, f"{filename}.json", "application/json"

    else:
        raise ValueError(f"Unsupported format: {format}")


def export_config(config: dict, filename: Optional[str] = None) -> tuple:
    """
    Export pipeline configuration to JSON.

    Args:
        config: Configuration dictionary
        filename: Optional filename (without extension)

    Returns:
        Tuple of (data_bytes, filename_with_extension, mime_type)
    """
    if filename is None:
        filename = "pipeline_config"

    data = json.dumps(config, indent=2).encode('utf-8')
    return data, f"{filename}.json", "application/json"


def export_statistics(
    stats_data: list,
    format: str = "csv",
    filename: Optional[str] = None
) -> tuple:
    """
    Export distribution statistics.

    Args:
        stats_data: List of dictionaries with statistics
        format: Export format ('csv', 'json')
        filename: Optional filename (without extension)

    Returns:
        Tuple of (data_bytes, filename_with_extension, mime_type)
    """
    if filename is None:
        filename = "distribution_stats"

    df = pd.DataFrame(stats_data)

    if format == "csv":
        buffer = StringIO()
        df.to_csv(buffer, index=False)
        data = buffer.getvalue().encode('utf-8')
        return data, f"{filename}.csv", "text/csv"

    elif format == "json":
        data = json.dumps(stats_data, indent=2).encode('utf-8')
        return data, f"{filename}.json", "application/json"

    else:
        raise ValueError(f"Unsupported format: {format}")
