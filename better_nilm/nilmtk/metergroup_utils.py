"""
Attributes
----------
APPLIANCE_NAMES : dict
    {original name: new name}
    Used to change the name of appliances contained in the MeterGroup
"""

import pandas as pd
import pytz

from math import floor
from nilmtk.metergroup import MeterGroup
from nilmtk.timeframe import list_of_timeframes_from_list_of_dicts

from better_nilm.print_utils import HiddenPrints
from better_nilm.str_utils import homogenize_string


APPLIANCE_NAMES = {
    "sitemeter": "_main",
    "freezer": "fridge",
    "fridgefreezer": "fridge"
}


def get_good_sections(metergroup, sample_period, series_len,
                      max_series=None, step=None):
    """
    Get the good sections of a metergroup. That is, all the sections that
    meet the following requesites:
        - All its meters have been recording during that section.
        - The section contains enough consecutive records to fill at least
        one serie (defined by the sample_period and series_len).

    Params
    ------
    metergroup : nilmtk.metergroup.Metergroup
        List of electric meters, including the main meters of the house
    sample_period : int
        Time between consecutive electric load records, in seconds.
    series_len : int
        Number of consecutive records to take at once.
    max_series : int, default=None
        Maximum number of series to output.
    step : int, default=None
        Steps between serie origins. By default it is None, which makes
        that step = series_len (one series starts right after another
        ends, without any overlapping).

    Returns
    -------
    good_sections : list
        List of timeframes
    """
    assert type(metergroup) is MeterGroup, f"metergroup param must be type " \
                                           f"nilmtk.metergroup.MeterGroup\n" \
                                           f"Input param is type " \
                                           f"{type(metergroup)}"
    if step is None:
        step = series_len

    timestamps = []

    # Get the good sections of each meter
    for meter in metergroup.all_meters():
        # Take sections with enough size
        for section in meter.good_sections():
            delta = (section.end - section.start).total_seconds()
            if delta >= (sample_period * series_len):
                timestamps += [(v, k) for k, v in section.to_dict().items()]

    # Store a timestamp of the first meter. This timestamp will be our
    # reference to keep synchronized all the other timestamps
    ts_main = pd.Timestamp(timestamps[0][0])

    # Count the number of chunks available for the house
    total_chunks = 0
    # Sort timestamps by date
    timestamps = sorted(timestamps)
    # Initialize the list of timestamps and sections
    good_sections = []
    ts_start = None
    # We will be counting the overlapping sections. When overlapping equals
    # the number of meters, we will have a good section for every meter
    overlap = 0

    for stamp in timestamps:
        if stamp[1] == "start":
            overlap += 1
        else:
            overlap -= 1
            # If we had a start timestamp, close that section
            if ts_start is not None:
                # Timestamp must be in UTC or it will give us trouble
                ts_end = pd.Timestamp(stamp[0])
                ts_end = ts_end.tz_convert(pytz.timezone("UTC"))
                # Check that the sections allows to take at least
                # one data chunk
                timedelta = (ts_end - ts_start).total_seconds()
                dt = floor(timedelta / sample_period)
                chunks = floor((dt - series_len) / step) + 1
                # Take exactly the chunk size we need
                if chunks > 0:
                    # Add chunks to total
                    total_chunks += chunks
                    # Ensure we do not exceed the allowed limit
                    if (max_series is not None) and (total_chunks >
                                                      max_series):
                        exceed_chunks = total_chunks - max_series
                        chunks -= exceed_chunks
                        total_chunks = max_series
                    # Update end stamp
                    timedelta = ((chunks - 1) * step + series_len - 1) * \
                                sample_period
                    ts_end = ts_start + pd.Timedelta(seconds=timedelta)
                    good_timestamp = {"start": ts_start,
                                      "end": ts_end}
                    good_sections += [good_timestamp]
                    ts_start = None
        # When every meter overlaps, we open a new section
        if overlap == len(metergroup.instance()):
            ts_start = pd.Timestamp(stamp[0])
            # The timestamp should also be synchronized with the main meter
            # stimestamps
            timedelta = (ts_start - ts_main).total_seconds()
            timedelta = sample_period * (timedelta // sample_period)
            ts_start = ts_main + pd.Timedelta(seconds=timedelta)
            # Timestamp must be in UTC or it will give us trouble
            ts_start = ts_start.tz_convert(pytz.timezone("UTC"))
        # Stop when we reach the number of series
        if (max_series is not None) and (total_chunks >= max_series):
            break

    # The last section must have one more record, else it will be omitted by
    # the nilmtk extract function
    good_sections[-1]["end"] += pd.Timedelta(seconds=sample_period)

    # Change to list of timeframes
    good_sections = list_of_timeframes_from_list_of_dicts(good_sections)
    return good_sections


def df_from_sections(metergroup, sections, sample_period):
    """
    Params
    ------
    metergroup : nilmtk.metergroup.Metergroup
        List of electric meters, including the main meters of the house
    sections : list
        List of timeframes
    sample_period : int, default=6
        Time between consecutive electric load records, in seconds.
        By default we take 6 seconds.

    Returns
    -------
    df : pandas.DataFrame
        Aggregated load values of each meter within the given sections.
    """
    if type(sections) is not list:
        raise TypeError("good_sections must be of type list.\n"
                        f"Current type is: {type(sections)}")

    # Load dataframe composed by the good sections
    with HiddenPrints():
        df = metergroup.dataframe_of_meters(sections=sections,
                                            sample_period=sample_period)

    # Rename the columns according to their appliances
    columns = []
    for col in df.columns:
        try:
            app = metergroup.get_labels([col[0]])[0]
        except Exception:
            try:
                with HiddenPrints():
                    app = metergroup.get_labels([col[0][0][0]])[0].lower()
            except Exception:
                raise ValueError(f"Something went wrong when trying to name "
                                 f"'{col}' column")
        app = homogenize_string(app)
        columns += [APPLIANCE_NAMES.get(app, app)]

    # Rename columns
    df.columns = columns

    return df
