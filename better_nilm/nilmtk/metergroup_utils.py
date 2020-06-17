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
from nilmtk.utils import normalise_timestamp

from nilmtk.elecmeter import ElecMeterID
from nilmtk.metergroup import MeterGroupID

from better_nilm.print_utils import HiddenPrints
from better_nilm.str_utils import homogenize_string


APPLIANCE_NAMES = {
    "sitemeter": "_main",
    "freezer": "fridge",
    "fridgefreezer": "fridge",
    "washerdryer": "washingmachine"
}
    

def get_good_sections(metergroup, sample_period, series_len,
                      max_series=None, step=None):
    """
    Get the good sections of a metergroup. That is, all the sections that
    meet the following requisites:
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
        with HiddenPrints():
            good_sections = meter.good_sections(ac_type="best",
                                                physical_quantity="power")
        for section in good_sections:
            delta = (section.end - section.start).total_seconds()
            if delta >= (sample_period * series_len):
                timestamps += [(v, k) for k, v in section.to_dict().items()]

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
    num_meters = len(metergroup.instance())
    
    # The frequency will be used to normalize our timestamps
    # Normalization ensures every timestamp is synchronized
    freq = '{:d}S'.format(int(sample_period))

    for stamp in timestamps:
        timestamp = stamp[0]
        event = stamp[1]
        if event == "start":
            overlap += 1
        elif event == "end":
            overlap -= 1
            # If we had a start timestamp, close that section
            if ts_start is not None:
                ts_end = normalise_timestamp(timestamp, freq)
                # Timestamp must be in UTC or it will give us trouble
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
                    if (max_series is not None) and (total_chunks > max_series):
                        exceed_chunks = total_chunks - max_series
                        chunks -= exceed_chunks
                        total_chunks = max_series
                    # Update end stamp
                    timedelta = (chunks - 1) * step + series_len
                    timedelta *= sample_period
                    ts_end = ts_start + pd.Timedelta(seconds=timedelta)
                    # Store the timestamps in dict
                    good_timestamp = {"start": ts_start,
                                      "end": ts_end}
                    good_sections += [good_timestamp.copy()]
                    # Restart the timestamp record
                    ts_start = None
        # When every meter recording overlaps, we open a new section
        if overlap == num_meters:
            ts_start = normalise_timestamp(timestamp, freq)
            # Timestamp must be in UTC or it will give us trouble
            ts_start = ts_start.tz_convert(pytz.timezone("UTC"))
        if overlap > num_meters:
            raise ValueError("More meters recording than the existing ones")
        # Stop when we reach the number of series
        if (max_series is not None) and (total_chunks >= max_series):
            break

    # Change to list of timeframes
    good_sections = list_of_timeframes_from_list_of_dicts(good_sections)
    return good_sections


def _df_preprocessing(df, metergroup):
    # Rename the columns according to their meter label
    columns = []
    for col in df.columns:
        if type(col) is ElecMeterID or type(col) is tuple:
            # If the meter is on its own, its current column name is:
            # col = ElecMeterID(instance, building, dataset)
            instance = col[0]
        elif type(col) is MeterGroupID:
            # If the meter is grouped with others, its current column name is:
            # MeterGroup(meters=(ElecMeterID(instance, building, dataset)))
            instance = col[0][0][0]
        else:
            raise ValueError(f"Unexpected type of meter ID for'{col}' "
                             f"column:\n {type(col)}")
        # We use its instance to get the appliance label
        with HiddenPrints():
            labels = metergroup.get_labels([instance])
        label = homogenize_string(labels[0])
        columns += [APPLIANCE_NAMES.get(label, label)]

    # Rename columns
    df.columns = columns

    return df


def df_from_sections(metergroup, sections, sample_period):
    """
    Extracts a dataframe from the metergroup, containing only the chosen time
    sections, with the given sample period.

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
                                            sample_period=sample_period,
                                            ac_type="best",
                                            physical_quantity="power")

    df = _df_preprocessing(df, metergroup)

    # Ensure sections are followed strictly
    for section in sections:
        ts_start = section.start
        ts_start = ts_start.tz_convert(pytz.timezone("UTC"))
        if ts_start not in df.index:
            ts_nearest = min(df.index, key=lambda x: abs(x.tz_convert(
                pytz.timezone("UTC")) - ts_start))
            raise ValueError(f"Start timestamp {ts_start} missing.\n"
                             f"Nearest is {ts_nearest}")
        ts_end = section.end - pd.Timedelta(seconds=sample_period)
        ts_end = ts_end.tz_convert(pytz.timezone("UTC"))
        if ts_end not in df.index:
            ts_nearest = min(df.index, key=lambda x: abs(x.tz_convert(
                pytz.timezone("UTC")) - ts_end))
            raise ValueError(f"End timestamp {ts_end} missing.\n"
                             f"Nearest is {ts_nearest}")
    
    return df


def df_whole(metergroup, sample_period, ffill=0):
    """
    Extracts a dataframe from the metergroup, containing only the chosen time
    sections, with the given sample period.

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

    # Load dataframe
    with HiddenPrints():
        df = metergroup.dataframe_of_meters(sample_period=sample_period,
                                            ac_type="best",
                                            physical_quantity="power")

    df = _df_preprocessing(df, metergroup)

    # Forward fill
    df.fillna(method='ffill', limit=ffill, inplace=True)

    # Fill the rest of elements
    df.fillna(0, inplace=True)

    return df
