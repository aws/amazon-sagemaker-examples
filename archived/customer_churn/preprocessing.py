import argparse
import os
import subprocess
import sys
import time
import warnings

import pandas as pd

subprocess.check_call([sys.executable, "-m", "pip", "install", "awswrangler"])
import awswrangler as wr

start_time = time.time()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dw-output-path")
    parser.add_argument("--processing-output-filename")

    args, _ = parser.parse_known_args()
    print("Received arguments {}".format(args))

    data_s3_uri = args.dw_output_path
    output_filename = args.processing_output_filename

    #     data_path = os.path.join('/opt/ml/processing/input', dw_output_name)
    #     df = pd.read_csv(data_path)
    df = wr.s3.read_csv(path=data_s3_uri, dataset=True)
    ## convert to time
    df["date"] = pd.to_datetime(df["ts"], unit="ms")
    df["ts_dow"] = df["date"].dt.weekday
    df["ts_date_day"] = df["date"].dt.date
    df["ts_is_weekday"] = [1 if x in [0, 1, 2, 3, 4] else 0 for x in df["ts_dow"]]
    df["registration_ts"] = pd.to_datetime(df["registration"], unit="ms").dt.date
    ## add labels
    df["churned_event"] = [1 if x == "Cancellation Confirmation" else 0 for x in df["page"]]
    df["user_churned"] = df.groupby("userId")["churned_event"].transform("max")

    ## convert pages categorical variables to numerical
    events_list = [
        "NextSong",
        "Thumbs Down",
        "Thumbs Up",
        "Add to Playlist",
        "Roll Advert",
        "Add Friend",
        "Downgrade",
        "Upgrade",
        "Error",
    ]
    usage_column_name = []
    for event in events_list:
        event_name = "_".join(event.split()).lower()
        usage_column_name.append(event_name)
        df[event_name] = [1 if x == event else 0 for x in df["page"]]
    ## feature engineering
    # average_events_weekday (numerical): average number of events per day during weekday
    # average_events_weekend (numerical): average number of events per day during the weekend
    base_df = (
        df.groupby(["userId", "ts_date_day", "ts_is_weekday"])
        .agg({"page": "count"})
        .groupby(["userId", "ts_is_weekday"])["page"]
        .mean()
        .unstack(fill_value=0)
        .reset_index()
        .rename(columns={0: "average_events_weekend", 1: "average_events_weekday"})
    )

    # num_ads_7d, num_songs_played_7d, num_songs_played_30d, num_songs_played_90d, num_ads_7d, num_error_7d
    base_df_daily = (
        df.groupby(["userId", "ts_date_day"])
        .agg({"page": "count", "nextsong": "sum", "roll_advert": "sum", "error": "sum"})
        .reset_index()
    )
    feature34 = (
        base_df_daily.groupby(["userId", "ts_date_day"])
        .tail(7)
        .groupby(["userId"])
        .agg({"nextsong": "sum", "roll_advert": "sum", "error": "sum"})
        .reset_index()
        .rename(
            columns={
                "nextsong": "num_songs_played_7d",
                "roll_advert": "num_ads_7d",
                "error": "num_error_7d",
            }
        )
    )
    feature5 = (
        base_df_daily.groupby(["userId", "ts_date_day"])
        .tail(30)
        .groupby(["userId"])
        .agg({"nextsong": "sum"})
        .reset_index()
        .rename(columns={"nextsong": "num_songs_played_30d"})
    )
    feature6 = (
        base_df_daily.groupby(["userId", "ts_date_day"])
        .tail(90)
        .groupby(["userId"])
        .agg({"nextsong": "sum"})
        .reset_index()
        .rename(columns={"nextsong": "num_songs_played_90d"})
    )
    # num_artists, num_songs, num_ads, num_thumbsup, num_thumbsdown, num_playlist, num_addfriend, num_error, user_downgrade,
    # user_upgrade, percentage_ad, days_since_active
    base_df_user = (
        df.groupby(["userId"])
        .agg(
            {
                "page": "count",
                "nextsong": "sum",
                "artist": "nunique",
                "song": "nunique",
                "thumbs_down": "sum",
                "thumbs_up": "sum",
                "add_to_playlist": "sum",
                "roll_advert": "sum",
                "add_friend": "sum",
                "downgrade": "max",
                "upgrade": "max",
                "error": "sum",
                "ts_date_day": "max",
                "registration_ts": "min",
                "user_churned": "max",
            }
        )
        .reset_index()
    )
    base_df_user["percentage_ad"] = base_df_user["roll_advert"] / base_df_user["page"]
    base_df_user["days_since_active"] = (
        base_df_user["ts_date_day"] - base_df_user["registration_ts"]
    ).dt.days
    # repeats ratio
    base_df_user["repeats_ratio"] = 1 - base_df_user["song"] / base_df_user["nextsong"]

    # num_sessions, avg_time_per_session, avg_events_per_session,
    base_df_session = (
        df.groupby(["userId", "sessionId"])
        .agg({"length": "sum", "page": "count", "date": "min"})
        .reset_index()
    )
    base_df_session["prev_session_ts"] = base_df_session.groupby(["userId"])["date"].shift(1)
    base_df_session["gap_session"] = (
        base_df_session["date"] - base_df_session["prev_session_ts"]
    ).dt.days
    user_sessions = (
        base_df_session.groupby("userId")
        .agg({"sessionId": "count", "length": "mean", "page": "mean", "gap_session": "mean"})
        .reset_index()
        .rename(
            columns={
                "sessionId": "num_sessions",
                "length": "avg_time_per_session",
                "page": "avg_events_per_session",
                "gap_session": "avg_gap_between_session",
            }
        )
    )

    # merge features together
    base_df["userId"] = base_df["userId"].astype("int")
    final_feature_df = base_df.merge(feature34, how="left", on="userId")
    final_feature_df = final_feature_df.merge(feature5, how="left", on="userId")
    final_feature_df = final_feature_df.merge(feature6, how="left", on="userId")
    final_feature_df = final_feature_df.merge(user_sessions, how="left", on="userId")
    final_feature_df = final_feature_df.merge(base_df_user, how="left", on="userId")

    final_feature_df = final_feature_df.fillna(0)
    # renaming columns
    final_feature_df.columns = [
        "userId",
        "average_events_weekend",
        "average_events_weekday",
        "num_songs_played_7d",
        "num_ads_7d",
        "num_error_7d",
        "num_songs_played_30d",
        "num_songs_played_90d",
        "num_sessions",
        "avg_time_per_session",
        "avg_events_per_session",
        "avg_gap_between_session",
        "num_events",
        "num_songs",
        "num_artists",
        "num_unique_songs",
        "num_thumbs_down",
        "num_thumbs_up",
        "num_add_to_playlist",
        "num_ads",
        "num_add_friend",
        "num_downgrade",
        "num_upgrade",
        "num_error",
        "ts_date_day",
        "registration_ts",
        "user_churned",
        "percentage_ad",
        "days_since_active",
        "repeats_ratio",
    ]
    # only keep created feature columns
    final_feature_df = final_feature_df[
        [
            "userId",
            "user_churned",
            "average_events_weekend",
            "average_events_weekday",
            "num_songs_played_7d",
            "num_ads_7d",
            "num_error_7d",
            "num_songs_played_30d",
            "num_songs_played_90d",
            "num_sessions",
            "avg_time_per_session",
            "avg_events_per_session",
            "avg_gap_between_session",
            "num_events",
            "num_songs",
            "num_artists",
            "num_thumbs_down",
            "num_thumbs_up",
            "num_add_to_playlist",
            "num_ads",
            "num_add_friend",
            "num_downgrade",
            "num_upgrade",
            "num_error",
            "percentage_ad",
            "days_since_active",
            "repeats_ratio",
        ]
    ]

    print("shape of file to append:\t\t{}".format(final_feature_df.shape))
    iter_end_time = time.time()
    end_time = time.time()
    print("minutes elapsed: {}".format(str((end_time - start_time) / 60)))

    final_features_output_path = os.path.join("/opt/ml/processing/output", output_filename)
    print("Saving processed data to {}".format(final_features_output_path))
    final_feature_df.to_csv(final_features_output_path, header=True, index=False)
