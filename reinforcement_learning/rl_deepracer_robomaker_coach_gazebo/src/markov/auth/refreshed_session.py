""" This Module implements a session factory with automatic session renewal.
    This is needed because the assumed session credentials by default
    expire after 1 hour. However, we need the session to last longer without
    interrupting the current flow.
    Reference:
    https://dev.to/li_chastina/auto-refresh-aws-tokens-using-iam-role-and-boto3-2cjf
    https://github.com/cloud-custodian/cloud-custodian/blob/master/c7n/credentials.py
"""
from datetime import datetime, timedelta

import pytz
from boto3 import Session
from botocore.credentials import RefreshableCredentials
from botocore.session import get_session


def refreshed_session(region="us-east-1"):
    """Assume a boto3.Session With automatic credential renewal.
       NOTE: We have to poke at botocore internals a few times.

    Args:
        region (str, optional): The region of the session.
                                Defaults to 'us-east-1'.

    Returns:
        session (Session): an boto3 session with RefreshableCredentials
    """

    def _refresh():
        credentials = Session().get_credentials()
        # set the expiry time to one hour from now.
        # Note this is not the real session expiry time,
        # but we are focusing refresh every 1 hour.
        return dict(
            access_key=credentials.access_key,
            secret_key=credentials.secret_key,
            token=credentials.token,
            expiry_time=(pytz.utc.localize(datetime.utcnow()) + timedelta(hours=1)).isoformat(),
        )

    session_credentials = RefreshableCredentials.create_from_metadata(
        metadata=_refresh(), refresh_using=_refresh, method="session-cred"
    )

    re_session = get_session()
    re_session._credentials = session_credentials
    re_session.set_config_variable("region", region)
    return Session(botocore_session=re_session)
