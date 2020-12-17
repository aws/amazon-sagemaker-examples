'''
Utility file to help with s3 operations.
'''
from urllib.parse import urlparse
import boto3

from typing import NamedTuple
from typing import Callable

from io import BytesIO, StringIO, TextIOWrapper

s3r = boto3.resource('s3')
s3 = boto3.client('s3')

class S3Ref(NamedTuple):
    """
     Typed tuple class to store reference to a s3 bucket and key.
    """
    bucket:str
    key:str

    @classmethod
    def from_uri(cls, s3_uri:str):
        s3_path = urlparse(s3_uri, allow_fragments=False)
        return cls(s3_path.netloc, s3_path.path[1:])

    def get_uri(self) -> str:
        return "s3://{}/{}".format(self.bucket, self.key)

def create_ref_at_parent_key(s3_ref:S3Ref, filename:str) -> S3Ref:
    """
     Create a S3Ref at the same path as the parent key
    """
    key_paths = s3_ref.key.split("/")
    key_paths[-1] = filename
    return S3Ref(s3_ref.bucket, "/".join(key_paths))

def get_content_size(s3_ref:S3Ref) -> int:
    """
      Get the file size in bytes.
    """
    response = s3.head_object(Bucket=s3_ref.bucket, Key=s3_ref.key)
    return int(response['ContentLength'])

def copy(source:S3Ref, dest:S3Ref) -> None:
    """
      Copy S3 file.
    """
    copy_source = {
        'Bucket': source.bucket,
        'Key': source.key
    }
    dest_bucket = s3r.Bucket(dest.bucket)
    dest_bucket.copy(copy_source, dest.key)


def download(source:S3Ref) -> StringIO:
    """
     Downloads a file to a string stream.
    """
    bytestream = BytesIO()
    s3.download_fileobj(source.bucket, source.key, bytestream)
    bytestream.seek(0)
    return TextIOWrapper(bytestream, encoding='utf-8')

def upload(memoryfile:StringIO, dest:S3Ref) -> None:
    """
     Upload file from local storage to s3.
    """
    s3.upload_fileobj(BytesIO(memoryfile.getvalue().encode()), dest.bucket, dest.key)

def get_count_with_query(source:S3Ref, query:str) -> int:
    """
     Run a s3_select query and return the resulting count.
    """
    event_stream = s3.select_object_content(
        Bucket=source.bucket,
        Key=source.key,
        ExpressionType='SQL',
        Expression=query,
        InputSerialization = {"JSON": {"Type": "LINES"}},
        OutputSerialization = {"CSV": {}}
    )

    count = 0
    for s3_select_event in event_stream["Payload"]:
        if 'Records' in s3_select_event:
            count = int(s3_select_event['Records']['Payload'])
            break

    return count

def query_helper(source:S3Ref, query:str, dest:S3Ref=None,
                 transform:Callable=None) -> StringIO:
    """
    query_helper runs the given s3_select query on the given object.
     - The results are saved in a in memory file (StringIO) and returned.
     - If dest is specified, the file is copied to the provided S3Ref
     - If transform callable is specified, tranform is called first with the
        temp file before uploading to the destination s3.
    """

    event_stream = s3.select_object_content(
        Bucket=source.bucket,
        Key=source.key,
        ExpressionType='SQL',
        Expression=query,
        InputSerialization = {"JSON": {"Type": "LINES"}},
        OutputSerialization = {"JSON": {}}
    )

    # Iterate over events in the event stream as they come
    output = StringIO()
    for s3_select_event in event_stream["Payload"]:
        if 'Records' in s3_select_event:
            data = s3_select_event['Records']['Payload']
            output.write(data.decode('utf-8'))

    if transform:
        output.seek(0)
        output = transform(output)
    if dest is not None:
        upload(output, dest)
    output.seek(0)
    return output

def download_with_query(source:S3Ref, query:str) -> StringIO:
    """
     download only the contents in source which match the query
    """
    return query_helper(source, query)

def copy_with_query(source:S3Ref, dest:S3Ref, query:str) -> StringIO:
    """
     copy the contents in source which match the query to the given destination.
    """
    return query_helper(source, query, dest)

def copy_with_query_and_transform(source:S3Ref,
            dest:S3Ref,
            query:str,
            transform:Callable) -> StringIO:
    """
     copy the contents in source which match the query to the given destination
     after transforming the local file by calling a transform callable.
    """
    return query_helper(source, query, dest, transform)
