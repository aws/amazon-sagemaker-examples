import numpy as np
import pandas as pd
from sagemaker.serializers import NumpySerializer, SimpleBaseSerializer


class ParquetSerializer(SimpleBaseSerializer):
    """Serialize data to a buffer using the .parquet format."""

    def __init__(self, content_type="application/x-parquet"):
        """Initialize a ``ParquetSerializer`` instance.

        Args:
            content_type (str): The MIME type to signal to the inference endpoint when sending
                request data (default: "application/x-parquet").
        """
        super(ParquetSerializer, self).__init__(content_type=content_type)

    def serialize(self, data):
        """Serialize data to a buffer using the .parquet format.

        Args:
            data (object): Data to be serialized. Can be a Pandas Dataframe,
                file, or buffer.

        Returns:
            io.BytesIO: A buffer containing data serialized in the .parquet format.
        """
        if isinstance(data, pd.DataFrame):
            return data.to_parquet()

        # files and buffers. Assumed to hold parquet-formatted data.
        if hasattr(data, "read"):
            return data.read()

        raise ValueError(
            f"{data} format is not supported. Please provide a DataFrame, parquet file, or buffer."
        )


class MultiModalSerializer(SimpleBaseSerializer):
    """
    Serializer for multi-modal use case.
    When passed in a dataframe, the serializer will serialize the data to be parquet format.
    When passed in a numpy array, the serializer will serialize the data to be numpy format.
    """

    def __init__(self, content_type="application/x-parquet"):
        """Initialize a ``MultiModalSerializer`` instance.

        Args:
            content_type (str): The MIME type to signal to the inference endpoint when sending
                request data (default: "application/x-parquet").
                To BE NOTICED, this content_type will not used by MultiModalSerializer
                as it doesn't support dynamic updating. Instead, we pass expected content_type to
                `initial_args` of `predict()` call to endpoints.
        """
        super(MultiModalSerializer, self).__init__(content_type=content_type)
        self.parquet_serializer = ParquetSerializer()
        self.numpy_serializer = NumpySerializer()

    def serialize(self, data):
        """Serialize data to a buffer using the .parquet format or numpy format.

        Args:
            data (object): Data to be serialized. Can be a Pandas Dataframe,
                or numpy array

        Returns:
            io.BytesIO: A buffer containing data serialized in the .parquet or .npy format.
        """
        if isinstance(data, pd.DataFrame):
            return self.parquet_serializer.serialize(data)

        if isinstance(data, np.ndarray):
            return self.numpy_serializer.serialize(data)

        raise ValueError(
            f"{data} format is not supported. Please provide a DataFrame, or numpy array."
        )


class JsonLineSerializer(SimpleBaseSerializer):
    """Serialize data to a buffer using the .jsonl format."""

    def __init__(self, content_type="application/jsonl"):
        """Initialize a ``JsonLineSerializer`` instance.

        Args:
            content_type (str): The MIME type to signal to the inference endpoint when sending
                request data (default: "application/jsonl").
        """
        super(JsonLineSerializer, self).__init__(content_type=content_type)

    def serialize(self, data):
        """Serialize data to a buffer using the .jsonl format.

        Args:
            data (pd.DataFrame): Data to be serialized.

        Returns:
            io.StringIO: A buffer containing data serialized in the .jsonl format.
        """
        if isinstance(data, pd.DataFrame):
            return data.to_json(orient="records", lines=True)

        raise ValueError(f"{data} format is not supported. Please provide a DataFrame.")
