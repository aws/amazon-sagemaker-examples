import json
from typing import Dict

import boto3


SERVICE_CODE = "AmazonSageMaker"
PRODUCT_FAMILY = "ML Instance"
PRODUCT_FAMILY_KEY = "productFamily"
PRICING_SERVICE_API_REGION = "us-east-1"  # All pricing APIs are hosted in IAD
REGION_KEY = "regionCode"
INSTANCE_NAME_KEY = "instanceName"
PLATO_INSTANCE_TYPE_KEY = "platoinstancetype"
PLATO_INSTANCE_TYPE = "Hosting"


def _create_pricing_filter(type: str, field: str, value: str) -> Dict[str, str]:
    return {"Type": type, "Field": field, "Value": value}


class PricingClient:
    """Boto3 client to access AWS Pricing."""

    def __init__(self) -> None:
        """Creates the boto3 client for AWS pricing."""
        self._client = boto3.client(service_name="pricing", region_name=PRICING_SERVICE_API_REGION)

    def get_price_per_unit(self, instance_type: str, region: str) -> float:
        """Returns the price per unit in USD of a SageMaker machine learning instance in a region."""
        filters = [
            _create_pricing_filter(type="TERM_MATCH", field=PRODUCT_FAMILY_KEY, value=PRODUCT_FAMILY),
            _create_pricing_filter(type="TERM_MATCH", field=REGION_KEY, value=region),
            _create_pricing_filter(type="TERM_MATCH", field=INSTANCE_NAME_KEY, value=instance_type),
            _create_pricing_filter(type="TERM_MATCH", field=PLATO_INSTANCE_TYPE_KEY, value=PLATO_INSTANCE_TYPE),
        ]
        response = self._client.get_products(ServiceCode=SERVICE_CODE, Filters=filters)
        price_list = json.loads(response["PriceList"][0])["terms"]["OnDemand"]
        price_dimensions = list(price_list.values())[0]["priceDimensions"]
        price_per_unit = list(price_dimensions.values())[0]["pricePerUnit"]["USD"]
        return float(price_per_unit)
