import os
import json
import logging
import base64
import jsonschema

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SCHEMA_FILE = '../utils/jsonl-capture-data.schema'

class DataLoader:
    """
    The DataLoader is a service that recursively searches all subdirectories of 
    the '/opt/ml/processing/input_data' directory for JSONL files and subsequently executes an
    ETL (Extract, Transform, Load) process. The DataLoader completes its job when all data has 
    been extracted, formatted, and loaded into '/opt/ml/processing/formatted_data/data.jsonl'.
    """

    def __init__(self):
        """
        Constructor. No parameters.
        
        """
        self.transformed_data = []

    def extract(self, file_path: str):
        """
        Extracts data from a JSONL file.

        :param file_path: The path to the JSONL file.
        :raises: ValueError if file_path is not a valid string.
        :returns: A list of data records extracted from the file. If file does not exist, returns empty list.
        """

        if not isinstance(file_path, str):
            raise ValueError("file_path must be a string")
        
        schema_filepath = os.path.join(os.path.dirname(__file__), SCHEMA_FILE)

        logger.info(f"Extracting data from file: {file_path}")
        extracted_data = []
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    try:
                        data = json.loads(line)
                        validate_json_against_schema(data, schema_filepath)
                    except json.JSONDecodeError:
                        logger.info(f"Invalid JSON data: {line}")
                        continue
                    except jsonschema.ValidationError as e:
                        logger.info(f"Validation error: {e}")
                        continue
                    extracted_data.append(data)
            return extracted_data
        except:
            return []
        

    def transform(self, data: list):
        """
        Applies transformation rules to the extracted data. The current rules format the data to be used with FMEval.

        :param data: A list of data records to be transformed. Each item is a dictionary.
        :raises: ValueError if data is not a list.
        :raises: Warning if invalid data is provided.
        :returns: The transformed data records.
        """
        logger.info("Transforming data...")

        if not isinstance(data, list):
            raise ValueError("data must be a list")

        transformed_data = []
        for record in data:
            try:
                content = json.loads(record["captureData"]["endpointInput"]["data"])["inputs"][0][0]["content"]
                model_output = json.loads(base64.b64decode(record["captureData"]["endpointOutput"]["data"]).decode("utf-8"))[0]["generation"]["content"]

                # Create the transformed data
                transformed_record = {
                    "content": content,
                    "answer": model_output
                }
                transformed_data.append(transformed_record)
            except (KeyError, IndexError, json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.warning(f"Error transforming record: {e}")
                continue

        return transformed_data

    def load(self, destination: str):
        """
        Loads the transformed data into a single JSONL file.
        :param destination: The destination filepath of the JSONL file.
        :raises: ValueError if destination is not a valid string.
        :returns: None.
        """

        if not isinstance(destination, str):
            raise ValueError("destination must be a string")
        
        
        logger.info(f"Loading data to: {destination}")

        # Create the directory if it doesn't exist
        formatted_data_dir = os.path.dirname(destination)
        if not os.path.exists(formatted_data_dir):
            os.makedirs(formatted_data_dir, exist_ok=True)

        # Open the file and write the data
        try:
            with open(destination, 'w') as file:
                for data_record in self.transformed_data:
                    file.write(json.dumps(data_record) + '\n')
        except PermissionError as e:

            logger.error(f"Permission error: {e}")


        
    def execute_etl(self, directory: str, destination: str):
        """
        Executes the ETL (Extract, Transform, Load) process. This function recursively searches the input data directory and performs
        ETL on all .jsonl files found.

        :param directory: The directory to search for capture data.
        :param destination: The destination filepath of the transformed data.
        :raises: ValueError if directory is not a valid string.
        :raises: ValueError if destination is not a valid string.
        :raises: Warning if invalid directory provided.
        :returns: None.
        """

        if not isinstance(directory, str):
            raise ValueError("directory must be a string")
        if not isinstance(destination, str):
            raise ValueError("destination must be a string")


        logger.info(f"current dir: {os.getcwd()}")
        logger.info(f"Executing ETL process for directory: {directory}")
        if os.path.exists(directory) and os.path.isdir(directory):
            # Iterate over each file and directory in the directory
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                if os.path.isdir(item_path):
                    # Recursively call the function for subdirectories
                    self.execute_etl(item_path, destination)
                else:
                    # Check if the file is a .jsonl file and process it
                    if item.endswith(".jsonl"):
                        logger.info(f"Processing file: {item_path}")
                        extracted_data = self.extract(item_path)
                        transformed_data = self.transform(extracted_data)
                        self.transformed_data.extend(transformed_data)
                    else:
                        logger.info(f"Found file: {item_path}")

        else:
            logger.warning(f"The directory {directory} does not exist or is not a directory.")

        # Load the transformed data into a single JSONL file
        self.load(destination)


def validate_json_against_schema(data, schema_filepath):
    """
    Validates that the data fits the schema defined in the schema file.

    :param data: The data to validate.
    :param schema_filepath: The path to the schema file.
    :raises: jsonschema.ValidationError if the data does not match the schema.
    """
    with open(schema_filepath) as sf:
            schema = json.load(sf)
            jsonschema.validate(instance=data, schema=schema)