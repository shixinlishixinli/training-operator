from dataclasses import dataclass, field
from urllib.parse import urlparse
from .abstract_model_provider import modelProvider
from .abstract_dataset_provider import datasetProvider
from minio import Minio
#from minio.error import S3Error

@dataclass
class MinioDatasetPrams:
    endpoint_url: str
    bucket_name: str
    access_key: str 
    secret_key: str 

    def is_valid_url(self, url):
        try:
            parsed_url = urlparse(url)
            print(parsed_url)
            return all([parsed_url.scheme, parsed_url.netloc])
        except ValueError:
            return False

    def __post_init__(self):
        # Custom checks or validations can be added here
        if (
            self.bucket_name is None
            or self.endpoint_url is None
        ):
            raise ValueError("bucket_name or endpoint_url or file_key is None")
        self.is_valid_url(self.endpoint_url)
        print ('this is MinioDatasetPrams')

class MinioDataset(datasetProvider):
    def load_config(self, serialised_args):
        
        print("minio dataset load_config")
    def download_dataset(self):
        # Create Minio client
        # HACK: Mount volume for test
        print ("this is miniodataset")
        minio_client = Minio(self.minio_endpoint, self.minio_accesskey, self.minio_secretkey)
        object_name = "git-base/pytorch_model.bin"
        #file_path = "/home/jovyan/model/pytorch_model.bin"
        file_path = "./pytorch_model_minio.bin"
        with open(file_path, "wb") as file_data:
            file_data.write(minio_client.get_object("my-bucket", object_name).read())

        print("Object downloaded successfully")


@dataclass
class MinioModelParams:
    model_uri: str
    #transformer_type: TRANSFORMER_TYPES
    access_token: str = None

    def __post_init__(self):
        # Custom checks or validations can be added here
        if self.model_uri == "" or self.model_uri is None:
            raise ValueError("model_uri cannot be empty.")

class MinioModel(modelProvider):
    def load_config(self, serialised_args):
        # implementation for loading the config
        print("minio model load_config")

    def download_model_and_tokenizer(self):
        # implementation for downloading the model
        print("downloading minio model")

@dataclass
class LLMTrainParams:
    num_per_core: str
