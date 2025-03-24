import boto3
import json

config_file_path = 'params.json'

with open(config_file_path, 'r') as config_file:
    config_data = json.load(config_file)
    config = config_data.get('config', {})

# Define the LocalStack endpoint
localstack_endpoint = 'http://localhost:4566'

aws_sqs_endpoint = 'https://sqs.eu-north-1.amazonaws.com/851725454860/'

# Initialize the SQS client with LocalStack endpoint
sqs = boto3.client('sqs', 
                   endpoint_url=aws_sqs_endpoint, 
                   region_name=config["aws_region"],  # LocalStack uses 'us-east-1' by default
                   aws_access_key_id=config["aws_access_key"], 
                   aws_secret_access_key=config["aws_secret_access_key"])

# Get the URL for the queue you created
# queue_name = 'api_req_queue'
queue_name = 'api_req_queue.fifo'
queue_url = sqs.get_queue_url(QueueName=queue_name)['QueueUrl']

print("queue_url: ",queue_url)

