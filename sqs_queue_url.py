import boto3

# Define the LocalStack endpoint
localstack_endpoint = 'http://localhost:4566'

# Initialize the SQS client with LocalStack endpoint
sqs = boto3.client('sqs', 
                   endpoint_url=localstack_endpoint, 
                   region_name='us-east-1',  # LocalStack uses 'us-east-1' by default
                   aws_access_key_id='fake_access_key', 
                   aws_secret_access_key='fake_secret_key')

# Get the URL for the queue you created
queue_name = 'api_req_queue'
queue_url = sqs.get_queue_url(QueueName=queue_name)['QueueUrl']

