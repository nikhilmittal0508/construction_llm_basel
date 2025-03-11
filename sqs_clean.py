import boto3

# Connect to LocalStack SQS
sqs = boto3.client(
    'sqs',
    endpoint_url='http://localhost:4566',  # LocalStack URL
    region_name='us-east-1'
)

# Queue URL
# queue_url = 'http://localhost:4566/000000000000/your-queue-name'

queue_url = "http://sqs.us-east-1.localhost.localstack.cloud:4566/000000000000/api_req_queue"

# Purge the queue
response = sqs.purge_queue(QueueUrl=queue_url)

print("Queue successfully emptied!")
