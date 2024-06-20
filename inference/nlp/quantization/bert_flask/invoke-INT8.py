import boto3
import json
import time

client = boto3.client('sagemaker-runtime')
context="The Panthers finished the regular season with a 15-1 record, and quarterback Cam Newton was named the NFL Most Valuable Player (MVP). They defeated the Arizona Cardinals 49-15 in the NFC Championship Game and advanced to their second Super Bowl appearance since the franchise was founded in 1995. The Broncos finished the regular season with a 12-4 record, and denied the New England Patriots a chance to defend their title from Super Bowl XLIX by defeating them 20-18 in the AFC Championship Game. They joined the Patriots, Dallas Cowboys, and Pittsburgh Steelers as one of four teams that have made eight appearances in the Super Bowl."

question="Who was named the MVP?"
###-----------------------------------------------------------------------------
#question = "Who did Broncos denied a chance to defend their title?"
#question = "When was the franchise founded?"
#question = "In which championship did Panthers defeated the Arizona Cardinals?"
###-----------------------------------------------------------------------------

custom_attributes = "c000b4f9-df62-4c85-a0bf-7c525f9104a4"  # An example of a trace ID.
endpoint_name = "c6ibothbert"                               # Your endpoint name.
content_type = "application/json"                           # The MIME type of the input data in the request body.
accept = "*/*"                                              # The desired MIME type of the inference in the response.

start_time = time.time()
response = client.invoke_endpoint(
    EndpointName=endpoint_name, 
    CustomAttributes=custom_attributes, 
    ContentType=content_type,
    Accept=accept,
    Body=json.dumps({'context': '', 'question': question, 'version': 'INT8'})
    )
print("--- %s seconds ---" % (time.time() - start_time))
my_json = json.loads(response['Body'].read())
print(my_json)                         
# If model receives and updates the custom_attributes header 
# by adding "Trace id: " in front of custom_attributes in the request,
# custom_attributes in response becomes
# "Trace ID: c000b4f9-df62-4c85-a0bf-7c525f9104a4"
 
 