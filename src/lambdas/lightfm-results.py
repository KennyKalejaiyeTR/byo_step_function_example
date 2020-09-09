import json
import boto3
import csv
import time
from jinja2 import Template

sm_client = boto3.client("sagemaker")
s3_client = boto3.client("s3")
ses_client = boto3.client("ses")

with open("template/template.html", "r") as f:
    template = Template(f.read())

def get_experiment_data(exp_name, trial_name=None):
    
    rows = []
    header = {}
    
    while True: # retry in case of throttling
        try:
            trials = sm_client.list_trials(ExperimentName=exp_name)['TrialSummaries']
            break
        except sm_client.exceptions.ClientError:
            time.sleep(1)

    for trial in trials:
        results = {}
        trial_row = {"TrialName": trial["TrialName"]}
        
        while True: # retry in case of throttling
            try:
                components = sm_client.list_trial_components(TrialName=trial["TrialName"])["TrialComponentSummaries"]
                break
            except sm_client.exceptions.ClientError:
                time.sleep(1)
        
        header.update(trial_row)
    

        for component in components:
            component_row = trial_row

            comp_name = {k:v for k,v in component.items() if k in ("TrialComponentName", "DisplayName")}
            header.update(comp_name)

            
            while True: # retry in case of throttling
                try:
                    component = sm_client.describe_trial_component(TrialComponentName=comp_name["TrialComponentName"])
                    break
                except sm_client.exceptions.ClientError:
                    time.sleep(1)
            
            
            parameters = component["Parameters"]
            for p, v in parameters.items():
                _, parameters[p] = next(iter(parameters[p].items()))

            component_row.update(comp_name)
            component_row.update(parameters)
            header.update(parameters)
            rows.append(component_row)
            
            if component_row["TrialName"] == trial_name:
                email_html = template.render(data = component_row, 
                                             experiment = exp_name, 
                                             trial_name = trial_name)

    
    return header, rows, email_html


def lambda_handler(event, context):
    
    payload = event["input"]
    experiment = payload["experiment_name"]
    trial = payload["trial_name"]
    
    header, rows, email_html = get_experiment_data(experiment, trial)
    
    with open("/tmp/experiment_tracker.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header.keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    
    s3_client.upload_file("/tmp/experiment_tracker.csv", "szgreengarden", f"lightfm/results/experiment={experiment}/results.csv")
    
    ses_client.send_email(
    Source='szamarin@amazon.com',
    Destination={
        'ToAddresses': [
            'szamarin@amazon.com',
        ]
    },
    Message={
        'Subject': {
            'Data': 'Training workflow notification',
            'Charset': 'UTF-8'
        },
        'Body': {
            'Html': {
                'Data': email_html,
                'Charset': 'UTF-8'
            }
        }
    }
    )
    
