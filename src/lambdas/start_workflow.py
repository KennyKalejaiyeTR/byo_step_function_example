import json
import boto3
import re
from datetime import datetime

sm_client = boto3.client("sagemaker")

def lambda_handler(payload, context):
    
    payload = payload["input"]
    
    try:
        response = sm_client.create_experiment(
            ExperimentName=payload["experiment_name"],
            Description=payload["experiment_description"],
        )
        
    except sm_client.exceptions.ClientError:
        print("Experiment already exists")
    
    ts = re.sub("[\s:.]","-",str(datetime.now()))
    trial_name = f"{payload['experiment_name']}-trial-{ts}"
    
    resp = sm_client.create_trial(
        TrialName=trial_name,
        ExperimentName=payload["experiment_name"],
    )
    
    script_args = []
    for arg in ("lr", "epochs", "loss_function"):
        if payload.get(arg):
            script_args.append(f"--{arg}")
            script_args.append(str(payload[arg]))
    script_args.append("--trial_name")
    script_args.append(trial_name)
    
    payload.update({"script_args": script_args})
    payload.update({"trial_name": trial_name})
    payload.update({"training_job_name": f"{trial_name}-train"})
    payload.update({"eval_job_name": f"{trial_name}-eval"})
    payload.update({"output_model_path": f"{payload['output_model_path']}/{trial_name}"})
        
    
    return {
        'statusCode': 200,
        'body': payload
    }
