# Automation

The pipeline runs every six hours as a SageMaker Processing job, started by
Step Functions, triggered by EventBridge. The job runs the ECR container image.

## Deployed resources (us-east-1)

| Resource | Name |
| --- | --- |
| EventBridge rule (6h schedule) | `wm3-6h` -> state machine |
| Step Functions state machine | `wm3-cycle` (`statemachine.json`) |
| Step Functions role | `wm3-sfn-role` (CreateProcessingJob, PassRole, SNS, sync-rule events) |
| EventBridge role | `wm3-events-role` (StartExecution) |
| Container image | ECR `wm3-pipeline:latest` |
| CloudWatch dashboard | `WeatherMesh3` (`dash.json`) |
| Alarms | `wm3-cycle-failure-or-stale`, `wm3-output-invalid` -> SNS `wm3-alerts` |

## Deploy

```bash
# state machine
aws stepfunctions create-state-machine --name wm3-cycle \
  --definition file://statemachine.json --type STANDARD \
  --role-arn arn:aws:iam::194290773983:role/wm3-sfn-role --region us-east-1

# 6-hourly trigger -> state machine
aws events put-rule --name wm3-6h --schedule-expression "cron(0 0/6 * * ? *)" --region us-east-1
aws events put-targets --rule wm3-6h --region us-east-1 \
  --targets "Id=wm3-sfn,Arn=<state-machine-arn>,RoleArn=arn:aws:iam::194290773983:role/wm3-events-role"

# run one cycle now
aws stepfunctions start-execution \
  --state-machine-arn arn:aws:states:us-east-1:194290773983:stateMachine:wm3-cycle --region us-east-1
```

The state machine passes `WM3_WEIGHTS_S3` so the container pulls weights from S3, and
`WM3_OUTPUT_BUCKET`, `WM3_SNS_TOPIC` for output and alerts. It retries twice and routes
failures to SNS.

## Retired: notebook cron

`cron-onstart.sh` was the earlier runner: a SageMaker notebook lifecycle that installed a
6-hourly cron and ran the source through `uv`. It is kept for reference. The notebook is
now stopped, because Processing runs the same code on an ephemeral GPU without the idle
cost of an always-on instance.
