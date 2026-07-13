#!/bin/bash
# SageMaker notebook lifecycle (on-start): THE live 24h+ runner for the pipeline.
# Installs a 6-hourly cron that runs one WM-3 cycle, and kicks one off immediately.
# This is the actual automation (no Step Functions / EventBridge in this project).
set -ex

cat > /home/ec2-user/SageMaker/wm3_cron.sh <<'C'
#!/bin/bash
export HOME=/home/ec2-user
export PATH=$HOME/.local/bin:$PATH
B=wm3-forecasts-194290773983
# uv lives in ~/.local which does NOT survive notebook stop/start -> reinstall if missing
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG=/home/ec2-user/SageMaker/wm3_cron_$TS.log
{
  cd /home/ec2-user/SageMaker/WeatherMesh-3 || exit 3
  git fetch origin && git checkout origin/main -- server
  uv pip install -q -r server/requirements-pipeline.txt
  WM3_OUTPUT_BUCKET=$B AWS_REGION=us-east-1 \
  WM3_SNS_TOPIC=arn:aws:sns:us-east-1:194290773983:wm3-alerts \
    uv run python -m server.wm3pipe.run_cycle --lead-hours 6 \
      --weights model/WeatherMesh3.pt --device cuda
} > "$LOG" 2>&1
RC=$?
aws s3 cp "$LOG" "s3://$B/_runlogs/cron_$TS.log" || true
# surface bootstrap-level failures (uv/git/etc.) that run_cycle never reached
[ $RC -ne 0 ] && aws cloudwatch put-metric-data --namespace WeatherMesh3 \
  --metric-name cycle_success --value 0 --dimensions Name=pipeline,Value=weathermesh3 \
  --region us-east-1 || true
C
chmod +x /home/ec2-user/SageMaker/wm3_cron.sh
chown ec2-user:ec2-user /home/ec2-user/SageMaker/wm3_cron.sh

# ensure the cron daemon is running so the 6-hourly schedule actually fires
systemctl enable --now crond 2>/dev/null || service crond start 2>/dev/null || true
echo "0 */6 * * * /home/ec2-user/SageMaker/wm3_cron.sh" | crontab -u ec2-user -
# run one cycle immediately on boot
sudo -u ec2-user -H nohup /home/ec2-user/SageMaker/wm3_cron.sh >/dev/null 2>&1 &
exit 0
