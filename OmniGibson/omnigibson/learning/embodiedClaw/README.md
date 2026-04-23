# embodiedClaw

## Quick Launch

Recommended default: use the bash supervisor for one episode at a time, then
either spawn a fresh master automatically or hand off to a fresh interactive
agent session.

Manual interactive master:
```bash
TASK_NAME=picking_up_trash \
TASK_ID=1 \
INSTANCE_ID=0 \
LOG_PATH=/home/user/dataset/embodiedClaw_test \
DECISION_MODE=image_state \
PAUSE_BEFORE=false \
SPAWN_MASTER=0 \
bash /home/user/codebase/B1K-DataGen/BEHAVIOR-1K/OmniGibson/omnigibson/learning/embodiedClaw/run_supervised_episode.sh
```

Automatic `claude -p` master:
```bash
TASK_NAME=picking_up_trash \
TASK_ID=1 \
INSTANCE_ID=0 \
LOG_PATH=/home/user/dataset/embodiedClaw_test_gyj \
DECISION_MODE=imageless_state \
PAUSE_BEFORE=false \
SPAWN_MASTER=1 \
bash /home/user/BEHAVIOR-1K/OmniGibson/omnigibson/learning/embodiedClaw/run_supervised_episode.sh
```

State-only prompt example:
```
Use the embodiedclaw skill as the master agent for one episode.
Handoff:
- task_name=picking_up_trash
- task_id=1
- instance_id=0
- decision_mode=imageless_state
- pause_before_decision=false
- log_path=/home/user/dataset/embodiedClaw_test
- decisions_log=/home/user/dataset/embodiedClaw_test/decisions.jsonl
```

State + images prompt example:
```
Use the embodiedclaw skill as the master agent for one episode.
Handoff:
- task_name=picking_up_trash
- task_id=1
- instance_id=0
- decision_mode=image_state
- pause_before_decision=false
- log_path=/home/user/dataset/embodiedClaw_test
- decisions_log=/home/user/dataset/embodiedClaw_test/decisions.jsonl
```
